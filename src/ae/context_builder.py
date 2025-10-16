"""Build structured prompts and context packages for phase executions."""

from __future__ import annotations

import ast
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from . import prompts
from .memory.code_index.symbol_index import SymbolIndex, SymbolRecord
from .phases import PhaseName
from .tools.snippets import SnippetRequest, collect_snippets, normalize_static_findings


@dataclass(slots=True)
class ContextPackage:
    """Container for the system and user prompts supplied to the model."""

    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextBuilder:
    """Context assembly helper that builds bounded, sectioned prompt packages."""

    DEFAULT_TOKEN_BUDGET = 10_000
    _CHARS_PER_TOKEN = 4
    _MAX_SYMBOL_SNIPPETS = 10
    _MAX_SNIPPETS_PER_FILE = 3
    _MAX_FILE_PREVIEW_CHARS = 1_200
    _STATIC_FINDING_CONTEXT = 10
    _REPO_FILE_LIMIT = 0
    _REPO_TREE_LINE_LIMIT = 0
    _REPO_ALWAYS_EXCLUDE_DIRS = {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".idea",
        ".vscode",
        "node_modules",
    }
    _REPO_TOP_LEVEL_EXCLUDE_DIRS = {
        "data",
        "logs",
        "tmp",
        "build",
        "dist",
        ".venv",
        "venv",
        "env",
        ".env",
    }
    _REPO_EXCLUDE_FILE_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".cache"}
    _REPO_EXCLUDE_FILES = {".DS_Store"}
    _REQUEST_FILE_HINT_KEYS = {
        "recent_changes",
        "changed_paths",
        "touched_files",
        "touched_paths",
        "suspect_files",
    }
    _REQUEST_TEST_ID_KEYS = {"failing_tests"}
    _PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-\\/]+\.py")

    def __init__(
        self,
        *,
        policy_text: str | None = None,
        system_preamble: str | None = None,
        repo_root: Path | str | None = None,
        data_root: Path | str | None = None,
        logs_root: Path | str | None = None,
        token_budget: int | None = None,
        guidance: Sequence[str] | None = None,
    ) -> None:
        self._policy_text = policy_text or ""
        self._system_preamble = system_preamble or (
            "You are Agentic Engineer, a meticulous coding assistant. "
            "Always return well-formed JSON that matches the expected schema."
        )
        self._token_budget = token_budget or self.DEFAULT_TOKEN_BUDGET
        self._repo_root = self._resolve_repo_root(repo_root)
        self._data_root = self._resolve_data_root(data_root, self._repo_root)
        self._logs_root = self._resolve_logs_root(logs_root, self._data_root, self._repo_root)
        self._index_root = self._data_root / "index"
        self._symbol_index_path = self._index_root / "symbols.json"
        self._symbol_index: SymbolIndex | None = None
        self._symbol_index_loaded = False
        self._repo_file_cache: list[tuple[str, bool]] | None = None
        cleaned_guidance: list[str] = []
        for item in guidance or ():
            if isinstance(item, str):
                trimmed = item.strip()
                if trimmed:
                    cleaned_guidance.append(trimmed)
        self._guidance_lines = tuple(cleaned_guidance)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        repo_root: Path | None = None,
    ) -> ContextBuilder:
        """Instantiate a builder using project configuration values."""
        policy_path = (
            (config.get("policy") or {}).get("capsule_path")
            if isinstance(config.get("policy"), Mapping)
            else None
        )

        repo_root_path = repo_root or cls._infer_repo_root(config)
        data_root_path = cls._infer_data_root(config, repo_root_path)
        logs_root_path = cls._infer_logs_root(config, repo_root_path, data_root_path)

        policy_text = None
        if policy_path:
            policy_file = Path(policy_path)
            if not policy_file.is_absolute():
                policy_file = (repo_root_path / policy_file).resolve()
            if policy_file.exists():
                policy_text = policy_file.read_text(encoding="utf-8").strip()

        context_config = config.get("context") if isinstance(config.get("context"), Mapping) else {}
        token_budget = context_config.get("token_budget")
        if not isinstance(token_budget, int) or token_budget <= 0:
            token_budget = None

        system_preamble = context_config.get("system_preamble")
        if not isinstance(system_preamble, str):
            system_preamble = None

        guidance_values: list[str] = []
        if "guidance" in context_config:
            guidance_raw = context_config.get("guidance")
            if isinstance(guidance_raw, str):
                trimmed = guidance_raw.strip()
                if trimmed:
                    guidance_values.append(trimmed)
            elif isinstance(guidance_raw, Sequence):
                for entry in guidance_raw:
                    if isinstance(entry, str):
                        trimmed = entry.strip()
                        if trimmed:
                            guidance_values.append(trimmed)

        return cls(
            policy_text=policy_text,
            system_preamble=system_preamble,
            repo_root=repo_root_path,
            data_root=data_root_path,
            logs_root=logs_root_path,
            token_budget=token_budget,
            guidance=guidance_values or None,
        )

    def build(self, phase: str, request: Any) -> ContextPackage:
        """Assemble a bounded context package for the given phase request."""
        request_data = self._coerce_data(request)
        snippets = self._extract_snippets(request_data)
        symbol_hints = self._extract_symbol_hints(request_data)
        file_hints = self._extract_file_hints(request_data)
        payload_view = self._prepare_payload_view(request_data, snippets)

        sections = self._collect_sections(
            phase,
            payload_view,
            symbol_hints=symbol_hints,
            file_hints=file_hints,
            snippets=snippets,
        )

        instructions_text = prompts.render_json_instruction(self._repo_root)
        instruction_tokens = self._estimate_tokens(instructions_text)
        budget_for_sections = max(self._token_budget - instruction_tokens, 0)

        prompt_body, section_meta, token_estimate = self._assemble_prompt(sections, budget=budget_for_sections)

        remaining_budget = max(self._token_budget - token_estimate, 0)
        instructions_entry = {
            "label": "instructions",
            "included": False,
            "truncated": False,
            "tokens": 0,
        }
        instruction_text_used = ""
        if instruction_tokens <= remaining_budget:
            instruction_text_used = instructions_text
            instructions_entry["included"] = True
            instructions_entry["tokens"] = instruction_tokens
        elif remaining_budget > 0:
            instruction_text_used = self._truncate_text_to_tokens(instructions_text, remaining_budget)
            if instruction_text_used:
                instructions_entry["included"] = True
                instructions_entry["truncated"] = True
                instructions_entry["tokens"] = self._estimate_tokens(instruction_text_used)

        if instruction_text_used:
            token_estimate += instructions_entry["tokens"]
            if prompt_body:
                user_prompt = f"{prompt_body.strip()}\n\n{instruction_text_used.strip()}"
            else:
                user_prompt = instruction_text_used.strip()
        else:
            user_prompt = prompt_body.strip()

        section_meta.append(instructions_entry)

        system_prompt = self._compose_system_prompt(phase)
        metadata: dict[str, Any] = {
            "phase": phase,
            "request": payload_view,
            "token_budget": self._token_budget,
            "token_estimate": min(token_estimate, self._token_budget),
            "sections": section_meta,
        }
        if symbol_hints:
            metadata["symbol_hints"] = sorted(symbol_hints)
        if file_hints:
            metadata["file_hints"] = sorted(file_hints)
        if self._guidance_lines:
            metadata["guidance"] = list(self._guidance_lines)
        if snippets:
            metadata["snippets"] = [
                {
                    "path": entry["path"],
                    "start_line": entry.get("start_line"),
                    "end_line": entry.get("end_line"),
                    "reason": entry.get("reason"),
                }
                for entry in snippets
            ]

        return ContextPackage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=metadata,
        )

    def _compose_system_prompt(self, phase: str) -> str:
        if not self._policy_text:
            return f"{self._system_preamble} You are currently executing the {phase} phase."

        return (
            f"{self._system_preamble} You are currently executing the {phase} phase.\n\n"
            "## Repository Policy Capsule\n"
            f"{self._policy_text}"
        )

    @property
    def repo_root(self) -> Path:
        """Return the repository root used for context assembly."""
        return self._repo_root

    @property
    def data_root(self) -> Path:
        """Return the data root directory configured for context assembly."""
        return self._data_root

    @property
    def logs_root(self) -> Path:
        """Return the logs directory derived from the data root."""
        return self._logs_root

    def _coerce_data(self, request: Any) -> dict[str, Any]:
        """Normalize request objects to dictionaries."""
        if is_dataclass(request):
            return asdict(request)
        if isinstance(request, Mapping):
            return dict(request)
        return {"payload": request}

    def _extract_snippets(self, request_data: dict[str, Any]) -> list[dict[str, Any]]:
        raw = request_data.get("snippets")

        snippets: list[dict[str, Any]] = []
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            for item in raw:
                if not isinstance(item, Mapping):
                    continue
                path = item.get("path")
                content = item.get("content")
                if not isinstance(path, str) or not isinstance(content, str):
                    continue
                snippet_entry: dict[str, Any] = {
                    "path": path.strip(),
                    "content": content,
                }
                start_line = item.get("start_line")
                end_line = item.get("end_line")
                reason = item.get("reason")
                if isinstance(start_line, int):
                    snippet_entry["start_line"] = start_line
                if isinstance(end_line, int):
                    snippet_entry["end_line"] = end_line
                if isinstance(reason, str) and reason.strip():
                    snippet_entry["reason"] = reason.strip()
                snippets.append(snippet_entry)

        snippet_keys = {
            (entry["path"], entry.get("start_line"), entry.get("end_line")) for entry in snippets
        }

        code_requests_raw = request_data.get("code_requests")
        static_findings_raw = request_data.get("static_findings")

        requests: list[SnippetRequest] = []
        if isinstance(code_requests_raw, Sequence) and not isinstance(
            code_requests_raw,
            (str, bytes, bytearray),
        ):
            for item in code_requests_raw:
                if not isinstance(item, Mapping):
                    continue
                path = item.get("path")
                if not isinstance(path, str):
                    continue
                start_line = item.get("start_line")
                end_line = item.get("end_line")
                surround = item.get("surround")
                reason = item.get("reason")
                requests.append(
                    SnippetRequest(
                        path=path.strip(),
                        start_line=start_line if isinstance(start_line, int) and start_line > 0 else None,
                        end_line=end_line if isinstance(end_line, int) and end_line > 0 else None,
                        surround=surround if isinstance(surround, int) and surround >= 0 else None,
                        reason=reason if isinstance(reason, str) and reason.strip() else None,
                    )
                )

        if requests or static_findings_raw:
            normalized_findings = normalize_static_findings(static_findings_raw or [])
            extra_snippets = collect_snippets(
                self._repo_root,
                requests,
                static_findings=normalized_findings or None,
                finding_context=self._STATIC_FINDING_CONTEXT,
            )
            for snippet in extra_snippets:
                key = (snippet.path, snippet.start_line, snippet.end_line)
                if key in snippet_keys:
                    continue
                snippet_entry = {
                    "path": snippet.path,
                    "content": snippet.content,
                    "start_line": snippet.start_line,
                    "end_line": snippet.end_line,
                }
                if snippet.reason:
                    snippet_entry["reason"] = snippet.reason
                snippets.append(snippet_entry)
                snippet_keys.add(key)

        return snippets

    def _prepare_payload_view(
        self,
        request_data: dict[str, Any],
        snippets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload_view = dict(request_data)
        if snippets:
            payload_view["snippets"] = [
                {
                    "path": entry["path"],
                    "start_line": entry.get("start_line"),
                    "end_line": entry.get("end_line"),
                    "reason": entry.get("reason"),
                }
                for entry in snippets
            ]
        elif "snippets" in payload_view:
            payload_view.pop("snippets", None)
        return payload_view

    def _collect_sections(
        self,
        phase: str,
        request_data: dict[str, Any],
        *,
        symbol_hints: set[str],
        file_hints: set[str],
        snippets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if phase.lower() == "diagnose":
            return self._collect_diagnose_sections(
                phase,
                request_data,
                symbol_hints=symbol_hints,
                file_hints=file_hints,
                snippets=snippets,
            )

        sections: list[dict[str, Any]] = [
            {
                "label": "phase_header",
                "text": f"# Phase: {phase}",
                "priority": 0,
                "allow_truncate": False,
            },
            {
                "label": "phase_brief",
                "text": self._render_phase_brief(phase),
                "priority": 5,
                "allow_truncate": False,
            },
        ]

        summary = self._render_task_summary(request_data)
        if summary:
            sections.append(
                {
                    "label": "task_summary",
                    "text": summary,
                    "priority": 10,
                    "allow_truncate": False,
                }
            )

        sections.extend(self._render_list_sections(request_data))
        sections.extend(self._render_text_sections(request_data))

        if self._guidance_lines:
            guidance_text = prompts.render_project_guidance(self._guidance_lines)
        else:
            guidance_text = ""
        if guidance_text:
            sections.append(
                {
                    "label": "project_guidance",
                    "text": guidance_text,
                    "priority": 8,
                    "allow_truncate": False,
                }
            )

        snippet_sections = self._render_snippet_sections(snippets)
        sections.extend(snippet_sections)

        payload = self._render_payload(request_data)
        sections.append(
            {
                "label": "request_payload",
                "text": payload,
                "priority": 60,
                "allow_truncate": True,
            }
        )

        for label, text in self._response_contract_sections(phase, request_data):
            sections.append(
                {
                    "label": label,
                    "text": text,
                    "priority": 65,
                    "allow_truncate": False,
                }
            )

        code_sections = self._render_code_sections(symbol_hints, file_hints)
        if code_sections:
            sections.append(
                {
                    "label": "code_header",
                    "text": "## Code Context",
                    "priority": 70,
                    "allow_truncate": False,
                }
            )
            sections.extend(code_sections)

        python_outline_section = self._render_workspace_python_outline_section()
        if python_outline_section:
            sections.append(
                {
                    "label": "workspace_python_outline",
                    "text": python_outline_section,
                    "priority": 68,
                    "allow_truncate": False,
                }
            )

        requirements_section = self._render_requirements_section()
        if requirements_section:
            sections.append(
                {
                    "label": "workspace_requirements",
                    "text": requirements_section,
                    "priority": 69,
                    "allow_truncate": False,
                }
            )

        repo_tree_section = self._render_repo_tree_section()
        if repo_tree_section:
            sections.append(
                {
                    "label": "repo_tree",
                    "text": repo_tree_section,
                    "priority": 72,
                    "allow_truncate": False,
                }
            )

        return sections

    def _collect_diagnose_sections(
        self,
        phase: str,
        request_data: dict[str, Any],
        *,
        symbol_hints: set[str],
        file_hints: set[str],
        snippets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Assemble a compact prompt profile tuned for the diagnose phase."""

        sections: list[dict[str, Any]] = []

        brief = self._render_phase_brief(phase)
        brief_single_line = " ".join(brief.split()) if brief else ""
        header_lines = [f"# Phase: {phase}"]
        if brief_single_line:
            header_lines.append(brief_single_line)
        sections.append(
            {
                "label": "phase_header",
                "text": "\n".join(header_lines),
                "priority": 0,
                "allow_truncate": False,
            }
        )

        summary = self._render_task_summary(request_data)
        if summary:
            sections.append(
                {
                    "label": "task_summary",
                    "text": summary,
                    "priority": 5,
                    "allow_truncate": False,
                }
            )

        compact_lines: list[str] = []
        compact_fields = [
            ("goal", "Goal"),
            ("diff_goal", "Diff Goal"),
            ("reason", "Reason"),
        ]
        for key, label in compact_fields:
            value = request_data.get(key)
            if isinstance(value, str) and value.strip():
                compact_lines.append(f"- {label}: {value.strip()}")

        list_field_map = {
            "constraints": "Constraints",
            "questions": "Questions",
            "open_questions": "Open Questions",
            "failing_tests": "Failing Tests",
            "recent_changes": "Recent Changes",
            "suggested_changes": "Suggested Changes",
            "blockers": "Blockers",
            "touched_files": "Relevant Files",
            "iteration_guidance": "Iteration Guidance",
        }
        for key, label in list_field_map.items():
            values = self._coerce_to_strings(request_data.get(key))
            if not values:
                continue
            if len(values) == 1 and len(values[0]) <= 160:
                compact_lines.append(f"- {label}: {values[0]}")
                continue
            preview = " | ".join(value.strip() for value in values[:5])
            if len(values) > 5:
                preview += f" | (+{len(values) - 5} more)"
            compact_lines.append(f"- {label}: {preview}")

        if compact_lines:
            sections.append(
                {
                    "label": "diagnose_overview",
                    "text": "## Diagnose Overview\n" + "\n".join(compact_lines),
                    "priority": 12,
                    "allow_truncate": False,
                }
            )

        logs_value = request_data.get("logs")
        if isinstance(logs_value, str) and logs_value.strip():
            sections.append(
                {
                    "label": "text:logs",
                    "text": f"## Logs\n```text\n{logs_value.strip()}\n```",
                    "priority": 20,
                    "allow_truncate": True,
                }
            )

        snippet_sections = self._render_snippet_sections(snippets)
        sections.extend(snippet_sections)

        payload = self._render_payload(request_data)
        sections.append(
            {
                "label": "request_payload",
                "text": payload,
                "priority": 60,
                "allow_truncate": True,
            }
        )

        code_sections = self._render_code_sections(symbol_hints, file_hints, max_sections=5)
        if code_sections:
            sections.append(
                {
                    "label": "code_header",
                    "text": "## Code Context (condensed)",
                    "priority": 65,
                    "allow_truncate": True,
                }
            )
            sections.extend(code_sections)

        return sections

    def _render_phase_brief(self, phase: str) -> str:
        return prompts.render_phase_brief(phase)

    def _render_task_summary(self, request_data: dict[str, Any]) -> str:
        bullets: list[str] = []
        task_id = request_data.get("task_id")
        plan_id = request_data.get("plan_id")
        goal = request_data.get("goal")
        diff_goal = request_data.get("diff_goal")
        reason = request_data.get("reason")

        if isinstance(task_id, str) and task_id:
            bullets.append(f"- Task ID: {task_id}")
        if isinstance(plan_id, str) and plan_id:
            bullets.append(f"- Plan ID: {plan_id}")
        if isinstance(goal, str) and goal:
            bullets.append(f"- Goal: {goal}")
        if isinstance(diff_goal, str) and diff_goal:
            bullets.append(f"- Diff Goal: {diff_goal}")
        if isinstance(reason, str) and reason:
            bullets.append(f"- Reason: {reason}")

        if not bullets:
            return ""
        return "## Task Summary\n" + "\n".join(bullets)

    def _render_list_sections(self, request_data: dict[str, Any]) -> list[dict[str, Any]]:
        list_fields = {
            "constraints": ("Constraints", 20),
            "questions": ("Questions", 20),
            "open_questions": ("Open Questions", 20),
            "test_plan": ("Test Plan", 25),
            "notes": ("Notes", 25),
            "violations": ("Policy Violations", 30),
            "failing_tests": ("Failing Tests", 30),
            "recent_changes": ("Recent Changes", 30),
            "suggested_changes": ("Suggested Changes", 30),
            "blockers": ("Blockers", 30),
            "proposed_interfaces": ("Proposed Interfaces", 25),
            "follow_up": ("Follow Up Items", 35),
            "touched_files": ("Relevant Files", 25),
        }

        sections: list[dict[str, Any]] = []
        for key, (title, priority) in list_fields.items():
            values = self._coerce_to_strings(request_data.get(key))
            if not values:
                continue
            bullet_lines = "\n".join(f"- {item}" for item in values)
            sections.append(
                {
                    "label": f"list:{key}",
                    "text": f"## {title}\n{bullet_lines}",
                    "priority": priority,
                    "allow_truncate": False,
                }
            )
        return sections

    def _render_text_sections(self, request_data: dict[str, Any]) -> list[dict[str, Any]]:
        text_fields = {
            "context": ("Additional Context", 22, "text"),
            "logs": ("Logs", 65, "text"),
            "current_diff": ("Current Diff", 65, "diff"),
        }
        sections: list[dict[str, Any]] = []
        for key, (title, priority, kind) in text_fields.items():
            value = request_data.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            content = value.strip()
            if kind == "diff":
                content = f"```diff\n{content}\n```"
            sections.append(
                {
                    "label": f"text:{key}",
                    "text": f"## {title}\n{content}",
                    "priority": priority,
                    "allow_truncate": False,
                }
            )
        return sections

    def _render_snippet_sections(self, snippets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        for index, entry in enumerate(snippets):
            path = entry.get("path")
            content = entry.get("content") or ""
            if not isinstance(path, str) or not content.strip():
                continue
            start_line = entry.get("start_line")
            end_line = entry.get("end_line")
            reason = entry.get("reason")
            header_parts = [f"## Snippet: {path}"]
            if isinstance(start_line, int) and isinstance(end_line, int):
                header_parts.append(f"(lines {start_line}-{end_line})")
            elif isinstance(start_line, int):
                header_parts.append(f"(from line {start_line})")
            elif isinstance(end_line, int):
                header_parts.append(f"(up to line {end_line})")
            header = " ".join(header_parts)
            snippet_body = content.rstrip()
            block = f"{header}\n```python\n{snippet_body}\n```"
            if isinstance(reason, str) and reason.strip():
                block = f"{block}\n- Reason: {reason.strip()}"
            sections.append(
                {
                    "label": f"snippet:{path}#{index}",
                    "text": block,
                    "priority": 55,
                    "allow_truncate": False,
                }
            )
        return sections

    def _render_repo_tree_section(self) -> str:
        entries = self._repo_file_list()
        if not entries:
            return ""

        tree_root: dict[str, object] = {}
        for path, is_dir in entries:
            trimmed = path.rstrip("/")
            if not trimmed:
                continue
            parts = trimmed.split("/")
            node = tree_root
            for index, part in enumerate(parts):
                is_last = index == len(parts) - 1
                if not is_last:
                    child = node.setdefault(part, {})
                    if not isinstance(child, dict):
                        child = {}
                        node[part] = child
                    node = child
                    continue
                if is_dir:
                    current = node.get(part)
                    if not isinstance(current, dict):
                        node[part] = {}
                else:
                    node.setdefault(part, None)

        lines: list[str] = ["."]
        line_limit = int(self._REPO_TREE_LINE_LIMIT or 0)
        truncated = False

        def render(node: dict[str, object], prefix: str) -> None:
            nonlocal truncated
            entries = sorted(
                node.items(),
                key=lambda entry: (0 if isinstance(entry[1], dict) else 1, entry[0]),
            )
            for position, (name, child) in enumerate(entries):
                if line_limit and len(lines) >= line_limit:
                    truncated = True
                    return
                is_last = position == len(entries) - 1
                connector = "└── " if is_last else "├── "
                suffix = "/" if isinstance(child, dict) else ""
                lines.append(f"{prefix}{connector}{name}{suffix}")
                if isinstance(child, dict) and child:
                    continuation = "    " if is_last else "│   "
                    render(child, prefix + continuation)
                    if truncated:
                        return

        render(tree_root, "")

        header = "## Repository Tree"
        if truncated:
            visible = max(line_limit - 1, 0)
            header += f" (truncated to first {visible} entries)"
        tree_body = "\n".join(lines)
        return f"{header}\n{tree_body}"

    def _render_workspace_python_outline_section(self) -> str:
        entries = self._repo_file_list()
        python_files = [path for path, is_dir in entries if not is_dir and path.endswith(".py")]
        if not python_files:
            return ""

        index = self._load_symbol_index()
        outlines: list[str] = []
        for path_key in python_files:
            records: Sequence[SymbolRecord] = ()
            if index is not None:
                records = index.symbols_for_path(path_key)
            outline = self._format_python_outline(path_key, records)
            if not outline:
                outline = self._fallback_python_outline(path_key)
            if outline:
                outlines.append(outline)

        if not outlines:
            return ""
        body = "\n\n".join(outlines)
        return f"## Workspace Python Outline\n{body}"

    def _fallback_python_outline(self, path_key: str) -> str:
        file_path = self._repo_root / path_key
        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError:
            return ""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ""

        lines: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                signature = self._format_class_signature(node)
                lines.append(f"- {signature}")
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_sig = self._format_function_signature(member)
                        lines.append(f"  - {method_sig}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                signature = self._format_function_signature(node)
                lines.append(f"- {signature}")

        if not lines:
            return ""
        body = "\n".join(lines)
        return f"### {path_key} :: symbol outline\n{body}"

    def _format_class_signature(self, node: ast.ClassDef) -> str:
        bases = [self._safe_ast_unparse(base).strip() for base in node.bases if base is not None]
        bases = [base for base in bases if base]
        signature = f"class {node.name}"
        if bases:
            signature = f"{signature}({', '.join(bases)})"
        return signature

    def _format_function_signature(self, node: ast.AST) -> str:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args_text = self._safe_ast_unparse(node.args).strip()
            signature = f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}{node.name}("
            signature += args_text
            signature += ")"
            if node.returns is not None:
                return_text = self._safe_ast_unparse(node.returns).strip()
                if return_text:
                    signature = f"{signature} -> {return_text}"
            return signature
        return ""

    @staticmethod
    def _safe_ast_unparse(node: ast.AST | None) -> str:
        if node is None:
            return ""
        unparse = getattr(ast, "unparse", None)
        if callable(unparse):
            try:
                return unparse(node)
            except Exception:
                return ""
        return ""

    def _render_requirements_section(self) -> str:
        requirement_paths = self._find_requirements_files()
        if not requirement_paths:
            return ""

        sections: list[str] = []
        for path_str in requirement_paths:
            file_path = self._repo_root / path_str
            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError:
                continue
            text = content.rstrip("\n")
            if not text:
                text = "(empty file)"
            sections.append(f"### {path_str}\n```text\n{text}\n```")
        if not sections:
            return ""
        return "## Workspace Requirements\n" + "\n\n".join(sections)

    def _find_requirements_files(self) -> list[str]:
        entries = self._repo_file_list()
        candidates: set[str] = set()
        for path, is_dir in entries:
            if is_dir:
                continue
            name = Path(path).name.lower()
            if name == "requirements.txt" or (name.startswith("requirements") and name.endswith(".txt")):
                candidates.add(path)
        return sorted(candidates)

    def _repo_file_list(self) -> list[tuple[str, bool]]:
        if self._repo_file_cache is not None:
            return self._repo_file_cache

        root = self._repo_root
        directories: set[str] = set()
        files: set[str] = set()

        for current_root, dirs, filenames in os.walk(root):
            rel_dir = Path(current_root).relative_to(root)
            filtered_dirs = []
            for d in dirs:
                if d in self._REPO_ALWAYS_EXCLUDE_DIRS:
                    continue
                if rel_dir == Path(".") and d in self._REPO_TOP_LEVEL_EXCLUDE_DIRS:
                    continue
                if d.startswith(".") and d not in {".config"}:
                    continue
                filtered_dirs.append(d)
                relative_dir = (rel_dir / d) if rel_dir != Path(".") else Path(d)
                directories.add(relative_dir.as_posix())
            dirs[:] = filtered_dirs
            for filename in filenames:
                if filename in self._REPO_EXCLUDE_FILES or filename.startswith("."):
                    continue
                suffix = Path(filename).suffix
                if suffix in self._REPO_EXCLUDE_FILE_SUFFIXES:
                    continue
                relative_path = Path(filename) if rel_dir == Path(".") else rel_dir / filename
                files.add(relative_path.as_posix())

        combined: list[tuple[str, bool]] = [
            (path if path.endswith("/") else f"{path}/", True) for path in directories
        ]
        combined.extend((path, False) for path in files)
        combined.sort(key=lambda entry: entry[0])
        if self._REPO_FILE_LIMIT and len(combined) > self._REPO_FILE_LIMIT:
            combined = combined[: self._REPO_FILE_LIMIT]
        self._repo_file_cache = combined
        return self._repo_file_cache

    def _plan_response_contract(self, request_data: dict[str, Any]) -> str:
        goal = request_data.get("goal")
        goal_line = f"- Address the goal '{goal}' when describing the plan." if isinstance(goal, str) and goal else None
        bullet_points = [
            "- Return well-formed JSON that satisfies the planner response schema.",
            "- Populate `plan_summary` with a concise overview of the approach.",
            "- Emit at least five ordered tasks under `tasks`, each with a unique `id`; there is no upper limit.",
            "- Keep tasks narrowly scoped so each addresses one deliverable or intermediate milestone without becoming bulky.",
            "- Split complex deliverables across multiple tasks when needed so the agent always works on manageable increments.",
            "- Each task must include `title`, `summary`, `depends_on`, `constraints`, `deliverables`, `acceptance_criteria`, and `metadata`.",
            "- Populate `metadata.touched_files` and `metadata.validation_commands` with non-empty string arrays for every task.",
            "- Reference other tasks in `depends_on` using IDs or titles defined in this plan only.",
        ]
        if goal_line:
            bullet_points.insert(1, goal_line)
        content = "\n".join(bullet_points)
        return f"## Response Contract\n{content}"

    def _plan_analysis_response_contract(self, request_data: dict[str, Any]) -> str:
        deliverables = self._coerce_to_strings(request_data.get("deliverables"))
        bullet_points = [
            "## Analysis Response Contract",
            "- Return a JSON object that satisfies the PlannerAnalysis schema.",
            "- Identify key repository components with `name`, `summary`, `primary_paths`, `key_symbols`, and `related_tests`.",
            "- Populate `deliverable_map` to link each deliverable to the components or files that address it.",
            "- List existing tests, scripts, or commands that can validate future work under `validation_assets`.",
            "- Capture outstanding unknowns in `knowledge_gaps` and risks in `risks`.",
        ]
        if deliverables:
            joined = ", ".join(deliverables)
            bullet_points.insert(
                3,
                f"- Ensure `deliverable_map` includes entries for: {joined}.",
            )
        bullet_points.append("- Keep entries concise and focused on context that informs downstream planning.")
        return "\n".join(bullet_points)

    def _plan_critic_response_contract(self) -> str:
        bullet_points = [
            "## Plan Critic Response Contract",
            "- Return a JSON object that satisfies the PlannerCritique schema.",
            "- Use `blockers` for issues that prevent execution until resolved.",
            "- Record detailed findings in `issues`, tagging severity as `must_fix`, `should_fix`, or `nice_to_have`.",
            "- Populate `missing_deliverables` and `dependency_conflicts` when the plan omits deliverables or has ordering issues.",
            "- Offer non-blocking improvements in `recommendations`.",
            "- Reference tasks by their identifiers or titles whenever possible.",
        ]
        return "\n".join(bullet_points)

    def _plan_adjust_response_contract(self) -> str:
        example = {
            "adjustments": [
                {
                    "action": "Add deterministic plan enforcement",
                    "summary": "Introduce explicit schema contract in prompts.",
                    "details": "Document required keys and provide JSON examples.",
                    "priority": "high",
                    "rationale": "Model outputs deviated from expected schema.",
                    "notes": ["Ensure existing tooling remains backward compatible."],
                }
            ],
            "new_tasks": ["Update prompt builder to include explicit schema reminders."],
            "risks": ["Potential increase in prompt length; monitor token usage."],
            "notes": ["Roll out alongside schema-tolerant parsers."],
        }
        example_json = json.dumps(example, indent=2, sort_keys=True)
        contract_lines = [
            "## Plan Adjust Response Contract",
            "- Return a JSON object with the following keys:",
            "  - `adjustments`: array of objects describing proposed changes. Each object may include `action`, `summary`, `details`, `rationale`, `priority`, `id`, and `notes` (string or list).",
            "  - `new_tasks`: array of strings naming follow-up tasks to add to the plan.",
            "  - `risks`: array of strings summarising any newly identified risks.",
            "  - `notes`: array of strings with any additional context.",
            "- Omit fields only when they are not applicable; never return nulls.",
            "- Keep values concise and actionable.",
            "### Example Response",
            "```json",
            example_json,
            "```",
        ]
        return "\n".join(contract_lines)

    def _implement_response_contract(self, request_data: dict[str, Any]) -> str:
        touched_files = self._coerce_to_strings(request_data.get("touched_files"))
        file_hint_line = None
        if touched_files:
            files_display = ", ".join(touched_files)
            file_hint_line = (
                f"- Focus structured updates on the relevant files ({files_display}) unless additional edits are justified."
            )
        structured_only = bool(request_data.get("structured_edits_only"))
        contract_lines = [
            "## Implement Response Contract",
            "- Return well-formed JSON matching the implement phase response schema.",
            "- Populate `files` with any complete file payloads that should exist after this change. Each entry must include `path` and `content`, plus optional `encoding` and `executable` flags.",
            "- Use `edits` for targeted updates. Each entry must provide `path`, an `action` of `replace`, `insert`, or `delete`, optional `start_line`/`end_line` bounds, and `content` for inserted or replacement text.",
            "- Keep `summary` to a single concise sentence describing the applied change; omit extended rationale or command walkthroughs.",
        ]
        if file_hint_line:
            contract_lines.append(file_hint_line)
        if structured_only:
            contract_lines.append(
                "- Leave `diff` empty; host tooling will generate patches deterministically from the structured payload."
            )
        else:
            contract_lines.append(
                "- Leave `diff` empty when the structured payload is complete; only include a fallback unified diff if a change cannot be expressed structurally."
            )
        contract_lines.append(
            "- Provide a `no_op_reason` only when absolutely no changes are required; leave it blank when supplying structured updates."
        )
        contract_lines.append(
            "- When more context is required, populate `code_requests` with objects containing `path` and optional `start_line`, `end_line`, `surround`, and `reason`. The orchestrator will retrieve those snippets and rerun this phase."
        )
        contract_lines.append("- Provide `test_commands` that can be executed from the repository root.")
        contract_lines.append(
            "- Only populate `follow_up` when mandatory clean-up or verification remains; otherwise return an empty list."
        )
        contract_lines.append(
            "- Do not create auxiliary documentation or command-list files unless the task explicitly requests them."
        )
        return "\n".join(contract_lines)

    def _fix_violations_response_contract(self, request_data: dict[str, Any]) -> str:
        suspect_files = self._coerce_to_strings(request_data.get("suspect_files"))
        contract_lines = [
            "## Fix Violations Response Contract",
            "- Return well-formed JSON matching the fix_violations phase response schema.",
            "- Use `files` to supply complete file payloads for any files that should be rewritten; include `path`, `content`, and optional `encoding` or `executable` flags.",
            "- Use `edits` for targeted updates with `path`, an `action` of `replace`, `insert`, or `delete`, optional `start_line`/`end_line`, and `content` for insert or replace operations.",
            "- Prefer the structured fields and leave `patch` empty unless a fallback unified diff is absolutely necessary.",
            "- Only populate `no_op_reason` when you are intentionally reporting that no fixes are required; otherwise leave it blank and return the necessary structured updates.",
            "- Populate `touched_files` with the relative paths you changed, supply concise `rationale` entries, and list required next steps in `follow_up` (e.g. rerun static gates).",
        ]
        if suspect_files:
            files_display = ", ".join(suspect_files)
            contract_lines.insert(
                2,
                f"- Prioritise fixes in the flagged files ({files_display}) unless the violations require broader changes.",
            )
        return "\n".join(contract_lines)

    def _response_contract_sections(
        self,
        phase: str,
        request_data: dict[str, Any],
    ) -> list[tuple[str, str]]:
        if phase == PhaseName.PLAN.value:
            stage = ""
            if isinstance(request_data, Mapping):
                stage_value = request_data.get("stage") or request_data.get("_planner_pass")
                if isinstance(stage_value, str):
                    stage = stage_value.lower()
            if stage == "analysis":
                return [("plan_analysis_response_contract", self._plan_analysis_response_contract(request_data))]
            if stage.startswith("critic"):
                return [("plan_critic_response_contract", self._plan_critic_response_contract())]
            return [("plan_response_contract", self._plan_response_contract(request_data))]
        if phase == PhaseName.PLAN_ADJUST.value:
            return [("plan_adjust_response_contract", self._plan_adjust_response_contract())]
        if phase == PhaseName.IMPLEMENT.value:
            return [("implement_response_contract", self._implement_response_contract(request_data))]
        if phase == PhaseName.FIX_VIOLATIONS.value:
            return [("fix_violations_response_contract", self._fix_violations_response_contract(request_data))]
        return []

    def _render_code_sections(
        self,
        symbol_hints: set[str],
        file_hints: set[str],
        max_sections: int | None = None,
    ) -> list[dict[str, Any]]:
        index = self._load_symbol_index()
        if index is None:
            return []

        sections: list[dict[str, Any]] = []
        outline_sections = self._render_symbol_outline_sections(index, file_hints)
        if outline_sections:
            sections.extend(outline_sections)

        selected: list[SymbolRecord] = []
        seen: set[tuple[str, int, int]] = set()
        per_path: dict[str, int] = {}

        def add_record(record: SymbolRecord) -> None:
            key = (record.path, record.start, record.end)
            if key in seen:
                return
            path_count = per_path.get(record.path, 0)
            if path_count >= self._MAX_SNIPPETS_PER_FILE:
                return
            if len(selected) >= self._MAX_SYMBOL_SNIPPETS:
                return
            selected.append(record)
            seen.add(key)
            per_path[record.path] = path_count + 1

        for symbol in sorted(symbol_hints):
            for record in index.query(symbol):
                add_record(record)
                if len(selected) >= self._MAX_SYMBOL_SNIPPETS:
                    break
            if len(selected) >= self._MAX_SYMBOL_SNIPPETS:
                break

        for file_path in sorted(file_hints):
            if len(selected) >= self._MAX_SYMBOL_SNIPPETS:
                break
            records = index.symbols_for_path(file_path)
            if not records:
                continue
            top_level = [item for item in records if item.kind != "method"]
            candidates = top_level or records
            for record in candidates:
                add_record(record)
                if len(selected) >= self._MAX_SYMBOL_SNIPPETS:
                    break

        selected.sort(key=lambda record: (record.path, record.start))

        covered_files: set[str] = set()
        for record in selected:
            snippet = self._symbol_snippet(record)
            if not snippet:
                continue
            covered_files.add(record.path)
            sections.append(
                {
                    "label": f"code:symbol:{record.path}#{record.name}",
                    "text": snippet,
                    "priority": 80,
                    "allow_truncate": False,
                }
            )

        for file_path in sorted(file_hints):
            if file_path in covered_files:
                continue
            snippet = self._file_snippet(file_path)
            if not snippet:
                continue
            sections.append(
                {
                    "label": f"code:file:{file_path}",
                    "text": snippet,
                    "priority": 90,
                    "allow_truncate": False,
                }
            )

        if max_sections is not None and len(sections) > max_sections:
            return sections[:max_sections]
        return sections

    def _render_symbol_outline_sections(
        self,
        index: SymbolIndex,
        file_hints: set[str],
    ) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        for file_path in sorted(file_hints):
            if not file_path.endswith(".py"):
                continue
            records = index.symbols_for_path(file_path)
            outline = self._format_python_outline(file_path, records)
            if not outline:
                continue
            sections.append(
                {
                    "label": f"code:outline:{file_path}",
                    "text": outline,
                    "priority": 75,
                    "allow_truncate": False,
                }
            )
        return sections

    def _format_python_outline(
        self,
        path_key: str,
        records: Sequence[SymbolRecord],
    ) -> str:
        relevant = [record for record in records if record.kind in {"class", "function", "method"}]
        if not relevant:
            return ""

        ordered = sorted(relevant, key=lambda record: (record.start, record.end))
        module_prefix = self._module_prefix_for_path(path_key)

        methods_by_owner: dict[str, list[tuple[int, str]]] = {}
        loose_methods: list[tuple[int, str]] = []
        for record in ordered:
            if record.kind != "method":
                continue
            owner = record.qualified_name.rsplit(".", 1)[0] if "." in record.qualified_name else ""
            entry = (record.start, record.signature)
            if owner:
                methods_by_owner.setdefault(owner, []).append(entry)
            else:
                loose_methods.append(entry)

        lines: list[str] = []
        for record in ordered:
            if record.kind == "class":
                lines.append(f"- {record.signature}")
                method_entries = methods_by_owner.pop(record.qualified_name, [])
                for _, method_sig in sorted(method_entries):
                    lines.append(f"  - {method_sig}")
            elif record.kind == "function":
                lines.append(f"- {record.signature}")

        if methods_by_owner:
            owner_entries = [
                (min(entry[0] for entry in entries), owner, entries)
                for owner, entries in methods_by_owner.items()
            ]
            for _, owner, entries in sorted(owner_entries):
                owner_display = self._strip_module_prefix(owner, module_prefix)
                if owner_display:
                    lines.append(f"- {owner_display}")
                    for _, method_sig in sorted(entries):
                        lines.append(f"  - {method_sig}")
                else:
                    for _, method_sig in sorted(entries):
                        lines.append(f"- {method_sig}")

        if loose_methods:
            for _, method_sig in sorted(loose_methods):
                lines.append(f"- {method_sig}")

        if not lines:
            return ""
        body = "\n".join(lines)
        return f"### {path_key} :: symbol outline\n{body}"

    @staticmethod
    def _module_prefix_for_path(path_key: str) -> str:
        if not path_key.endswith(".py"):
            return ""
        module = path_key[:-3].replace("/", ".")
        if module.endswith("__init__"):
            module = module[: -len("__init__")]
            if module.endswith("."):
                module = module[:-1]
        return module.strip(".")

    @staticmethod
    def _strip_module_prefix(qualified_name: str, module_prefix: str) -> str:
        if not qualified_name or not module_prefix:
            return qualified_name
        prefix = f"{module_prefix}."
        if qualified_name.startswith(prefix):
            return qualified_name[len(prefix) :]
        return qualified_name

    @staticmethod
    def _render_payload(data: dict[str, Any]) -> str:
        """Convert structured request data into a prompt-friendly string."""
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return f"## Request Payload\n{serialized}"

    def _assemble_prompt(
        self,
        sections: Sequence[dict[str, Any]],
        *,
        budget: int | None = None,
    ) -> tuple[str, list[dict[str, Any]], int]:
        remaining = self._token_budget if budget is None else max(0, min(self._token_budget, budget))
        parts: list[str] = []
        metadata: list[dict[str, Any]] = []
        tokens_used = 0

        for section in sorted(sections, key=lambda item: item["priority"]):
            text = section["text"].strip()
            allow_truncate = bool(section.get("allow_truncate"))
            estimated = self._estimate_tokens(text)

            included = False
            truncated = False
            applied_text = ""
            tokens_for_section = 0

            if estimated <= remaining:
                applied_text = text
                tokens_for_section = estimated
                included = True
            elif allow_truncate and remaining > 0:
                applied_text = self._truncate_text_to_tokens(text, remaining)
                if applied_text:
                    tokens_for_section = self._estimate_tokens(applied_text)
                    included = True
                    truncated = True

            metadata.append(
                {
                    "label": section["label"],
                    "included": included,
                    "truncated": truncated,
                    "tokens": tokens_for_section,
                }
            )

            if not included:
                continue

            parts.append(applied_text)
            remaining = max(0, remaining - tokens_for_section)
            tokens_used += tokens_for_section
            if remaining <= 0:
                break

        prompt_text = "\n\n".join(parts).strip()
        return prompt_text, metadata, tokens_used

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self._CHARS_PER_TOKEN))

    def _truncate_text_to_tokens(self, text: str, allowed_tokens: int) -> str:
        if allowed_tokens <= 0:
            return ""
        max_chars = allowed_tokens * self._CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars].rstrip()
        if not truncated:
            return ""
        return f"{truncated}\n... (truncated)"

    def _extract_symbol_hints(self, request_data: dict[str, Any]) -> set[str]:
        hints: set[str] = set()
        for key, value in request_data.items():
            lower_key = key.lower()
            if "symbol" in lower_key or lower_key.endswith("functions") or lower_key.endswith("methods"):
                hints.update(self._coerce_to_strings(value))
        return {hint for hint in hints if hint}

    def _extract_file_hints(self, request_data: dict[str, Any]) -> set[str]:
        hints: set[str] = set()
        for key, value in request_data.items():
            lower_key = key.lower()
            if lower_key == "snippets":
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                    for item in value:
                        if not isinstance(item, Mapping):
                            continue
                        path_value = item.get("path")
                        if not isinstance(path_value, str):
                            continue
                        normalized = self._normalize_repo_path(path_value)
                        if normalized:
                            hints.add(normalized)
                continue
            if lower_key in self._REQUEST_FILE_HINT_KEYS:
                for item in self._coerce_to_strings(value):
                    normalized = self._normalize_repo_path(item)
                    if normalized:
                        hints.add(normalized)
                continue
            if lower_key == "test_plan":
                for command in self._coerce_to_strings(value):
                    for candidate in self._extract_paths_from_test_command(command):
                        normalized = self._normalize_repo_path(candidate)
                        if normalized:
                            hints.add(normalized)
                continue
            if lower_key in self._REQUEST_TEST_ID_KEYS:
                for item in self._coerce_to_strings(value):
                    candidate = self._extract_path_from_test_id(item)
                    if not candidate:
                        continue
                    normalized = self._normalize_repo_path(candidate)
                    if normalized:
                        hints.add(normalized)
                continue
            if lower_key == "logs":
                hints.update(self._extract_paths_from_logs(value))
                continue
            if "file" in lower_key or lower_key.endswith(("paths", "modules")):
                for item in self._coerce_to_strings(value):
                    normalized = self._normalize_repo_path(item)
                    if normalized:
                        hints.add(normalized)
        return hints

    def _coerce_to_strings(self, value: Any) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            collected: list[str] = []
            for item in value:
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        collected.append(stripped)
            return collected
        return []

    def _normalize_repo_path(self, path_str: str) -> str | None:
        fallback: str | None = None
        for variant in self._candidate_repo_paths(path_str):
            candidate = Path(variant)
            try:
                if candidate.is_absolute():
                    candidate = candidate.resolve()
                else:
                    candidate = (self._repo_root / candidate).resolve()
                candidate = candidate.relative_to(self._repo_root)
            except ValueError:
                continue
            candidate_key = candidate.as_posix()
            if (self._repo_root / candidate).exists():
                return candidate_key
            if fallback is None:
                fallback = candidate_key
        return fallback

    @staticmethod
    def _candidate_repo_paths(path_str: str) -> list[str]:
        trimmed = path_str.strip()
        if not trimmed:
            return []

        variants: list[str] = []
        seen: set[str] = set()

        def _add(entry: str) -> None:
            key = entry.strip()
            if not key or key in seen:
                return
            seen.add(key)
            variants.append(key)

        _add(trimmed)

        normalised = trimmed.replace("\\", "/")
        _add(normalised)

        cleaned = normalised
        while cleaned.startswith("./"):
            cleaned = cleaned[2:]
        _add(cleaned)

        parts = [part for part in cleaned.split("/") if part]
        for index in range(1, len(parts)):
            candidate = "/".join(parts[index:])
            _add(candidate)

        return variants

    def _extract_paths_from_logs(self, value: Any) -> set[str]:
        texts: list[str] = []
        if isinstance(value, str) and value.strip():
            texts.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    texts.append(item)
        hints: set[str] = set()
        for entry in texts:
            for candidate in self._path_candidates_from_text(entry):
                normalized = self._normalize_repo_path(candidate)
                if normalized:
                    hints.add(normalized)
        return hints

    def _extract_paths_from_test_command(self, command: str) -> list[str]:
        if not isinstance(command, str):
            return []
        command = command.strip()
        if not command:
            return []
        paths = []
        for candidate in self._path_candidates_from_text(command):
            if candidate not in paths:
                paths.append(candidate)
        return paths

    def invalidate_repository_cache(self) -> None:
        """Clear cached repository metadata so future prompts reflect current state."""

        self._repo_file_cache = None
        self._symbol_index = None
        self._symbol_index_loaded = False

    def _extract_path_from_test_id(self, identifier: str) -> str | None:
        if not isinstance(identifier, str):
            return None
        trimmed = identifier.strip()
        if not trimmed:
            return None
        # pytest node IDs separate path and selector with '::'
        path_part = trimmed.split("::", 1)[0].strip()
        if not path_part:
            return None
        return path_part

    def _path_candidates_from_text(self, value: str) -> list[str]:
        return [match.group(0) for match in self._PATH_TOKEN_RE.finditer(value)]

    def _symbol_snippet(self, record: SymbolRecord) -> str:
        file_path = self._repo_root / record.path
        try:
            content = file_path.read_bytes()
        except OSError:
            return ""
        snippet_bytes = content[record.start : record.end]
        try:
            snippet = snippet_bytes.decode("utf-8")
        except UnicodeDecodeError:
            snippet = snippet_bytes.decode("utf-8", errors="ignore")
        snippet = snippet.strip()
        if not snippet:
            return ""

        parts = [f"### {record.path} :: {record.name}"]
        if record.signature:
            parts.append(f"- Signature: `{record.signature}`")
        parts.append(f"```python\n{snippet}\n```")
        return "\n".join(parts)

    def _file_snippet(self, path_key: str) -> str:
        file_path = self._repo_root / path_key
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            return ""
        snippet = content.strip()
        if not snippet:
            return ""
        truncated = False
        if len(snippet) > self._MAX_FILE_PREVIEW_CHARS:
            snippet = snippet[: self._MAX_FILE_PREVIEW_CHARS].rstrip()
            snippet = f"{snippet}\n# ... (file truncated)"
            truncated = True
        outline = self._outline_summary(path_key, content)
        if outline:
            snippet = f"{snippet}\n# {outline}"
        return f"### {path_key} (excerpt)\n```python\n{snippet}\n```"

    def _outline_summary(self, path_key: str, content: str | None = None) -> str:
        index = self._load_symbol_index()
        records: Sequence[SymbolRecord] = ()
        if index is not None:
            records = index.symbols_for_path(path_key)

        names: list[str] = []
        for record in records:
            if record.kind == "function":
                names.append(f"{record.name}()")
            elif record.kind == "class":
                names.append(record.name)
            elif record.kind == "method":
                names.append(f"{record.name}()")
        if not names:
            source = content
            if source is None:
                try:
                    source = (self._repo_root / path_key).read_text(encoding="utf-8")
                except OSError:
                    source = None
            if source:
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    tree = None
                if tree is not None:
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef):
                            names.append(f"{node.name}()")
                        elif isinstance(node, ast.ClassDef):
                            names.append(node.name)
                            for item in node.body:
                                if isinstance(item, ast.FunctionDef):
                                    names.append(f"{node.name}.{item.name}()")
        if not names:
            return ""
        deduped: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        max_items = 12
        display = ", ".join(deduped[:max_items])
        if len(deduped) > max_items:
            remaining = len(deduped) - max_items
            display = f"{display}, ... (+{remaining} more)"
        return f"Existing symbols: {display}"

    def _load_symbol_index(self) -> SymbolIndex | None:
        if self._symbol_index_loaded:
            return self._symbol_index
        if self._symbol_index_path.exists():
            self._symbol_index = SymbolIndex(self._symbol_index_path)
        self._symbol_index_loaded = True
        return self._symbol_index

    @staticmethod
    def _resolve_repo_root(repo_root: Path | str | None) -> Path:
        """Determine an absolute repository root path from a user-provided hint."""
        if repo_root is None:
            return Path.cwd().resolve()
        path = Path(repo_root)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    @staticmethod
    def _resolve_data_root(data_root: Path | str | None, repo_root: Path) -> Path:
        """Resolve the data directory, defaulting to ``repo_root / data`` when unset."""
        if data_root is None:
            return (repo_root / "data").resolve()
        path = Path(data_root)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return path

    @staticmethod
    def _resolve_logs_root(logs_root: Path | str | None, data_root: Path, repo_root: Path) -> Path:
        """Resolve the logs directory based on explicit hints or the data directory."""
        if logs_root is None:
            return (data_root / "logs").resolve()
        path = Path(logs_root)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return path

    @staticmethod
    def _infer_repo_root(config: Mapping[str, Any], *, base_path: Path | None = None) -> Path:
        """Infer the repository root from configuration data."""
        project = config.get("project")
        repo_root_value = "."
        if isinstance(project, Mapping):
            candidate = project.get("repo_root")
            if isinstance(candidate, str):
                repo_root_value = candidate
        repo_root = Path(repo_root_value)
        if repo_root.is_absolute():
            return repo_root.resolve()
        base = base_path
        if base is None:
            base = Path.cwd()
        else:
            base = base.resolve()
        return (base / repo_root).resolve()

    @staticmethod
    def _infer_data_root(config: Mapping[str, Any], repo_root: Path) -> Path:
        """Infer the data directory location from configuration data."""
        paths_section = config.get("paths")
        data_root_value = "data"
        if isinstance(paths_section, Mapping):
            candidate = paths_section.get("data")
            if isinstance(candidate, str):
                data_root_value = candidate
        data_root = Path(data_root_value)
        if not data_root.is_absolute():
            data_root = (repo_root / data_root).resolve()
        return data_root

    @staticmethod
    def _infer_logs_root(config: Mapping[str, Any], repo_root: Path, data_root: Path) -> Path:
        """Infer the logs directory location from configuration data."""
        paths_section = config.get("paths")
        if isinstance(paths_section, Mapping):
            candidate = paths_section.get("logs")
            if isinstance(candidate, str) and candidate.strip():
                path = Path(candidate.strip())
                if not path.is_absolute():
                    path = (repo_root / path).resolve()
                return path
        return (data_root / "logs").resolve()
