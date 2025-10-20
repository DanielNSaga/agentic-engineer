"""Local fallbacks implementing deterministic heuristics for each phase."""

from __future__ import annotations

import ast
import difflib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..context_builder import ContextBuilder
from ..memory.reflections import get_planner_preflight_seed
from ..models.llm_client import LLMClient
from ..planning.schemas import PlannerDecision, PlannerResponse, PlannerRisk, PlannerTask
from ..tools.snippets import SnippetRequest

if TYPE_CHECKING:
    from .analyze import AnalyzeRequest, AnalyzeResponse
    from .design import DesignRequest, DesignResponse
    from .diagnose import DiagnoseRequest, DiagnoseResponse
    from .fix_violations import FixViolationsRequest, FixViolationsResponse
    from .implement import ImplementRequest, ImplementResponse
    from .plan_adjust import PlanAdjustRequest, PlanAdjustResponse
    from .plan import PlanRequest, PlanResponse

__all__ = [
    "LocalPhaseLogic",
    "supports_local_logic",
]


_ASSERTION_RE = re.compile(r"AssertionError: assert (?P<actual>.+?) == (?P<expected>.+)$")
_ASSERTION_IN_RE = re.compile(r"AssertionError: assert (?P<needle>.+?) in (?P<haystack>.+)$")
_STATIC_LOCATION_RE = re.compile(r"(?P<path>[^:\s]+\.py):(?P<line>\d+)")
_IMPORT_ERROR_RE = re.compile(
    r"ImportError:\s+cannot import name ['\"]?(?P<symbol>[\w.]+)['\"]?\s+from\s+['\"]?(?P<module>[\w.]+)['\"]?(?:\s+\((?P<path>[^)]+)\))?",
    re.IGNORECASE,
)
_ATTRIBUTE_ERROR_RE = re.compile(
    r"AttributeError:\s+(?:module|type object)\s+['\"](?P<module>[\w.]+)['\"]\s+has no attribute\s+['\"](?P<symbol>[\w.]+)['\"]",
    re.IGNORECASE,
)
_MODULE_NOT_FOUND_RE = re.compile(
    r"ModuleNotFoundError:\s+No module named ['\"](?P<module>[\w.]+)['\"]",
    re.IGNORECASE,
)
_NAME_ERROR_RE = re.compile(
    r"NameError:\s+name ['\"](?P<symbol>[\w.]+)['\"] is not defined",
    re.IGNORECASE,
)


@dataclass(slots=True)
class MissingSymbolSignal:
    kind: str
    symbol: str
    module: str | None = None
    path_hint: str | None = None
    message: str | None = None


def supports_local_logic(client: LLMClient) -> bool:
    """Return True when the supplied client opts into local heuristics."""
    return bool(getattr(client, "supports_local_logic", False))


@dataclass(slots=True)
class LocalPhaseLogic:
    """Lightweight heuristics that emulate structured agent behaviour locally."""

    context_builder: ContextBuilder

    # ------------------------------------------------------------------ public
    def analyze(self, request: "AnalyzeRequest") -> "AnalyzeResponse":
        from .analyze import AnalyzeResponse

        summary = f"Investigate task {request.task_id}: {request.goal.strip()}."
        plan_steps = [
            "Inspect referenced files or symbols mentioned in the request context.",
            "Review existing tests covering the relevant behaviour to understand expected outcomes.",
            "Draft a minimal change that satisfies the stated goal.",
            "Identify tests or commands to validate the change.",
        ]
        if request.constraints:
            plan_steps.append("Verify all stated constraints are respected before finalizing.")
        risks = [f"Constraint: {item}" for item in request.constraints]
        risks.append("Risk: implementation diverges from existing tests if expectations are overlooked.")
        return AnalyzeResponse(
            summary=summary,
            plan_steps=plan_steps,
            risks=risks,
        )

    def design(self, request: "DesignRequest") -> "DesignResponse":
        from .design import DesignResponse

        design_summary = f"Outline design for task {request.task_id} targeting {request.goal}."
        interface_changes = list(request.proposed_interfaces)
        rationale = [
            "Maintain backwards compatibility unless the goal requires otherwise.",
        ]
        if request.constraints:
            rationale.extend(f"Respect constraint: {constraint}" for constraint in request.constraints)
        validation_plan = [
            "Confirm behaviour against existing tests or fixtures that describe expected outcomes.",
            "Run targeted tests or commands mentioned in the implementation plan.",
        ]
        return DesignResponse(
            design_summary=design_summary,
            interface_changes=interface_changes,
            rationale=rationale,
            validation_plan=validation_plan,
        )

    def implement(self, request: "ImplementRequest") -> "ImplementResponse | None":
        from ..structured import StructuredFileArtifact
        from .implement import ImplementResponse

        repo_root = self.context_builder.repo_root
        for relative in request.touched_files:
            target = (repo_root / relative).resolve()
            if target.suffix != ".py" or not target.exists():
                continue
            diff_info = self._synthesise_greeting_update(target, request)
            if diff_info:
                _, updated = diff_info
                summary = f"Update {relative} to prepare the greeting change."
                follow_up = list(request.notes)
                follow_up.append("Review the generated patch for correctness.")
                test_commands = list(request.test_plan)
                artifact = StructuredFileArtifact(path=relative, content=updated)
                return ImplementResponse(
                    summary=summary,
                    test_commands=test_commands,
                    follow_up=follow_up,
                    files=[artifact],
                )
        return None

    def fix_violations(self, request: "FixViolationsRequest") -> "FixViolationsResponse | None":
        from ..structured import StructuredFileArtifact
        from .fix_violations import FixViolationsResponse

        repo_root = self.context_builder.repo_root
        edits: dict[Path, list[str]] = {}
        rationales: list[str] = []

        for violation in request.violations:
            match = _STATIC_LOCATION_RE.search(violation)
            if not match:
                continue
            relative = match.group("path")
            line_no = int(match.group("line"))
            path = (repo_root / relative).resolve()
            if not path.exists() or path.suffix != ".py":
                continue
            lines = edits.setdefault(path, path.read_text(encoding="utf-8").splitlines())
            index = line_no - 1
            if not (0 <= index < len(lines)):
                continue
            stripped = lines[index].strip()
            if "print(" in stripped:
                lines[index] = self._comment_line(lines[index])
                rationales.append(f"Comment out print statement flagged by static checks ({relative}:{line_no}).")
            elif "unused import" in violation.lower() or "F401" in violation:
                lines[index] = ""
                rationales.append(f"Remove unused import flagged by static checks ({relative}:{line_no}).")
        if not edits and request.current_diff:
            for relative in self._paths_from_diff(request.current_diff):
                path = (repo_root / relative).resolve()
                if not path.exists() or path.suffix != ".py":
                    continue
                lines = path.read_text(encoding="utf-8").splitlines()
                modified = False
                for index, line in enumerate(lines):
                    stripped = line.strip()
                    if "print(" in line:
                        lines[index] = self._comment_line(line)
                        rationales.append(f"Comment out print statement introduced in {relative}.")
                        modified = True
                    if stripped.startswith("import logging"):
                        lines[index] = ""
                        rationales.append(f"Remove temporary logging import in {relative}.")
                        modified = True
                if modified:
                    edits[path] = lines

        diffs = []
        artifacts: list[StructuredFileArtifact] = []
        touched_paths: list[str] = []
        for path, updated_lines in edits.items():
            original = path.read_text(encoding="utf-8")
            updated = "\n".join(updated_lines).rstrip() + "\n"
            if updated == original:
                continue
            try:
                relative = path.relative_to(repo_root).as_posix()
            except ValueError:
                relative = path.as_posix()
            artifacts.append(StructuredFileArtifact(path=relative, content=updated))
            touched_paths.append(relative)

        if not artifacts:
            return None

        follow_up = ["Re-run static gates to confirm violations are cleared."]
        return FixViolationsResponse(
            rationale=rationales or ["Automatically addressed static analysis feedback."],
            follow_up=follow_up,
            touched_files=tuple(touched_paths),
            files=artifacts,
        )

    def diagnose(self, request: "DiagnoseRequest") -> "DiagnoseResponse | None":
        from .diagnose import DiagnoseResponse

        logs_text = self._normalise_logs(request.logs)
        failing_tests = list(request.failing_tests)
        missing_signals = self._parse_missing_symbol_signals(logs_text)
        assertion_pairs = self._parse_assertion_pairs(logs_text)

        suspected: list[str] = []
        recommendations: list[str] = []
        lessons: list[str] = []
        code_requests: list[SnippetRequest] = list(request.code_requests)
        existing_paths = {req.path for req in code_requests if isinstance(req, SnippetRequest)}

        for signal in missing_signals:
            module_display = signal.module or "the imported module"
            description = f"Import failure: `{signal.symbol}` is unavailable from `{module_display}`."
            if signal.kind == "module_not_found":
                description = f"Missing module: `{module_display}` was not found when resolving imports."
            elif signal.kind == "name_error":
                description = f"Name error: `{signal.symbol}` is referenced before it is defined."
            if signal.message and signal.message not in description:
                description = f"{description} ({signal.message})"
            suspected.append(description)

            candidate_paths = self._candidate_paths_for_missing_symbol(signal, request)
            path_hint = candidate_paths[0] if candidate_paths else None

            if signal.kind == "module_not_found":
                if path_hint:
                    recommendations.append(
                        f"Create the module at `{path_hint}` or adjust the import to target an existing module."
                    )
                else:
                    recommendations.append(
                        f"Create a package/module for `{module_display}` or update the import to use the correct entry point."
                    )
                lessons.append(f"Module `{module_display}` is unresolved; ensure the package structure exports it.")
            elif signal.kind == "name_error":
                location_hint = path_hint or "the failing module"
                recommendations.append(
                    f"Define `{signal.symbol}` in `{location_hint}` before it is referenced, or update the code to use an existing symbol."
                )
                lessons.append(f"Ensure `{signal.symbol}` is declared where tests expect to use it.")
            else:
                export_location = path_hint or module_display.replace(".", "/") + ".py"
                recommendations.append(
                    f"Implement and export `{signal.symbol}` from `{export_location}` (update `__all__` or re-export it)."
                )
                lessons.append(f"Expose `{signal.symbol}` from `{module_display}` to satisfy import statements.")

            for path in candidate_paths:
                if path not in existing_paths:
                    code_requests.append(
                        SnippetRequest(
                            path=path,
                            surround=80,
                            reason=f"Inspect where `{signal.symbol}` should be defined or exported.",
                        )
                    )
                    existing_paths.add(path)

        if assertion_pairs:
            for actual, expected in assertion_pairs:
                suspected.append(f"Assertion mismatch: expected `{expected}` but received `{actual}`.")
            if not any("Align" in item for item in recommendations):
                recommendations.append("Align the implementation output with the values asserted by the failing tests.")
            lessons.append("Tests are asserting different values than the implementation currently returns.")

        if not suspected:
            suspected.append("Unable to extract a specific failure signal; inspect failing tests and recent changes.")
        if not recommendations:
            recommendations.append("Review failing tests and logs to translate failures into concrete code changes.")

        suspected = list(dict.fromkeys(suspected))
        recommendations = list(dict.fromkeys(recommendations))
        lessons = list(dict.fromkeys(lessons))

        confidence = 0.4
        if missing_signals:
            confidence = 0.78
        elif assertion_pairs:
            confidence = 0.6

        return DiagnoseResponse(
            suspected_causes=suspected,
            recommended_fixes=recommendations,
            additional_tests=failing_tests,
            code_requests=code_requests,
            iteration_lessons=lessons,
            confidence=confidence,
        )

    def _normalise_logs(self, logs: Any) -> str:
        if isinstance(logs, str):
            return logs
        if isinstance(logs, Sequence) and not isinstance(logs, (bytes, bytearray)):
            return "\n".join(item for item in logs if isinstance(item, str))
        return ""

    def _parse_missing_symbol_signals(self, logs_text: str) -> list[MissingSymbolSignal]:
        signals: list[MissingSymbolSignal] = []
        if not logs_text:
            return signals

        seen: set[tuple[str, str, str, str]] = set()

        def _add(signal: MissingSymbolSignal) -> None:
            key = (
                signal.kind,
                signal.symbol.lower(),
                (signal.module or "").lower(),
                (signal.path_hint or "").lower(),
            )
            if key in seen:
                return
            seen.add(key)
            signals.append(signal)

        for match in _IMPORT_ERROR_RE.finditer(logs_text):
            symbol = (match.group("symbol") or "").strip()
            module = (match.group("module") or "").strip()
            path_hint = (match.group("path") or "").strip() or None
            if not symbol:
                continue
            _add(
                MissingSymbolSignal(
                    kind="import_error",
                    symbol=symbol,
                    module=module or None,
                    path_hint=path_hint,
                    message=match.group(0).strip(),
                )
            )

        for match in _ATTRIBUTE_ERROR_RE.finditer(logs_text):
            symbol = (match.group("symbol") or "").strip()
            module = (match.group("module") or "").strip()
            if not symbol:
                continue
            _add(
                MissingSymbolSignal(
                    kind="attribute_error",
                    symbol=symbol,
                    module=module or None,
                    message=match.group(0).strip(),
                )
            )

        for match in _MODULE_NOT_FOUND_RE.finditer(logs_text):
            module = (match.group("module") or "").strip()
            if not module:
                continue
            _add(
                MissingSymbolSignal(
                    kind="module_not_found",
                    symbol=module,
                    module=module,
                    message=match.group(0).strip(),
                )
            )

        for match in _NAME_ERROR_RE.finditer(logs_text):
            symbol = (match.group("symbol") or "").strip()
            if not symbol:
                continue
            _add(
                MissingSymbolSignal(
                    kind="name_error",
                    symbol=symbol,
                    message=match.group(0).strip(),
                )
            )

        return signals

    def _parse_assertion_pairs(self, logs_text: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        if not logs_text:
            return pairs
        for line in logs_text.splitlines():
            stripped = line.strip()
            match_eq = _ASSERTION_RE.search(stripped)
            if match_eq:
                actual = self._strip_quotes(match_eq.group("actual").strip())
                expected = self._strip_quotes(match_eq.group("expected").strip())
                pairs.append((actual, expected))
                continue
            match_in = _ASSERTION_IN_RE.search(stripped)
            if match_in:
                haystack = self._strip_quotes(match_in.group("haystack").strip())
                needle = self._strip_quotes(match_in.group("needle").strip())
                pairs.append((haystack, needle))
        return pairs

    def _candidate_paths_for_missing_symbol(
        self,
        signal: MissingSymbolSignal,
        request: "DiagnoseRequest",
    ) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def _add(path: str | None) -> None:
            if not path or path in seen:
                return
            seen.add(path)
            candidates.append(path)

        if signal.path_hint:
            hint_path = self._repo_relative_path(signal.path_hint)
            if hint_path:
                _add(hint_path)

        module = signal.module or ""
        if module:
            module_key = module.replace(".", "/")
            file_candidate = self._repo_relative_path(f"{module_key}.py")
            package_candidate = self._repo_relative_path(f"{module_key}/__init__.py")
            _add(file_candidate)
            _add(package_candidate)

        lowered_symbol = signal.symbol.lower()

        for relative in request.recent_changes:
            if not isinstance(relative, str):
                continue
            candidate = relative.strip()
            if not candidate or lowered_symbol not in Path(candidate).stem.lower():
                continue
            resolved = self._repo_relative_path(candidate)
            if resolved:
                _add(resolved)

        for snippet in request.snippets:
            path = getattr(snippet, "path", None)
            if not isinstance(path, str):
                continue
            candidate = self._repo_relative_path(path)
            if not candidate:
                continue
            if lowered_symbol in Path(path).stem.lower():
                _add(candidate)

        return candidates

    def _repo_relative_path(self, path: str | Path | None, *, must_exist: bool = True) -> str | None:
        if not path:
            return None
        repo_root = self.context_builder.repo_root
        path_text = str(path).strip().strip("'\"")
        if path_text.endswith(")"):
            path_text = path_text.rstrip(")")
        candidate = Path(path_text)
        try:
            if candidate.is_absolute():
                resolved = candidate.resolve()
            else:
                resolved = (repo_root / candidate).resolve()
            relative = resolved.relative_to(repo_root)
        except (OSError, ValueError):
            return None
        if must_exist and not resolved.exists():
            return None
        return relative.as_posix()

    def plan(self, request: "PlanRequest") -> "PlanResponse":
        goal = request.goal.strip()
        constraints = list(request.constraints)
        deliverables = list(request.deliverables)
        notes = list(request.notes)

        preflight_tasks = self._build_preflight_tasks(request)

        base_tasks = [
            PlannerTask(
                id="plan::understand-requirements",
                title="Understand requirements",
                summary=f"Review repository state and clarify acceptance for goal: {goal}.",
                constraints=constraints,
                deliverables=["Requirements summary"],
                acceptance_criteria=[
                    "List current behaviour related to the goal.",
                    "Highlight unknowns or missing context.",
                ],
                notes=notes,
            ),
            PlannerTask(
                id="plan::implement-solution",
                title="Design and implement solution",
                summary=f"Propose and implement the changes needed to achieve '{goal}'.",
                depends_on=["plan::understand-requirements"],
                constraints=constraints,
                deliverables=deliverables or ["Code updates"],
                acceptance_criteria=[
                    "Implementation covers all deliverables.",
                    "Relevant tests are updated or added.",
                ],
            ),
            PlannerTask(
                id="plan::validate-outcome",
                title="Validate outcome",
                summary="Run automated checks and document risks before completion.",
                depends_on=["plan::implement-solution"],
                deliverables=["Validation notes"],
                acceptance_criteria=[
                    "All automated tests pass.",
                    "Known risks are documented with mitigations.",
                ],
            ),
        ]

        if preflight_tasks:
            preflight_ids = [task.id for task in preflight_tasks if task.id]
            if preflight_ids:
                self._extend_dependencies(base_tasks[0], preflight_ids)

        tasks = [*preflight_tasks, *base_tasks]

        decisions = [
            PlannerDecision(
                id="plan::decision-scope",
                title="Initial scope assumptions",
                content=f"Focus on the primary repository components impacted by '{goal}'.",
                kind="assumption",
            )
        ]

        risks = [
            PlannerRisk(
                id="plan::risk-regression",
                description="Changes may introduce regressions in untouched modules.",
                mitigation="Add targeted tests and rely on automated gates.",
                impact="medium",
                likelihood="medium",
            )
        ]

        summary_parts = []
        if preflight_tasks:
            titles = ", ".join(task.title for task in preflight_tasks)
            summary_parts.append(f"Pre-flight remediation: {titles}.")
        summary_parts.append(f"Three-step plan to accomplish '{goal}'.")
        plan_summary = " ".join(summary_parts).strip()

        return PlannerResponse(
            plan_name=f"Plan for {goal or 'repository goal'}",
            plan_summary=plan_summary,
            tasks=tasks,
            decisions=decisions,
            risks=risks,
        )

    def plan_adjust(self, request: "PlanAdjustRequest") -> "PlanAdjustResponse":
        from .plan_adjust import PlanAdjustResponse

        adjustments = list(request.suggested_changes)
        if not adjustments:
            adjustments = [f"Re-assess task {request.task_id} due to: {request.reason}."]
        new_tasks = [f"Investigate blocker: {blocker}" for blocker in request.blockers]
        risks = ["Schedule slip"] if request.blockers else []
        notes = [request.reason] if request.reason else []
        return PlanAdjustResponse(
            adjustments=adjustments,
            new_tasks=new_tasks,
            drop_tasks=[],
            risks=risks,
            notes=notes,
        )

    # ----------------------------------------------------------------- helpers
    def _build_preflight_tasks(self, request: "PlanRequest") -> list[PlannerTask]:
        repo_state = self._evaluate_repo_preflight_state(request)
        reflections = get_planner_preflight_seed()

        tasks: list[PlannerTask] = []
        for reflection in reflections:
            context = reflection.context or {}
            if not isinstance(context, Mapping):
                continue
            applies_when = context.get("applies_when")
            if not self._reflection_applies(applies_when, repo_state):
                continue
            task_payload = context.get("task")
            if not isinstance(task_payload, Mapping):
                continue
            task_data = json.loads(json.dumps(task_payload))
            metadata = dict(task_data.get("metadata") or {})
            metadata.setdefault("reflection_id", reflection.id)
            metadata.setdefault("reflection_scope", reflection.scope)
            tags = context.get("tags")
            if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes, bytearray)):
                metadata.setdefault("reflection_tags", [str(tag) for tag in tags])
            task_data["metadata"] = metadata
            tasks.append(PlannerTask(**task_data))
        return tasks

    @staticmethod
    def _reflection_applies(applies_when: Any, repo_state: Mapping[str, bool]) -> bool:
        if not applies_when:
            return True
        if isinstance(applies_when, Mapping):
            for key, expected in applies_when.items():
                if isinstance(expected, bool):
                    if bool(repo_state.get(key)) != expected:
                        return False
            return True
        if isinstance(applies_when, str):
            return bool(repo_state.get(applies_when))
        return False

    def _evaluate_repo_preflight_state(self, request: "PlanRequest") -> dict[str, bool]:
        repo_root = self.context_builder.repo_root
        package_roots = self._resolve_package_roots(repo_root, request.known_context)
        goal_text = request.goal if isinstance(request.goal, str) else ""
        goal_lower = goal_text.lower()
        goal_mentions_cli = any(
            token in goal_lower
            for token in (
                "cli",
                "command line",
                "command-line",
                "commandline",
            )
        )
        state = {
            "missing_typed_wrappers": self._repo_missing_typed_wrappers(repo_root, package_roots),
            "missing_package_exports": self._repo_missing_package_exports(package_roots),
            "missing_module_docstrings": self._repo_missing_docstrings(package_roots),
            "missing_test_scaffolding": self._repo_missing_test_scaffolding(repo_root, request.known_context),
            "goal_mentions_cli": goal_mentions_cli,
        }
        return state

    def _resolve_package_roots(self, repo_root: Path, known_context: Mapping[str, Any]) -> list[Path]:
        package_roots: list[Path] = []
        src_packages = known_context.get("src_packages")
        if isinstance(src_packages, Sequence) and not isinstance(src_packages, (str, bytes, bytearray)):
            for entry in src_packages:
                if not isinstance(entry, str):
                    continue
                cleaned = entry.strip().strip("/")
                if not cleaned:
                    continue
                candidate = (repo_root / cleaned).resolve()
                try:
                    candidate.relative_to(repo_root)
                except ValueError:
                    continue
                if candidate.is_dir():
                    package_roots.append(candidate)
        if not package_roots:
            src_root = repo_root / "src"
            if src_root.is_dir():
                for child in sorted(src_root.iterdir(), key=lambda path: path.name):
                    if child.is_dir() and not child.name.startswith("."):
                        package_roots.append(child)
        return package_roots[:5]

    def _repo_missing_typed_wrappers(self, repo_root: Path, package_roots: Sequence[Path]) -> bool:
        for root in package_roots:
            if self._root_has_stub_artifacts(root):
                return False
        src_root = repo_root / "src"
        if src_root.is_dir() and self._root_has_stub_artifacts(src_root):
            return False
        return True

    @staticmethod
    def _root_has_stub_artifacts(root: Path) -> bool:
        if (root / "py.typed").exists():
            return True
        for path in root.rglob("*.pyi"):
            return True
        return False

    def _repo_missing_package_exports(self, package_roots: Sequence[Path]) -> bool:
        if not package_roots:
            return False
        for package in package_roots:
            init_path = package / "__init__.py"
            if not init_path.exists():
                return True
            text = self._safe_read_text(init_path)
            if "__all__" not in text:
                return True
        return False

    def _repo_missing_docstrings(self, package_roots: Sequence[Path], *, sample_limit: int = 200) -> bool:
        inspected = 0
        for package in package_roots:
            if not package.is_dir():
                continue
            for path in package.rglob("*.py"):
                if path.name in {"__init__.py", "__main__.py"}:
                    continue
                inspected += 1
                if inspected > sample_limit:
                    return False
                text = self._safe_read_text(path)
                if not text.strip():
                    return True
                try:
                    module = ast.parse(text)
                except SyntaxError:
                    continue
                if ast.get_docstring(module) is None:
                    return True
        return False

    def _repo_missing_test_scaffolding(
        self,
        repo_root: Path,
        known_context: Mapping[str, Any],
    ) -> bool:
        tests_root = repo_root / "tests"
        if tests_root.is_dir():
            for pattern in ("test_*.py", "*_test.py"):
                if any(tests_root.rglob(pattern)):
                    return False
            return True
        return not bool(known_context.get("tests_present"))

    @staticmethod
    def _safe_read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    @staticmethod
    def _extend_dependencies(task: PlannerTask, dependencies: Sequence[str]) -> None:
        existing = set(task.depends_on)
        for dep in dependencies:
            if dep and dep not in existing:
                task.depends_on.append(dep)
                existing.add(dep)

    def _synthesise_greeting_update(self, path: Path, request: ImplementRequest) -> tuple[str, str] | None:
        goal = request.diff_goal.lower()
        if "include the name" not in goal and "include name" not in goal:
            return None

        original = path.read_text(encoding="utf-8")
        lines = original.splitlines()
        tree = ast.parse(original)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.args.args:
                continue
            first_arg = node.args.args[0].arg
            return_stmt = self._find_constant_return(node)
            if return_stmt is None:
                continue
            index = return_stmt.lineno - 1
            indent = self._leading_whitespace(lines[index])
            lines[index] = f"{indent}return \"hello there\""
            debug_line = f'{indent}print(f"[debug] greeting called for {{{first_arg}}}")'
            if debug_line not in lines:
                lines.insert(index, debug_line)
            if not any(line.strip().startswith("import logging") for line in lines):
                lines.insert(0, "")
                lines.insert(0, "import logging")
            updated = "\n".join(lines).rstrip() + "\n"
            return self._build_diff(path, original, updated), updated
        return None

    @staticmethod
    def _find_constant_return(node: ast.FunctionDef) -> ast.Return | None:
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                value = stmt.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return stmt
        return None

    @staticmethod
    def _leading_whitespace(line: str) -> str:
        return line[: len(line) - len(line.lstrip(" \t"))]

    @staticmethod
    def _comment_line(line: str) -> str:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        return f"{indent}# {stripped.strip()}"

    @staticmethod
    def _paths_from_diff(diff_text: str) -> list[str]:
        paths: set[str] = set()
        for line in diff_text.splitlines():
            if line.startswith("diff --git"):
                parts = line.split()
                if len(parts) >= 3:
                    candidate = parts[2]
                    if candidate.startswith("b/") or candidate.startswith("a/"):
                        candidate = candidate[2:]
                    paths.add(candidate)
        return sorted(paths)

    def _build_diff(self, path: Path, original: str, updated: str) -> str:
        repo_root = self.context_builder.repo_root
        rel = path
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            rel = path
        rel_posix = rel.as_posix()
        original_lines = original.splitlines()
        updated_lines = updated.splitlines()
        diff_lines = list(
            difflib.unified_diff(
                original_lines,
                updated_lines,
                fromfile=f"a/{rel_posix}",
                tofile=f"b/{rel_posix}",
                lineterm="",
            )
        )
        if not diff_lines:
            return ""
        header = f"diff --git a/{rel_posix} b/{rel_posix}"
        return "\n".join([header, *diff_lines]) + "\n"

    @staticmethod
    def _strip_quotes(value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]
        if value.startswith("repr(") and value.endswith(")"):
            inner = value[5:-1]
            return LocalPhaseLogic._strip_quotes(inner)
        return value
