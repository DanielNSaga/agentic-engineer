"""Static gate orchestration for policy and lint workflows.

The gate runner wires together the Code Policy Capsule (CPC) checks with a
small set of static analysis commands. Future iterations can extend this module
to drive additional tools and to react to violations automatically. For now the
focus is on providing a deterministic API that the CLI (and tests) can exercise.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Literal

import shutil
import subprocess
import textwrap
import yaml

from ae.policy.cpc import CPCChecker, PolicyViolation
from .snippets import StaticFinding
from .static_output import resolve_static_parser

StaticStatus = Literal["passed", "failed", "skipped"]


@dataclass(slots=True)
class StaticCheck:
    """Description of a static analysis command."""

    name: str
    command: Sequence[str]
    optional: bool = True

    def run(self, cwd: Path) -> "StaticCheckResult":
        executable = self.command[0]
        if shutil.which(executable) is None:
            status: StaticStatus = "skipped" if self.optional else "failed"
            return StaticCheckResult(
                name=self.name,
                command=list(self.command),
                status=status,
                exit_code=None,
                stdout="",
                stderr=f"Executable not available: {executable}",
                repo_root=cwd,
            )

        process = subprocess.run(  # noqa: S603  # command is sourced from policy config
            list(self.command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        status = "passed" if process.returncode == 0 else "failed"
        return StaticCheckResult(
            name=self.name,
            command=list(self.command),
            status=status,
            exit_code=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            repo_root=cwd,
        )


@dataclass(slots=True)
class StaticCheckResult:
    """Result produced by :class:`StaticCheck`."""

    name: str
    command: List[str]
    status: StaticStatus
    exit_code: int | None
    stdout: str
    stderr: str
    repo_root: Path | None = None

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    def short_message(self) -> str:
        if self.status == "passed":
            return f"{self.name}: passed"
        if self.status == "skipped":
            return f"{self.name}: skipped ({self.stderr.strip()})"
        snippet = self._format_failure_snippet()
        return f"{self.name}: failed ({snippet})"

    def _format_failure_snippet(self) -> str:
        parser = self._resolve_parser()
        combined_output = "\n".join(part for part in (self.stdout, self.stderr) if part)
        if parser is not None and combined_output.strip():
            findings = parser(combined_output)
            for finding in findings:
                location = finding.path.strip() if finding.path else "(unknown)"
                line = finding.line_start
                if isinstance(line, int) and line > 0:
                    location = f"{location}:{line}"
                message = finding.message.strip() if finding.message else ""
                if location and message:
                    return f"{location} :: {message}"
        fallback = self.stderr.strip() or self.stdout.strip()
        return fallback.splitlines()[0] if fallback else "exit code != 0"

    def _resolve_parser(
        self,
    ) -> Callable[[str | Sequence[str] | Iterable[str]], list[StaticFinding]] | None:
        tokens: list[str] = []
        seen_tokens: set[str] = set()

        def _register(value: str | None) -> None:
            if not isinstance(value, str):
                return
            stripped = value.strip()
            if not stripped:
                return
            lowered = stripped.lower()
            if lowered in seen_tokens:
                return
            seen_tokens.add(lowered)
            tokens.append(stripped)
            stem = Path(stripped).stem
            if stem:
                stem_lower = stem.lower()
                if stem_lower not in seen_tokens:
                    seen_tokens.add(stem_lower)
                    tokens.append(stem)

        _register(self.name)
        for entry in self.command or []:
            _register(entry)

        return resolve_static_parser(self.repo_root, tokens)


@dataclass(slots=True)
class GateReport:
    """Aggregated result of running the policy capsule and static checks."""

    policy_violations: List[PolicyViolation]
    static_results: List[StaticCheckResult]

    @property
    def has_failures(self) -> bool:
        if self.policy_violations:
            return True
        return any(result.failed for result in self.static_results)

    def format_summary(self) -> str:
        """Return a human readable summary of the run."""

        lines: List[str] = []
        if self.policy_violations:
            lines.append("Policy violations:")
            for violation in self.policy_violations:
                location = violation.path.as_posix() if violation.path else "(unknown)"
                if violation.line is not None:
                    location = f"{location}:{violation.line}"
                lines.append(f"- {violation.rule} :: {location} :: {violation.message}")
        else:
            lines.append("Policy capsule checks passed.")

        if self.static_results:
            lines.append("Static checks:")
            for result in self.static_results:
                lines.append(f"- {result.short_message()}")
        else:
            lines.append("No static checks configured.")

        return "\n".join(lines)


DEFAULT_STATIC_CHECKS: tuple[StaticCheck, ...] = ()


def _normalise_static_checks(policy_section: Mapping[str, Any]) -> List[StaticCheck]:
    """Expand raw configuration entries into `StaticCheck` definitions."""
    raw = policy_section.get("static_checks")
    if not raw:
        return [StaticCheck(name=item.name, command=item.command, optional=item.optional) for item in DEFAULT_STATIC_CHECKS]

    checks: List[StaticCheck] = []
    for entry in raw:
        if isinstance(entry, str):
            parts = entry.split()
            if not parts:
                continue
            checks.append(StaticCheck(name=parts[0], command=parts, optional=True))
            continue

        if isinstance(entry, Mapping):
            command = entry.get("command") or entry.get("cmd")
            if isinstance(command, str):
                cmd_parts = command.split()
            else:
                cmd_parts = list(command or [])

            if not cmd_parts:
                continue

            name = str(entry.get("name")) if entry.get("name") else cmd_parts[0]
            optional = bool(entry.get("optional", True))
            checks.append(StaticCheck(name=name, command=cmd_parts, optional=optional))
            continue

    return checks


def _load_config(config_path: Path) -> dict[str, Any]:
    """Read the agent configuration from disk, returning an empty mapping if missing."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_repo_root(config: Mapping[str, Any], config_path: Path, repo_root: Path | None) -> Path:
    """Determine the repository root from CLI overrides or configuration."""
    if repo_root is not None:
        return repo_root

    project = config.get("project") or {}
    candidate = project.get("repo_root")
    if candidate:
        return (config_path.parent / Path(candidate)).resolve()

    return config_path.parent.resolve()


def _materialise_checks(config: Mapping[str, Any]) -> List[StaticCheck]:
    """Construct the list of static checks enabled by the configuration."""
    policy_section = config.get("policy") or {}
    if not isinstance(policy_section, Mapping):
        policy_section = {}
    checks = _normalise_static_checks(policy_section)
    return checks


def run_policy_and_static(
    config_path: Path | str = "config.yaml",
    *,
    repo_root: Path | None = None,
) -> GateReport:
    """Execute CPC checks followed by static analysis commands.

    Parameters
    ----------
    config_path:
        Path to the configuration file.  Defaults to ``config.yaml`` in the
        repository root.  Relative paths are resolved against the current
        working directory.
    repo_root:
        Optional repository root override.  When omitted the path is resolved
        using ``project.repo_root`` from the configuration file.
    """

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    config = _load_config(config_path)
    repo_root = _resolve_repo_root(config, config_path, repo_root)

    cpc_checker = CPCChecker(repo_root=repo_root, config_path=config_path)
    violations = cpc_checker.run()

    static_checks = _materialise_checks(config)
    results = [check.run(repo_root) for check in static_checks]

    return GateReport(policy_violations=violations, static_results=results)


def describe_policy_capsule() -> str:
    """Return a formatted description of the known CPC rules."""

    from ae.policy.cpc import CPC_RULES  # Local import to avoid cycles.

    lines = ["Code Policy Capsule Rules:"]
    for rule in CPC_RULES.values():
        detail = textwrap.fill(rule.detail, width=88, subsequent_indent="  ")
        lines.append(f"- {rule.code} :: {rule.title}")
        lines.append(f"  {detail}")
    return "\n".join(lines)


__all__ = [
    "GateReport",
    "StaticCheck",
    "StaticCheckResult",
    "run_policy_and_static",
    "describe_policy_capsule",
]
