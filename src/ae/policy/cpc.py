"""Code Policy Capsule (CPC) helpers.

The implementation is intentionally minimal: it focuses on validating the runtime
configuration, enforcing a baseline typing policy, and (optionally) sanity
checking the repository layout. The goal is to provide deterministic signals that
plug cleanly into the orchestrator as automated fix loops evolve.

The module exposes two public abstractions:

``CPC_RULES``
    Registry listing the available rules and their human readable
    descriptions.  This is primarily used for documentation and for
    surfacing summaries alongside violations.

``CPCChecker``
    A callable object that evaluates the enabled rules against the
    repository on disk and returns structured ``PolicyViolation``
    instances.

Each violation contains the rule code, an explanatory message, and the
file/line that triggered the failure.  Line numbers default to ``1`` to
indicate a file-level issue when a more precise location is not readily
available (for example when a required config section is entirely
missing).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(slots=True)
class PolicyViolation:
    """Structured representation of a CPC violation."""

    rule: str
    message: str
    path: Path | None = None
    line: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable view of the violation."""

        return {
            "rule": self.rule,
            "message": self.message,
            "path": self.path.as_posix() if self.path else None,
            "line": self.line,
        }


@dataclass(slots=True)
class RuleDefinition:
    """Metadata describing a policy rule."""

    code: str
    title: str
    detail: str


def _require_keys(
    data: Mapping[str, Any], required: Iterable[str]
) -> tuple[bool, List[str]]:
    """Return a boolean flag and sorted list of missing keys."""

    missing = sorted(key for key in required if key not in data)
    return not missing, missing


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk, guarding against unexpected types."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):  # pragma: no cover - defensive guard
        raise TypeError(f"Expected mapping at top level of {path}")
    return dict(data)


def _resolve_capsule_path(repo_root: Path, policy_data: Mapping[str, Any]) -> Path:
    """Resolve the configured capsule path relative to the repository root."""
    capsule_path = policy_data.get("capsule_path")
    if not capsule_path:
        return repo_root / "src" / "ae" / "policy" / "capsule.txt"
    candidate = Path(capsule_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate


def _check_cfg001(context: "CPCChecker") -> Iterable[PolicyViolation]:
    """Ensure the runtime configuration defines the expected top-level keys."""

    required_keys = ("project", "iteration", "policy", "paths")
    config_data = context.config_data
    ok, missing = _require_keys(config_data, required_keys)
    if ok:
        return []

    message = "Missing config section(s): " + ", ".join(missing)
    return [
        PolicyViolation(
            rule="CFG001",
            message=message,
            path=context.config_path,
            line=1,
        )
    ]


def _check_lay001(context: "CPCChecker") -> Iterable[PolicyViolation]:
    """Validate the repository obeys the expected layout contracts."""

    repo_root = context.repo_root
    expected = ["src", "tests", "src/ae/policy", "src/ae/tools"]
    missing = [item for item in expected if not (repo_root / item).exists()]

    if missing:
        return [
            PolicyViolation(
                rule="LAY001",
                message="Missing required repository paths: " + ", ".join(missing),
                path=repo_root,
                line=1,
            )
        ]

    capsule_path = _resolve_capsule_path(repo_root, context.policy_data)
    if not capsule_path.exists():
        return [
            PolicyViolation(
                rule="LAY001",
                message=f"Policy capsule not found at {capsule_path.as_posix()}",
                path=capsule_path,
                line=1,
            )
        ]

    capsule_text = capsule_path.read_text(encoding="utf-8").strip()
    if not capsule_text:
        return [
            PolicyViolation(
                rule="LAY001",
                message="Policy capsule is empty; expected guidance.",
                path=capsule_path,
                line=1,
            )
        ]

    return []


CPC_RULES: Dict[str, RuleDefinition] = {
    "CFG001": RuleDefinition(
        code="CFG001",
        title="Configuration completeness",
        detail="Ensure config.yaml exposes the required top-level sections so the agent "
        "can load runtime metadata, policy settings, and storage paths.",
    ),
    "LAY001": RuleDefinition(
        code="LAY001",
        title="Repository layout",
        detail="Verify key directories and the policy capsule exist.  Optional and only "
        "enabled when policy.enforce_layout is true.",
    ),
}


RuleHandler = Callable[["CPCChecker"], Iterable[PolicyViolation]]

RULE_DISPATCH: Dict[str, RuleHandler] = {
    "CFG001": _check_cfg001,
    "LAY001": _check_lay001,
}


class CPCChecker:
    """Evaluate the Code Policy Capsule rules against a repository."""

    def __init__(self, repo_root: Path, config_path: Path | None = None) -> None:
        self.repo_root = repo_root
        self.config_path = config_path or repo_root / "config.yaml"
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.config_data = _load_yaml(self.config_path)
        self.policy_data = self._resolve_policy_section(self.config_data)
        self.pyproject_path = repo_root / "pyproject.toml"

    @staticmethod
    def _resolve_policy_section(config_data: Mapping[str, Any]) -> Mapping[str, Any]:
        policy_section = config_data.get("policy") or {}
        if not isinstance(policy_section, Mapping):
            return {}
        return policy_section

    def enabled_rules(self) -> List[str]:
        policy_cfg = self.policy_data
        enforced = {"CFG001"}
        if bool(policy_cfg.get("enforce_layout")):
            enforced.add("LAY001")

        disabled: set[str] = set()
        disabled_raw = policy_cfg.get("disable_rules", [])
        if isinstance(disabled_raw, str):
            disabled.add(disabled_raw)
        elif isinstance(disabled_raw, Iterable):
            for item in disabled_raw:
                disabled.add(str(item))

        return sorted(rule for rule in enforced if rule not in disabled)

    def run(self) -> List[PolicyViolation]:
        """Execute enabled CPC rules and return collected violations."""

        violations: List[PolicyViolation] = []
        for code in self.enabled_rules():
            handler = RULE_DISPATCH.get(code)
            if handler is None:  # pragma: no cover - defensive guard
                continue
            for violation in handler(self):
                violations.append(violation)

        violations.sort(key=lambda item: (item.rule, item.path.as_posix() if item.path else ""))
        return violations


__all__ = [
    "CPCChecker",
    "CPC_RULES",
    "PolicyViolation",
    "RuleDefinition",
]
