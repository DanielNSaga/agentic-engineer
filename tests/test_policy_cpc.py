from __future__ import annotations

from pathlib import Path

import textwrap
import yaml

from ae.policy.cpc import CPCChecker


STRICT_PYPROJECT = textwrap.dedent(
    """
    [project]
    name = "demo"
    version = "0.0.0"
    """
).strip()


def _write_config(path: Path, *, include_paths: bool = True, enforce_layout: bool = False) -> None:
    policy: dict[str, object] = {"capsule_path": "capsule.txt"}
    if enforce_layout:
        policy["enforce_layout"] = True

    config: dict[str, object] = {
        "project": {"name": "demo", "repo_root": "."},
        "iteration": {"current": 0, "max_calls": 1, "goal": "demo"},
        "policy": policy,
    }

    if include_paths:
        config["paths"] = {"data": "data"}

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)


def _write_strict_pyproject(path: Path, *, missing_flags: list[str] | None = None) -> None:
    data = STRICT_PYPROJECT
    if missing_flags:
        for flag in missing_flags:
            data = data.replace(f"{flag} = true\n", "")
    path.write_text(data + "\n", encoding="utf-8")


def test_cpc_detects_missing_config_sections(tmp_path: Path) -> None:
    repo_root = tmp_path
    config_path = repo_root / "config.yaml"
    pyproject_path = repo_root / "pyproject.toml"
    capsule_path = repo_root / "capsule.txt"

    capsule_path.write_text("policy capsule", encoding="utf-8")
    _write_config(config_path, include_paths=False)
    _write_strict_pyproject(pyproject_path)

    checker = CPCChecker(repo_root=repo_root, config_path=config_path)
    violations = checker.run()

    assert any(violation.rule == "CFG001" for violation in violations)


def test_cpc_layout_rule_requires_capsule_when_enabled(tmp_path: Path) -> None:
    repo_root = tmp_path
    config_path = repo_root / "config.yaml"
    pyproject_path = repo_root / "pyproject.toml"

    # Prepare the expected repository layout.
    (repo_root / "src" / "ae" / "policy").mkdir(parents=True)
    (repo_root / "src" / "ae" / "tools").mkdir(parents=True)
    (repo_root / "tests").mkdir()

    capsule_path = repo_root / "src" / "ae" / "policy" / "capsule.txt"
    capsule_path.write_text("", encoding="utf-8")

    _write_config(config_path, enforce_layout=True)
    _write_strict_pyproject(pyproject_path)

    checker = CPCChecker(repo_root=repo_root, config_path=config_path)
    violations = checker.run()

    assert any(violation.rule == "LAY001" for violation in violations)
