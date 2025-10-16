from __future__ import annotations

from pathlib import Path

import textwrap
import yaml

from ae.tools.gates import GateReport, run_policy_and_static


def _base_config(static_checks: list[dict[str, object]] | None = None) -> dict[str, object]:
    policy: dict[str, object] = {"capsule_path": "capsule.txt"}
    if static_checks is not None:
        policy["static_checks"] = static_checks

    return {
        "project": {"name": "demo", "repo_root": "."},
        "iteration": {"current": 0, "max_calls": 1, "goal": "demo"},
        "policy": policy,
        "paths": {"data": "data"},
    }


STRICT_PYPROJECT = textwrap.dedent(
    """
    [project]
    name = "demo"
    version = "0.0.0"
    """
).strip()


def _write_repo(tmp_path: Path, config: dict[str, object]) -> tuple[Path, Path]:
    repo_root = tmp_path
    config_path = repo_root / "config.yaml"
    pyproject_path = repo_root / "pyproject.toml"
    capsule_path = repo_root / "capsule.txt"

    capsule_path.write_text("policy capsule", encoding="utf-8")
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)

    pyproject_path.write_text(STRICT_PYPROJECT + "\n", encoding="utf-8")
    return repo_root, config_path


def test_run_policy_and_static_reports_policy_failure(tmp_path: Path) -> None:
    config = _base_config()
    config.pop("paths")  # Force CFG001.

    repo_root, config_path = _write_repo(tmp_path, config)
    report = run_policy_and_static(config_path=config_path, repo_root=repo_root)

    assert isinstance(report, GateReport)
    assert report.policy_violations


def test_run_policy_and_static_skips_optional_checks_when_missing(tmp_path: Path) -> None:
    config = _base_config(
        static_checks=[
            {"name": "absent", "command": ["definitely-not-installed"], "optional": True}
        ]
    )

    repo_root, config_path = _write_repo(tmp_path, config)
    report = run_policy_and_static(config_path=config_path, repo_root=repo_root)

    assert report.policy_violations == []
    assert report.static_results[0].status == "skipped"


def test_run_policy_and_static_reports_static_failures(tmp_path: Path) -> None:
    config = _base_config(
        static_checks=[
            {
                "name": "failing",
                "command": ["python", "-c", "import sys; sys.exit(1)"],
                "optional": False,
            }
        ]
    )

    repo_root, config_path = _write_repo(tmp_path, config)
    report = run_policy_and_static(config_path=config_path, repo_root=repo_root)

    assert report.policy_violations == []
    assert report.static_results[0].status == "failed"
