from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass(slots=True)
class TinyRepo:
    """Fixture payload representing the synthetic repository under test."""

    root: Path
    config_path: Path

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Invoke ``python -m ae.cli`` with the provided arguments."""

        env = os.environ.copy()
        pythonpath = str(SRC)
        if env.get("PYTHONPATH"):
            pythonpath = os.pathsep.join([pythonpath, env["PYTHONPATH"]])
        env["PYTHONPATH"] = pythonpath

        command = [sys.executable, "-m", "ae.cli", *args]
        return subprocess.run(  # noqa: S603 - command constructed from known values
            command,
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )


@pytest.fixture()
def tiny_repo(tmp_path: Path) -> TinyRepo:
    """Create a tiny git repository with config + policy for CLI smoke tests."""

    repo_root = tmp_path / "tiny-repo"
    repo_root.mkdir()

    def run_git(*cmd: str) -> None:
        subprocess.run(
            ["git", *cmd],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

    run_git("init")
    run_git("config", "user.email", "agent@example.com")
    run_git("config", "user.name", "Agentic Engineer")

    (repo_root / "policy").mkdir()
    (repo_root / "policy" / "capsule.txt").write_text(
        "Operate safely. Prefer small, reversible changes.\n",
        encoding="utf-8",
    )

    src_dir = repo_root / "src" / "tiny_app"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text(
        textwrap.dedent(
            """
            \"\"\"Tiny app package used for CLI smoke tests.\"\"\"

            from .calculator import add

            __all__ = ["add"]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (src_dir / "calculator.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations


            def add(left: int, right: int) -> int:
                return left + right
            """
        ).lstrip(),
        encoding="utf-8",
    )

    tests_dir = repo_root / "tests"
    tests_dir.mkdir()
    (tests_dir / "conftest.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import sys
            from pathlib import Path

            ROOT = Path(__file__).resolve().parents[1]
            SRC = ROOT / "src"

            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (tests_dir / "test_calculator.py").write_text(
        textwrap.dedent(
            """
            from tiny_app import add


            def test_add_returns_sum() -> None:
                assert add(2, 3) == 5
            """
        ).lstrip(),
        encoding="utf-8",
    )

    (repo_root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [build-system]
            requires = ["setuptools"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "tiny-app"
            version = "0.0.1"
            description = "Fixture package for CLI smoke tests."
            """
        ).lstrip(),
        encoding="utf-8",
    )

    (repo_root / "config.yaml").write_text(
        textwrap.dedent(
            """
            project:
              name: tiny-agentic-repo
              description: Fixture repo for CLI smoke testing.
              repo_root: .
            iteration:
              current: 0
              goal: smoke-test
              max_calls: 10
            policy:
              capsule_path: policy/capsule.txt
              enable_checks: true
              fail_fast: false
            paths:
              data: data
              db_path: data/ae.sqlite
              logs: data/logs
              cache: data/cache
              config: config.yaml
            models:
              default: gpt-5-offline
            sandbox:
              mode: workspace-write
              network: disabled
              approvals: never
            """
        ).lstrip(),
        encoding="utf-8",
    )

    run_git("add", ".")
    run_git("commit", "-m", "Initial tiny repo state")

    return TinyRepo(root=repo_root, config_path=repo_root / "config.yaml")
