from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import pytest
import yaml

from ae.structured import StructuredEditOperation, StructuredFileArtifact
from ae.tools.patch import (
    PatchError,
    apply_patch,
    build_structured_patch,
    canonicalise_unified_diff,
)
from ae.tools.pytest_runner import run_pytest
from ae.tools.vcs import GitRepository


def _prepare_repo(repo_root: Path) -> tuple[GitRepository, Path]:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    repo = GitRepository(repo_root)
    repo.git("config", "user.email", "agent@example.com")
    repo.git("config", "user.name", "Agentic Engineer")

    tracked = repo_root / "tracked.txt"
    tracked.write_text("alpha\n", encoding="utf-8")
    repo.git("add", "tracked.txt")
    repo.git("commit", "-m", "init", check=True)
    return repo, tracked


def test_build_structured_patch_generates_diff(tmp_path: Path) -> None:
    repo, tracked = _prepare_repo(tmp_path)
    artifact = StructuredFileArtifact(path="tracked.txt", content="alpha\nbeta\n")

    diff = build_structured_patch(repo_root=repo.root, files=[artifact], edits=[])

    assert diff.startswith("diff --git a/tracked.txt b/tracked.txt")
    apply_patch(diff, repo_root=repo.root)
    assert tracked.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_build_structured_patch_applies_edits(tmp_path: Path) -> None:
    repo, tracked = _prepare_repo(tmp_path)
    edit = StructuredEditOperation(path="tracked.txt", action="insert", start_line=2, content="beta\n")

    diff = build_structured_patch(repo_root=repo.root, files=[], edits=[edit])

    assert "@@ -1 +1,2 @@" in diff
    apply_patch(diff, repo_root=repo.root)
    assert tracked.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_build_structured_patch_replaces_entire_file_without_end_line(tmp_path: Path) -> None:
    repo, tracked = _prepare_repo(tmp_path)
    edit = StructuredEditOperation(
        path="tracked.txt",
        action="replace",
        start_line=1,
        content="gamma\ndelta\n",
    )

    diff = build_structured_patch(repo_root=repo.root, files=[], edits=[edit])

    apply_patch(diff, repo_root=repo.root)
    assert tracked.read_text(encoding="utf-8") == "gamma\ndelta\n"


def test_build_structured_patch_deletes_file(tmp_path: Path) -> None:
    repo, tracked = _prepare_repo(tmp_path)
    edit = StructuredEditOperation(path="tracked.txt", action="delete")

    diff = build_structured_patch(repo_root=repo.root, files=[], edits=[edit])

    apply_patch(diff, repo_root=repo.root)
    assert not tracked.exists()
    assert Path("tracked.txt") in repo.working_tree_changes()


def test_build_structured_patch_deletes_directory(tmp_path: Path) -> None:
    repo, tracked = _prepare_repo(tmp_path)

    package_root = tmp_path / "pkg"
    package_root.mkdir()
    module = package_root / "module.py"
    module.write_text("print('hello')\n", encoding="utf-8")
    repo.git("add", "pkg/module.py")
    repo.git("commit", "-m", "add pkg", check=True)

    edit = StructuredEditOperation(path="pkg", action="delete")

    diff = build_structured_patch(repo_root=repo.root, files=[], edits=[edit])

    apply_patch(diff, repo_root=repo.root)
    assert not package_root.exists()
    assert Path("pkg/module.py") in repo.working_tree_changes()
    assert tracked.exists()


def test_canonicalise_unified_diff_handles_apply_patch_format(tmp_path: Path) -> None:
    repo, tracked = _prepare_repo(tmp_path)
    patch = textwrap.dedent(
        """
        *** Begin Patch
        *** Update File: tracked.txt
        @@ -1 +1 @@
        -alpha
        +bravo
        *** End Patch
        """
    ).strip()

    diff = canonicalise_unified_diff(patch, repo_root=repo.root)

    assert diff.startswith("diff --git a/tracked.txt b/tracked.txt")
    apply_patch(diff, repo_root=repo.root)
    assert tracked.read_text(encoding="utf-8") == "bravo\n"


def test_apply_patch_applies_diff(tmp_path: Path) -> None:
    repo, tracked = _prepare_repo(tmp_path)

    tracked.write_text("alpha\nbeta\n", encoding="utf-8")
    diff = repo.diff("tracked.txt")
    repo.git("restore", "--worktree", "--staged", "--", "tracked.txt")

    result = apply_patch(diff, repo_root=repo.root)

    assert tracked.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert Path("tracked.txt") in result.paths
    assert repo.working_tree_changes() == [Path("tracked.txt")]


def test_apply_patch_rejects_unsafe_paths(tmp_path: Path) -> None:
    repo, _ = _prepare_repo(tmp_path)

    malicious_patch = textwrap.dedent(
        """
        diff --git a/../evil.txt b/../evil.txt
        --- a/../evil.txt
        +++ b/../evil.txt
        @@ -0,0 +1 @@
        +boom
        """
    ).strip()

    with pytest.raises(PatchError):
        apply_patch(malicious_patch, repo_root=repo.root)


def test_apply_patch_corrects_hunk_mismatch(tmp_path: Path) -> None:
    repo, _ = _prepare_repo(tmp_path)

    malformed_patch = textwrap.dedent(
        """
        diff --git a/new.txt b/new.txt
        new file mode 100644
        index 0000000..1111111
        --- /dev/null
        +++ b/new.txt
        @@ -0,0 +1,3 @@
        +alpha
        +beta
        """
    ).strip()

    result = apply_patch(malformed_patch, repo_root=repo.root)

    created = tmp_path / "new.txt"
    assert created.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert Path("new.txt") in result.paths
    assert Path("new.txt") in repo.working_tree_changes()


def test_apply_patch_skips_identical_new_file(tmp_path: Path) -> None:
    repo, _ = _prepare_repo(tmp_path)

    target = tmp_path / "existing.txt"
    target.write_text("same\ncontent\n", encoding="utf-8")
    repo.git("add", "existing.txt")
    repo.git("commit", "-m", "add existing", check=True)

    patch = textwrap.dedent(
        """
        diff --git a/existing.txt b/existing.txt
        new file mode 100644
        index 0000000..1111111
        --- /dev/null
        +++ b/existing.txt
        @@ -0,0 +1,2 @@
        +same
        +content
        """
    ).strip()

    result = apply_patch(patch, repo_root=repo.root)

    assert result.paths == ()
    assert target.read_text(encoding="utf-8") == "same\ncontent\n"
    assert repo.working_tree_changes() == []


def test_apply_patch_replaces_existing_new_file(tmp_path: Path) -> None:
    repo, _ = _prepare_repo(tmp_path)

    target = tmp_path / "config.cfg"
    target.write_text("old=1\n", encoding="utf-8")
    repo.git("add", "config.cfg")
    repo.git("commit", "-m", "add old config", check=True)

    patch = textwrap.dedent(
        """
        diff --git a/config.cfg b/config.cfg
        new file mode 100644
        index 0000000..1111111
        --- /dev/null
        +++ b/config.cfg
        @@ -0,0 +1,2 @@
        +username=alice
        +password=secret
        """
    ).strip()

    result = apply_patch(patch, repo_root=repo.root)

    backup = tmp_path / "config.cfg.pre_patch"
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == "old=1\n"

    assert target.read_text(encoding="utf-8") == "username=alice\npassword=secret\n"
    assert Path("config.cfg") in result.paths
    assert Path("config.cfg.pre_patch") in result.paths


def test_apply_patch_moves_top_level_package_into_src(tmp_path: Path) -> None:
    repo, _ = _prepare_repo(tmp_path)

    src_pkg = tmp_path / "src" / "password_manager"
    src_pkg.mkdir(parents=True)
    (src_pkg / "__init__.py").write_text("__version__ = '0.0.0'\n", encoding="utf-8")

    patch = textwrap.dedent(
        """
        diff --git a/password_manager/cli.py b/password_manager/cli.py
        new file mode 100644
        index 0000000..1111111
        --- /dev/null
        +++ b/password_manager/cli.py
        @@ -0,0 +1,2 @@
        +def main():
        +    return 0
        """
    ).strip()

    result = apply_patch(patch, repo_root=repo.root)

    moved = Path("src/password_manager/cli.py")
    assert moved in result.paths
    assert (tmp_path / moved).exists()
    assert not (tmp_path / "password_manager").exists()


def test_apply_patch_rejects_crlf_line_endings(tmp_path: Path) -> None:
    repo, _ = _prepare_repo(tmp_path)
    patch = "diff --git a/tracked.txt b/tracked.txt\r\n"
    patch += "--- a/tracked.txt\r\n"
    patch += "+++ b/tracked.txt\r\n"
    patch += "@@ -1 +1 @@\r\n"
    patch += "-alpha\r\n"
    patch += "+bravo\r\n"

    with pytest.raises(PatchError) as excinfo:
        apply_patch(patch, repo_root=repo.root)

    assert "LF line endings" in str(excinfo.value)


def test_apply_patch_enforces_allowed_paths_from_guidance(tmp_path: Path) -> None:
    repo, _ = _prepare_repo(tmp_path)
    config = {
        "project": {"name": "demo"},
        "context": {
            "guidance": [
                "Keep all new application code under src/app and corresponding test modules.",
                "Do not create or modify repository root scaffolding files such as README.md, config.yaml, or pyproject.toml.",
            ]
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    patch = textwrap.dedent(
        """
        diff --git a/src/other/app.py b/src/other/app.py
        new file mode 100644
        index 0000000..1111111
        --- /dev/null
        +++ b/src/other/app.py
        @@ -0,0 +1,2 @@
        +VALUE = 1
        +print(VALUE)
        """
    ).strip()

    with pytest.raises(PatchError) as excinfo:
        apply_patch(patch, repo_root=repo.root)

    telemetry = excinfo.value.details.get("telemetry")
    assert telemetry is not None
    assert telemetry["guidance_violations"]
    assert any("outside allowed paths" in violation for violation in telemetry["guidance_violations"])


def test_guidance_placeholder_expands_package_name(tmp_path: Path) -> None:
    repo, _ = _prepare_repo(tmp_path)
    config = {
        "project": {"name": "Password-Manager"},
        "context": {
            "guidance": [
                "Place all new application code under src/<package_name> and mirror it with tests/<package_name> modules.",
            ]
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    patch = textwrap.dedent(
        """
        diff --git a/src/password_manager/core.py b/src/password_manager/core.py
        new file mode 100644
        index 0000000..1111111
        --- /dev/null
        +++ b/src/password_manager/core.py
        @@ -0,0 +1,2 @@
        +VALUE = 1
        +print(VALUE)
        """
    ).strip()

    result = apply_patch(patch, repo_root=repo.root)

    path = Path("src/password_manager/core.py")
    assert path in result.paths
    assert (tmp_path / path).exists()


def test_apply_patch_respects_patch_size_limit(tmp_path: Path) -> None:
    repo, _ = _prepare_repo(tmp_path)
    config = {
        "project": {"name": "demo"},
        "iteration": {"max_patch_bytes": 128},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    body = "\n".join("+line{}".format(i) for i in range(200))
    patch = textwrap.dedent(
        f"""
        diff --git a/src/app/huge.py b/src/app/huge.py
        new file mode 100644
        index 0000000..1111111
        --- /dev/null
        +++ b/src/app/huge.py
        @@ -0,0 +200 @@
        {body}
        """
    ).strip()

    with pytest.raises(PatchError) as excinfo:
        apply_patch(patch, repo_root=repo.root)

    telemetry = excinfo.value.details.get("telemetry")
    assert telemetry is not None
    assert "exceeds limit" in telemetry["guidance_violations"][0]


def test_apply_patch_emits_git_check_telemetry_on_failure(tmp_path: Path) -> None:
    repo, _ = _prepare_repo(tmp_path)

    patch = textwrap.dedent(
        """
        diff --git a/tracked.txt b/tracked.txt
        --- a/tracked.txt
        +++ b/tracked.txt
        @@ -1 +1 @@
        -omega
        +theta
        """
    ).strip()

    with pytest.raises(PatchError) as excinfo:
        apply_patch(patch, repo_root=repo.root)

    telemetry = excinfo.value.details.get("telemetry")
    assert telemetry is not None
    assert telemetry["check"]["returncode"] != 0
    assert telemetry["failing_hunks"]


def test_checkpoint_rollback_restores_state(tmp_path: Path) -> None:
    repo, tracked = _prepare_repo(tmp_path)

    persistent = tmp_path / "notes.txt"
    persistent.write_text("keep\n", encoding="utf-8")

    checkpoint = repo.create_checkpoint("baseline")

    tracked.write_text("alpha\nbeta\n", encoding="utf-8")
    scratch = tmp_path / "scratch.txt"
    scratch.write_text("temp\n", encoding="utf-8")

    checkpoint.rollback()

    assert tracked.read_text(encoding="utf-8") == "alpha\n"
    assert not scratch.exists()
    assert persistent.exists()
    assert repo.working_tree_changes(include_untracked=False) == []
    assert repo.working_tree_changes() == [Path("notes.txt")]


def test_run_pytest_reports_success(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ok.py").write_text(
        textwrap.dedent(
            """
            def test_ok():
                assert 2 + 2 == 4
            """
        ).lstrip(),
        encoding="utf-8",
    )

    result = run_pytest(["tests"], cwd=tmp_path)

    assert result.status == "passed"
    assert result.exit_code == 0
    assert result.collected == 1
    assert "1 passed" in (result.stdout + result.stderr)


def test_run_pytest_normalises_prefixed_paths(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        textwrap.dedent(
            """
            def test_marker() -> None:
                assert 1 + 1 == 2
            """
        ).lstrip(),
        encoding="utf-8",
    )

    result = run_pytest(["password-manager/tests/test_sample.py"], cwd=tmp_path)

    assert result.status == "passed"
    assert result.command == ("pytest", "tests/test_sample.py")
    assert result.collected == 1


def test_run_pytest_injects_src_path(tmp_path: Path) -> None:
    src_pkg = tmp_path / "src" / "sample_app"
    src_pkg.mkdir(parents=True)
    (src_pkg / "__init__.py").write_text(
        textwrap.dedent(
            """
            def marker() -> str:
                return "ok"
            """
        ).lstrip(),
        encoding="utf-8",
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_import.py").write_text(
        textwrap.dedent(
            """
            from sample_app import marker


            def test_marker() -> None:
                assert marker() == "ok"
            """
        ).lstrip(),
        encoding="utf-8",
    )

    result = run_pytest(["tests"], cwd=tmp_path)

    assert result.status == "passed"
    assert result.exit_code == 0


def test_run_pytest_installs_requirements_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root_requirement = tmp_path / "requirements.txt"
    root_requirement.write_text("cryptography>=40.0.0\n", encoding="utf-8")

    pkg_requirement = tmp_path / "src" / "pkg" / "requirements.txt"
    pkg_requirement.parent.mkdir(parents=True)
    pkg_requirement.write_text("pluggy\n", encoding="utf-8")

    calls: list[tuple[str, ...]] = []

    def fake_run(
        args: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> SimpleNamespace:
        calls.append(tuple(str(part) for part in args))
        if args[0] == sys.executable and tuple(args[1:3]) == ("-m", "pip"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[0] == "pytest":
            return SimpleNamespace(returncode=0, stdout="collected 0 items", stderr="")
        raise AssertionError(f"Unexpected invocation: {args!r}")

    fake_subprocess = SimpleNamespace(run=fake_run)
    monkeypatch.setattr("ae.tools.pytest_runner.subprocess", fake_subprocess)

    result = run_pytest(["tests"], cwd=tmp_path)

    pip_calls = [call for call in calls if call[0] == sys.executable]
    assert len(pip_calls) == 2
    assert (sys.executable, "-m", "pip", "install", "-r", str(root_requirement)) in pip_calls
    assert (sys.executable, "-m", "pip", "install", "-r", str(pkg_requirement)) in pip_calls
    assert result.status == "passed"

    sentinel = tmp_path / ".agentic-engineer" / "deps-installed.json"
    assert sentinel.exists()
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    assert root_requirement.relative_to(tmp_path).as_posix() in payload["files"]
    assert pkg_requirement.relative_to(tmp_path).as_posix() in payload["files"]

    calls.clear()
    result = run_pytest(["tests"], cwd=tmp_path)
    pip_calls = [call for call in calls if call[0] == sys.executable]
    assert not pip_calls
    assert result.status == "passed"


def test_run_pytest_reports_failures(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_fail.py").write_text(
        textwrap.dedent(
            """
            def test_fail():
                assert False
            """
        ).lstrip(),
        encoding="utf-8",
    )

    result = run_pytest(["tests"], cwd=tmp_path)

    assert result.status == "failed"
    assert result.exit_code != 0
    assert result.collected == 1
    assert "FAILED" in (result.stdout + result.stderr)
