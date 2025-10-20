from __future__ import annotations

import json
from pathlib import Path

from ae.tools.phase_logs import load_phase_log
from ae.tools.phase_replay import PhaseReplayConfig, prepare_replay_workspace
from ae.tools.vcs import GitRepository
from ae.tools.workspace_state import capture_workspace_state


def _seed_repo(tmp_path: Path) -> GitRepository:
    repo = GitRepository.initialise(tmp_path)
    src_dir = repo.root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    tracked = src_dir / "app.py"
    tracked.write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    repo.git("add", "--all")
    repo.git("commit", "-m", "Initial content")
    return repo


def test_prepare_replay_workspace_applies_logged_snapshot(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path / "repo")

    tracked = repo.root / "src/app.py"
    tracked.write_text("def value() -> int:\n    return 2\n", encoding="utf-8")
    untracked = repo.root / "notes.txt"
    untracked.write_text("pending investigation\n", encoding="utf-8")

    snapshot = capture_workspace_state(repo, checkpoint_label="chk-task").to_dict()

    # Reset repository to clean state and commit a different change to simulate drift.
    repo.git("reset", "--hard", "HEAD")
    if untracked.exists():
        untracked.unlink()
    tracked.write_text("def value() -> int:\n    return 3\n", encoding="utf-8")
    repo.git("commit", "-am", "Apply fix")

    log_payload = {
        "phase": "diagnose",
        "request": {"plan_id": "plan-1", "task_id": "task-1"},
        "context": {"metadata": {"workspace_state": snapshot}},
        "attempts": [],
    }
    log_path = tmp_path / "phase_log.json"
    log_path.write_text(json.dumps(log_payload), encoding="utf-8")

    data_root = tmp_path / "data"
    config = PhaseReplayConfig(repo=repo, data_root=data_root, identifier="diagnose-task-1")
    workspace = prepare_replay_workspace(config, load_phase_log(log_path))

    restored_tracked = workspace.root / "src/app.py"
    restored_untracked = workspace.root / "notes.txt"
    assert restored_tracked.read_text(encoding="utf-8") == "def value() -> int:\n    return 2\n"
    assert restored_untracked.read_text(encoding="utf-8") == "pending investigation\n"
    assert workspace.snapshot_applied is True


def test_prepare_replay_workspace_handles_missing_snapshot(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path / "repo-missing")
    log_payload = {
        "phase": "diagnose",
        "request": {"plan_id": "plan-2", "task_id": "task-2"},
        "context": {"metadata": {}},
    }
    log_path = tmp_path / "phase_log_missing.json"
    log_path.write_text(json.dumps(log_payload), encoding="utf-8")

    data_root = tmp_path / "data"
    config = PhaseReplayConfig(repo=repo, data_root=data_root, identifier="diagnose-task-2")
    workspace = prepare_replay_workspace(
        config,
        load_phase_log(log_path),
        apply_snapshot=True,
    )

    assert workspace.snapshot_applied is False
    assert (workspace.root / "src/app.py").exists()
