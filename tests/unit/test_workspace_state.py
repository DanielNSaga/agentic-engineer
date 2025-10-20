from __future__ import annotations

from pathlib import Path

from ae.tools.vcs import GitRepository
from ae.tools.workspace_state import apply_workspace_state, capture_workspace_state


def test_capture_and_apply_workspace_state(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo = GitRepository.initialise(repo_root)

    tracked_path = repo.root / "src" / "module.py"
    tracked_path.parent.mkdir(parents=True, exist_ok=True)
    tracked_path.write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    repo.git("add", "--all")
    repo.git("commit", "-m", "Add module")

    # Modify tracked file and add an untracked payload.
    tracked_path.write_text("def value() -> int:\n    return 2\n", encoding="utf-8")
    untracked_path = repo.root / "notes.txt"
    untracked_path.write_text("pending work", encoding="utf-8")

    snapshot = capture_workspace_state(repo, checkpoint_label="chk-1")
    snapshot_dict = snapshot.to_dict()
    assert snapshot_dict["head"]
    assert "return 2" in (snapshot_dict.get("diff") or "")
    assert any(entry["path"] == "notes.txt" for entry in snapshot_dict.get("untracked", []))

    # Reset to clean state and drop untracked files.
    repo.git("reset", "--hard", snapshot_dict["head"], check=True)
    if untracked_path.exists():
        untracked_path.unlink()

    apply_workspace_state(repo, snapshot_dict)

    assert tracked_path.read_text(encoding="utf-8") == "def value() -> int:\n    return 2\n"
    assert untracked_path.exists()
    assert untracked_path.read_text(encoding="utf-8") == "pending work"
