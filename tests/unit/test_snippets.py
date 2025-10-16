from __future__ import annotations

from pathlib import Path

from ae.tools.snippets import SnippetRequest, collect_snippets


def test_collect_snippets_emits_missing_file_placeholder(tmp_path: Path) -> None:
    repo_root = tmp_path
    request = SnippetRequest(path="src/missing_module.py", reason="Need file content")
    snippets = collect_snippets(repo_root, [request])

    assert snippets, "expected missing-file placeholder snippet"
    snippet = snippets[0]
    assert snippet.path == "src/missing_module.py"
    assert "missing_module.py" in snippet.content
    assert "does not exist" in snippet.content.lower()
    assert snippet.start_line == 1
    assert snippet.end_line == 1


def test_collect_snippets_normalises_prefixed_request(tmp_path: Path) -> None:
    repo_root = tmp_path
    tests_dir = repo_root / "tests"
    tests_dir.mkdir()
    target_file = tests_dir / "test_sample.py"
    target_file.write_text(
        "def test_sample() -> None:\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )

    request = SnippetRequest(path="password-manager/tests/test_sample.py")
    snippets = collect_snippets(repo_root, [request])

    assert snippets, "expected snippet for existing test file"
    snippet = snippets[0]
    assert snippet.path == "tests/test_sample.py"
    assert "def test_sample" in snippet.content
