from __future__ import annotations

import subprocess
from pathlib import Path

from ae.memory.code_index.indexer import CodeIndexer


def run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_code_index_pipeline(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    run_git(repo, "init")
    run_git(repo, "config", "user.email", "agent@example.com")
    run_git(repo, "config", "user.name", "Agentic Engineer")

    initial_main = "def placeholder() -> None:\n    pass\n"
    initial_helper = "def helper() -> None:\n    pass\n"
    write_file(repo / "src" / "main.py", initial_main)
    write_file(repo / "src" / "helper.py", initial_helper)

    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", "Initial content")

    main_source = '''from __future__ import annotations

import math

from .helper import format_name


class Greeter:
    def __init__(self, name: str) -> None:
        self.name = name

    def greeting(self) -> str:
        return format_name(self.name)


def square_root(value: float) -> float:
    return math.sqrt(value)
'''

    helper_source = '''def format_name(name: str) -> str:
    return f"Hello, {name}"
'''

    write_file(repo / "src" / "main.py", main_source)
    write_file(repo / "src" / "helper.py", helper_source)

    indexer = CodeIndexer(repo_root=repo, data_root=repo / "data")
    processed = indexer.reindex()

    processed_paths = {path.as_posix() for path in processed}
    assert "src/main.py" in processed_paths
    assert "src/helper.py" in processed_paths

    index_root = repo / "data" / "index"
    assert (index_root / "symbols.json").exists()
    assert (index_root / "text.sqlite").exists()
    assert (index_root / "embeddings.json").exists()
    assert (index_root / "graph.json").exists()

    greeter_symbols = indexer.symbol_index.query("Greeter")
    assert greeter_symbols
    greeter = greeter_symbols[0]
    assert greeter.path == "src/main.py"
    assert greeter.start < greeter.end

    greeting_symbols = indexer.symbol_index.query("greeting")
    assert any(record.kind == "method" for record in greeting_symbols)

    text_hits = indexer.text_index.search("Hello")
    assert any(hit.endswith("helper.py") for hit in text_hits)

    embedding_hits = indexer.embedding_index.search("greeting", limit=2)
    assert embedding_hits
    assert embedding_hits[0][0] in {"src/main.py", "src/helper.py"}

    imports = indexer.graph_index.get_imports("src/main.py")
    assert "math" in imports
    assert "src.helper" in imports
