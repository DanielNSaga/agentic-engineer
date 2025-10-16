from __future__ import annotations

import json
from pathlib import Path

from ae.planning.bootstrap import _collect_repo_summary


def _seed_python_project(root: Path) -> None:
    (root / "src" / "pkg" / "storage").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)

    (root / "src" / "pkg" / "__init__.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (root / "src" / "pkg" / "storage" / "__init__.py").write_text(
        "def load():\n    return 1\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_storage.py").write_text(
        "from pkg import storage\n\n\ndef test_load():\n    storage.load()\n",
        encoding="utf-8",
    )


def test_workspace_contracts_capture_module_index_and_test_imports(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    _seed_python_project(repo_root)

    index_dir = repo_root / "data" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_dir.joinpath("symbols.json").write_text(
        json.dumps(
            {
                "src/pkg/storage/__init__.py": [
                    {
                        "name": "load",
                        "qualified_name": "pkg.storage.load",
                        "kind": "function",
                        "path": "src/pkg/storage/__init__.py",
                        "signature": "load()",
                        "start": 0,
                        "end": 10,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = _collect_repo_summary(repo_root, {"project": {}, "iteration": {}})
    contracts = summary.get("workspace_contracts")
    assert contracts is not None

    module_index = contracts.get("module_index")
    assert module_index is not None
    assert module_index["pkg"] == "src/pkg/__init__.py"
    assert module_index["pkg.storage"] == "src/pkg/storage/__init__.py"

    test_imports = contracts.get("test_imports")
    assert test_imports is not None
    assert "pkg" in test_imports
    assert "pkg.storage" in test_imports

    mapped = contracts.get("test_module_paths")
    assert mapped is not None
    assert mapped["pkg.storage"] == "src/pkg/storage/__init__.py"

    api_hints = contracts.get("api_hints")
    assert api_hints is not None
    assert any(hint.get("qualified_name") == "pkg.storage.load" for hint in api_hints)


def test_workspace_contracts_include_project_layout_rules() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    summary = _collect_repo_summary(repo_root, {"project": {}, "iteration": {}})
    contracts = summary.get("workspace_contracts")
    assert contracts is not None

    storage_contract = contracts.get("storage_entry_points")
    assert storage_contract is not None
    memory_store = storage_contract.get("memory_store")
    assert memory_store is not None
    assert memory_store.get("default_db_path") == "data/ae.sqlite"

    cli_entry_points = contracts.get("cli_entry_points")
    assert cli_entry_points is not None
    assert any(entry.get("command") == "init" for entry in cli_entry_points)

    layout_rules = contracts.get("layout_rules")
    assert layout_rules is not None
    assert any("data" in rule for rule in layout_rules)
