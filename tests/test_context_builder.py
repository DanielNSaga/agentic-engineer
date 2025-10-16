from __future__ import annotations

from pathlib import Path

from ae.context_builder import ContextBuilder
from ae.memory.code_index.symbol_index import SymbolIndex


def _bootstrap_index(repo_root: Path) -> None:
    data_root = repo_root / "data"
    index_root = data_root / "index"
    index_root.mkdir(parents=True, exist_ok=True)

    source_dir = repo_root / "src"
    source_dir.mkdir(parents=True, exist_ok=True)

    file_path = source_dir / "greetings.py"
    source = """class Greeter:
    def __init__(self, name: str) -> None:
        self.name = name

    def greeting(self) -> str:
        return f"Hello, {self.name}!"
"""
    file_path.write_text(source, encoding="utf-8")

    index = SymbolIndex(index_root / "symbols.json")
    index.index_file(Path("src/greetings.py"), source)


def test_context_builder_includes_symbol_slices(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _bootstrap_index(repo_root)

    guidance = [
        "Focus changes on src/greetings.py.",
        "Do not modify repository root metadata files such as README.md or pyproject.toml.",
    ]

    builder = ContextBuilder(
        policy_text="Policy reminder.",
        repo_root=repo_root,
        data_root=repo_root / "data",
        guidance=guidance,
    )

    request = {
        "task_id": "task-1",
        "goal": "Add a friendly greeting output.",
        "touched_files": ["src/greetings.py"],
        "constraints": ["Preserve the public interface."],
    }

    package = builder.build("implement", request)

    assert "Policy reminder." in package.system_prompt
    assert "## Code Context" in package.user_prompt
    assert "src/greetings.py" in package.user_prompt
    assert "class Greeter" in package.user_prompt
    assert "## Project Guidance" in package.user_prompt
    for line in guidance:
        assert f"- {line}" in package.user_prompt

    metadata = package.metadata
    assert metadata["token_estimate"] <= metadata["token_budget"]
    symbol_entries = [
        entry for entry in metadata["sections"] if entry["label"].startswith("code:symbol:")
    ]
    assert symbol_entries
    assert any(entry["included"] for entry in symbol_entries)
    guidance_entries = [
        entry for entry in metadata["sections"] if entry["label"] == "project_guidance"
    ]
    assert guidance_entries and guidance_entries[0]["included"]
    assert metadata["guidance"] == guidance


def test_context_builder_includes_snippets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = repo_root / "data"
    data_root.mkdir()

    builder = ContextBuilder(
        repo_root=repo_root,
        data_root=data_root,
    )

    snippet_text = "def example() -> None:\n    pass"
    request = {
        "task_id": "task-2",
        "diff_goal": "Inspect helper",
        "snippets": [
            {
                "path": "src/example.py",
                "start_line": 1,
                "end_line": 2,
                "content": snippet_text,
                "reason": "Need canonical implementation",
            }
        ],
    }

    package = builder.build("implement", request)

    assert "Snippet: src/example.py" in package.user_prompt
    assert snippet_text in package.user_prompt
    assert package.metadata["snippets"] == [
        {
            "path": "src/example.py",
            "start_line": 1,
            "end_line": 2,
            "reason": "Need canonical implementation",
        }
    ]
    request_snapshot = package.metadata["request"]
    assert "snippets" in request_snapshot
    assert request_snapshot["snippets"] == [
        {
            "path": "src/example.py",
            "start_line": 1,
            "end_line": 2,
            "reason": "Need canonical implementation",
        }
    ]

def test_context_builder_surfaces_static_finding_snippets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = repo_root / "data"
    data_root.mkdir()

    source_path = repo_root / "src/pkg/module.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "alpha()\nproblem_line()\nomega()\n",
        encoding="utf-8",
    )

    builder = ContextBuilder(
        repo_root=repo_root,
        data_root=data_root,
    )

    request = {
        "task_id": "task-3",
        "violations": ["python: src/pkg/module.py:2 runtime warning"],
        "static_findings": [
            {
                "path": "src/pkg/module.py",
                "line_start": 2,
                "line_end": 2,
                "message": "Example static finding",
            }
        ],
    }

    package = builder.build("fix_violations", request)

    assert "Snippet: src/pkg/module.py" in package.user_prompt
    assert "problem_line()" in package.user_prompt

    snippet_entries = [
        entry
        for entry in package.metadata["sections"]
        if entry["label"].startswith("snippet:src/pkg/module.py")
    ]
    assert snippet_entries and snippet_entries[0]["included"]
    request_snapshot = package.metadata["request"]
    assert request_snapshot["static_findings"][0]["path"] == "src/pkg/module.py"


def test_context_builder_exposes_workspace_metadata(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = repo_root / "data"
    data_root.mkdir()

    requirements_path = repo_root / "requirements.txt"
    requirements_path.write_text("click==8.1.7\npyyaml==6.0.1\n", encoding="utf-8")

    module_path = repo_root / "src/app"
    module_path.mkdir(parents=True, exist_ok=True)
    module_path.joinpath("__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    module_path.joinpath("main.py").write_text(
        "class Runner:\n"
        "    def __init__(self) -> None:\n"
        "        self.count = 0\n\n"
        "    def run(self, limit: int) -> int:\n"
        "        for _ in range(limit):\n"
        "            self.count += 1\n"
        "        return self.count\n\n"
        "def helper(flag: bool) -> str:\n"
        "    return 'ok' if flag else 'nope'\n",
        encoding="utf-8",
    )

    builder = ContextBuilder(
        repo_root=repo_root,
        data_root=data_root,
    )

    package = builder.build("implement", {"task_id": "workspace-check"})

    assert "## Workspace Python Outline" in package.user_prompt
    assert "### src/app/main.py :: symbol outline" in package.user_prompt
    assert "- class Runner" in package.user_prompt
    assert "helper(flag: bool) -> str" in package.user_prompt
    assert "## Workspace Requirements" in package.user_prompt
    assert "click==8.1.7" in package.user_prompt
    assert "### requirements.txt" in package.user_prompt

    entries = {entry["label"]: entry for entry in package.metadata["sections"]}
    assert entries["workspace_python_outline"]["included"]
    assert entries["workspace_requirements"]["included"]


def test_context_builder_normalises_prefixed_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = repo_root / "data"
    data_root.mkdir()

    tests_dir = repo_root / "tests"
    tests_dir.mkdir()
    target = tests_dir / "test_example.py"
    target.write_text("assert True\n", encoding="utf-8")

    builder = ContextBuilder(repo_root=repo_root, data_root=data_root)

    normalised = builder._normalize_repo_path("password-manager/tests/test_example.py")

    assert normalised == "tests/test_example.py"

def test_context_builder_respects_token_budget(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _bootstrap_index(repo_root)

    builder = ContextBuilder(
        policy_text="Policy reminder.",
        repo_root=repo_root,
        data_root=repo_root / "data",
        token_budget=40,
    )

    request = {
        "task_id": "task-1",
        "goal": "Add a friendly greeting output.",
        "touched_files": ["src/greetings.py"],
        "constraints": ["Preserve the public interface."],
    }

    package = builder.build("implement", request)
    metadata = package.metadata

    assert metadata["token_estimate"] <= metadata["token_budget"]
    assert any(
        entry["label"] == "instructions" and entry["included"] for entry in metadata["sections"]
    )
    assert not any(
        entry["label"].startswith("code:") and entry["included"] for entry in metadata["sections"]
    )


def test_context_builder_emits_repo_tree_for_all_phases(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    src_dir = repo_root / "src"
    src_dir.mkdir()
    (src_dir / "greetings.py").write_text("print('hi')\n", encoding="utf-8")
    empty_tests = repo_root / "tests"
    empty_tests.mkdir()
    data_root = repo_root / "data"
    data_root.mkdir()

    builder = ContextBuilder(
        repo_root=repo_root,
        data_root=data_root,
    )

    phases = ["plan", "analyze", "design", "implement", "plan_adjust", "fix_violations"]
    for phase in phases:
        package = builder.build(phase, {"goal": "demo"})
        assert "## Repository Tree" in package.user_prompt
        assert "src/" in package.user_prompt
        assert "README.md" in package.user_prompt
        assert "tests/" in package.user_prompt
        assert any(
            entry["label"] == "repo_tree" and entry["included"] for entry in package.metadata["sections"]
        )


def test_context_builder_includes_plan_adjust_contract(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = repo_root / "data"
    data_root.mkdir()

    builder = ContextBuilder(
        repo_root=repo_root,
        data_root=data_root,
    )

    request = {
        "plan_id": "PLAN-1",
        "task_id": "TASK-1",
        "reason": "Iteration failed validation.",
        "suggested_changes": ["Add schema contract to prompts."],
    }

    package = builder.build("plan_adjust", request)

    assert "## Plan Adjust Response Contract" in package.user_prompt
    assert "\"adjustments\"" in package.user_prompt
    contract_entries = [
        entry for entry in package.metadata["sections"] if entry["label"] == "plan_adjust_response_contract"
    ]
    assert contract_entries and contract_entries[0]["included"]


def test_context_builder_structured_implement_contract(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = repo_root / "data"
    data_root.mkdir()

    builder = ContextBuilder(
        repo_root=repo_root,
        data_root=data_root,
        policy_text="Policy reminder.",
    )

    request = {
        "task_id": "TASK-STRUCT",
        "diff_goal": "Update helper.",
        "touched_files": ["src/demo.py"],
        "structured_edits_only": True,
    }

    package = builder.build("implement", request)

    assert "Populate `files`" in package.user_prompt
    assert "Use `edits` for targeted updates." in package.user_prompt
    assert "Leave `diff` empty" in package.user_prompt
    assert "Provide a `no_op_reason`" in package.user_prompt
    assert "`diff` must be a valid unified diff" not in package.user_prompt


def test_context_builder_fix_violations_contract(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = repo_root / "data"
    data_root.mkdir()

    builder = ContextBuilder(
        repo_root=repo_root,
        data_root=data_root,
    )

    request = {
        "task_id": "TASK-FIX",
        "violations": ["src/demo.py:10: lint error"],
        "suspect_files": ["src/demo.py"],
    }

    package = builder.build("fix_violations", request)

    assert "## Fix Violations Response Contract" in package.user_prompt
    assert "leave `patch` empty" in package.user_prompt.lower()
    assert "Prioritise fixes in the flagged files (src/demo.py)" in package.user_prompt
    assert "Only populate `no_op_reason`" in package.user_prompt
