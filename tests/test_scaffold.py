from __future__ import annotations

from pathlib import Path

from ae.tools.scaffold import ensure_program_framework, ensure_project_scaffold


def test_ensure_project_scaffold_creates_precompliant_files(tmp_path: Path) -> None:
    created = ensure_project_scaffold(tmp_path, project_name="Demo App")

    expected = {
        Path("pyproject.toml"),
        Path(".gitignore"),
        Path("README.md"),
        Path("config.yaml"),
        Path("policy"),
        Path("policy/capsule.txt"),
        Path("src"),
        Path("src/demo_app"),
        Path("tests"),
    }
    created_set = set(created)
    assert expected.issubset(created_set)

    assert (tmp_path / "pyproject.toml").exists()
    assert not (tmp_path / "src" / "demo_app" / "cli.py").exists()
    assert not (tmp_path / "src" / "__init__.py").exists()
    src_init = tmp_path / "src" / "demo_app" / "__init__.py"
    tests_init = tmp_path / "tests" / "__init__.py"
    assert src_init.exists()
    assert src_init.read_text(encoding="utf-8").startswith('"""Initial package marker')
    assert tests_init.exists()
    assert tests_init.read_text(encoding="utf-8").startswith('"""Tests for')
    assert not (tmp_path / "tests" / "test_placeholder.py").exists()
    assert not (tmp_path / "tests" / "conftest.py").exists()
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8").count("policy:") == 1
    assert (tmp_path / "policy" / "capsule.txt").read_text(encoding="utf-8").strip()
    second_run = ensure_project_scaffold(tmp_path, project_name="Demo App")
    assert second_run == []


def test_ensure_project_scaffold_legacy_toggle(tmp_path: Path) -> None:
    created = ensure_project_scaffold(tmp_path, project_name="Legacy App", precompliant=False)
    created_set = set(created)
    assert Path("config.yaml") not in created_set
    assert Path("policy/capsule.txt") not in created_set
    assert Path(".gitignore") in created_set
    assert (tmp_path / "src").exists()
    assert (tmp_path / "src" / "legacy_app").exists()
    assert (tmp_path / "tests").exists()
    assert (tmp_path / "src" / "legacy_app" / "__init__.py").exists()
    assert (tmp_path / "tests" / "__init__.py").exists()
    assert not (tmp_path / "tests" / "conftest.py").exists()
    assert not (tmp_path / "tests" / "test_placeholder.py").exists()


def test_package_name_sanitization(tmp_path: Path) -> None:
    ensure_project_scaffold(tmp_path, project_name="123 Cool App!")
    assert (tmp_path / "src" / "app_123_cool_app").exists()


def test_runtime_gitignore_left_unchanged(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"demo\"\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("node_modules/\n", encoding="utf-8")

    created = ensure_project_scaffold(tmp_path, project_name="Existing App")

    gitignore_content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert gitignore_content == "node_modules/\n"
    assert created == []


def test_ensure_program_framework_creates_package_from_goal(tmp_path: Path) -> None:
    config = {
        "iteration": {
            "goal": "Implement a CLI within src/password_manager and corresponding tests/password_manager suite.",
        },
        "context": {
            "guidance": [
                "Keep all new application code under src/password_manager.",
                "Add pytest modules under tests/password_manager.",
            ]
        },
    }

    created = ensure_program_framework(tmp_path, config=config)
    created_set = set(created)

    expected_paths = {
        Path("src"),
        Path("src/password_manager"),
        Path("tests"),
        Path("tests/password_manager"),
    }
    assert expected_paths.issubset(created_set)

    assert (tmp_path / "src" / "password_manager" / "__init__.py").exists()
    assert not (tmp_path / "src" / "password_manager" / "cli.py").exists()
    assert not (tmp_path / "src" / "password_manager" / "py.typed").exists()
    assert (tmp_path / "tests" / "__init__.py").exists()
    assert (tmp_path / "tests" / "password_manager" / "__init__.py").exists()


def test_scaffold_respects_root_scaffold_guidance(tmp_path: Path) -> None:
    config_text = """
context:
  guidance:
    - Do not create or modify repository root scaffolding files such as config.yaml, or pyproject.toml.
iteration:
  goal: Implement a secure password manager CLI.
project:
  name: Password Manager
"""
    (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")

    created = ensure_project_scaffold(tmp_path, project_name="Password Manager")
    created_set = set(created)

    assert Path("pyproject.toml") in created_set
    assert (tmp_path / "pyproject.toml").exists()
    assert Path(".gitignore") not in created_set
    assert not (tmp_path / ".gitignore").exists()

    expected = {
        Path("src"),
        Path("src/password_manager"),
        Path("tests"),
        Path("tests/password_manager"),
    }
    assert expected.issubset(created_set)

    assert (tmp_path / "src" / "password_manager" / "__init__.py").exists()
    assert not (tmp_path / "src" / "password_manager" / "cli.py").exists()
    assert not (tmp_path / "src" / "password_manager" / "py.typed").exists()
    assert (tmp_path / "tests" / "__init__.py").exists()
    assert (tmp_path / "tests" / "password_manager" / "__init__.py").exists()
