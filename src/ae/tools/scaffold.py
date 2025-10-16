"""Project scaffolding helpers for first-time iteration runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import yaml

_MINIMAL_PYPROJECT_TEMPLATE = """[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_slug}"
version = "0.1.0"
description = "Skeleton for {project_title}."
readme = "README.md"
requires-python = ">=3.11"
authors = [{{name = "Agentic Engineer"}}]
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]
include = ["{package_name}*"]

[tool.setuptools.package-dir]
"" = "src"

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-q"
testpaths = ["tests"]

[tool.agentic_engineer.static_parsers]
version = 1

[tool.agentic_engineer.static_parsers.python]
identifiers = ["python"]
max_findings = 8

[[tool.agentic_engineer.static_parsers.python.patterns]]
type = "regex"
scope = "line"
pattern = '^(?P<path>.+?):(?P<line>\\d+):\\s*(?:(?:warning|error):\\s*)?(?P<message>.+)$'
message_template = 'python: {{message}}'
path_group = 'path'
line_group = 'line'

"""

_PRECOMPLIANT_README_TEMPLATE = """# {project_title}

This repository was bootstrapped by Agentic Engineer.

- `src/` holds application code. A package directory matching the project name has been created.
- `tests/` is ready for your test modules; no placeholder Python files are supplied.
- `policy/` stores the local policy capsule referenced by `config.yaml`.

Delete or rewrite these files as the project takes shape.
"""

_LEGACY_README_TEMPLATE = """# {project_title}

Initial repository scaffold created by Agentic Engineer.

Replace this README with project-specific documentation once goals are defined.
"""

_PRECOMPLIANT_CONFIG_TEMPLATE = """project:
  name: {project_slug}
  description: Agentic Engineer scaffold for {project_title}.
  repo_root: .
iteration:
  current: 0
  goal: Describe the primary objective before running `ae iterate`.
policy:
  capsule_path: policy/capsule.txt
  enable_checks: true
paths:
  data: data
  logs: data/logs
  cache: data/cache
  db_path: data/ae.sqlite
"""

_PRECOMPLIANT_CAPSULE_TEMPLATE = """Policy Capsule
====================

- Keep changes small and reversible.
- Never run pytest commands, these will run automatically. 
- Never commit, stash or push any files you work with. This will be automatically handled.
- Avoid commands requiring interactive input; rely on automated tests or pipe predefined input.
"""

_PATH_TOKEN_RE = re.compile(r"\b(?:src|tests)/[A-Za-z0-9_\-/]+")

_TESTS_CONFTST_TEMPLATE = """from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
"""

_REQUIREMENTS_TEMPLATE = "# Pin runtime dependencies here.\n"


def _has_existing_python_project(root: Path) -> bool:
    """Return ``True`` when the repository already contains Python sources/config."""

    sentinel_files = ("pyproject.toml", "setup.cfg", "setup.py")
    if any((root / name).exists() for name in sentinel_files):
        return True

    search_roots = [root / "src", root / "tests", root]
    patterns = ("*.py", "*.pyi")
    for base in search_roots:
        if not base.exists():
            continue
        for pattern in patterns:
            try:
                next(
                    path
                    for path in base.rglob(pattern)
                    if ".git" not in path.parts
                )
            except StopIteration:
                continue
            else:
                return True
    return False


def _slugify_project(name: str) -> str:
    """Produce a filesystem-friendly slug from a display name."""
    slug = name.strip().lower().replace(" ", "-")
    slug = "".join(char for char in slug if char.isalnum() or char == "-")
    return slug or "app"


def _package_name_from(name: str) -> str:
    """Derive a valid Python package identifier from arbitrary text."""
    candidate = name.strip().lower().replace("-", "_").replace(" ", "_")
    candidate = "".join(char for char in candidate if char.isalnum() or char == "_").strip("_")
    if not candidate:
        candidate = "app"
    if candidate[0].isdigit():
        candidate = f"app_{candidate}"
    return candidate


def _ensure_directory(path: Path) -> bool:
    """Create ``path`` as a directory if it does not yet exist."""

    if path.exists():
        return False
    path.mkdir(parents=True, exist_ok=True)
    return True


def _ensure_text_file(path: Path, *, content: str) -> bool:
    """Write ``content`` to ``path`` if the file does not already exist."""

    if path.exists():
        return False
    _ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")
    return True


def _ensure_package_init(path: Path, *, description: str) -> bool:
    """Ensure a package ``__init__.py`` exists."""
    _ensure_directory(path)
    init_path = path / "__init__.py"
    if init_path.exists():
        return False
    init_path.write_text(f'"""Initial package marker for {description} package."""\n', encoding="utf-8")
    return True


def _ensure_tests_init(path: Path, *, description: str) -> bool:
    """Ensure a tests package ``__init__.py`` exists."""
    _ensure_directory(path)
    init_path = path / "__init__.py"
    if init_path.exists():
        return False
    init_path.write_text(f'"""Tests for {description}."""\n', encoding="utf-8")
    return True



@dataclass(frozen=True)
class _ProgramSpec:
    """Normalised scaffold definition extracted from configuration data."""

    package_path: Path
    package_name: str
    title: str
    cli_stub: bool
    test_dirs: tuple[Path, ...]


def _load_config_mapping(
    root: Path,
    config: Mapping[str, Any] | Path | str | None,
) -> Mapping[str, Any]:
    """Return a mapping parsed from ``config`` with graceful fallbacks."""

    if config is None:
        config_path = root / "config.yaml"
        if not config_path.exists():
            return {}
        config = config_path

    if isinstance(config, Mapping):
        return config

    path = Path(config)
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.exists():
        return {}

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(loaded, Mapping):
        return loaded
    return {}


def _iter_strings(value: Any) -> Iterable[str]:
    """Yield all string values nested within ``value``."""
    if isinstance(value, str):
        yield value
        return

    if isinstance(value, Mapping):
        for candidate in value.values():
            yield from _iter_strings(candidate)
        return

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for candidate in value:
            yield from _iter_strings(candidate)


def _as_bool(value: Any) -> bool | None:
    """Interpret common truthy/falsey string flags, returning None when unsure."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _normalise_program_path(raw: str, *, fallback: str | None = None) -> str:
    """Convert user-supplied program paths into canonical ``src/``-relative form."""
    candidate = str(raw or "").strip().strip("/").replace("\\", "/")
    parts = [segment for segment in candidate.split("/") if segment]
    if not parts:
        return fallback or ""
    normalised = [_package_name_from(part) for part in parts]
    return "/".join(normalised)


def _normalise_test_path(raw: str) -> str:
    """Normalise configured test directories relative to the repository root."""
    return _normalise_program_path(raw, fallback="")


def _extract_program_specs(config: Mapping[str, Any]) -> list[_ProgramSpec]:
    """Collect scaffolding specifications from the agent configuration mapping."""
    if not config:
        return []

    builders: dict[str, dict[str, Any]] = {}
    tests_map: dict[str, set[str]] = {}

    def _ensure_builder(path: str) -> dict[str, Any]:
        normalised = _normalise_program_path(path, fallback="")
        if not normalised:
            return builders.setdefault("app", {"title": None, "cli_stub": False})
        return builders.setdefault(normalised, {"title": None, "cli_stub": False})

    def _record_tests(path: str) -> None:
        normalised = _normalise_test_path(path)
        if not normalised:
            return
        package_name = normalised.split("/", 1)[0]
        tests_map.setdefault(package_name, set()).add(normalised)

    def _ingest_program_entries(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, Mapping):
            package_value = value.get("package") or value.get("package_name") or value.get("module") or value.get("path")
            if not _should_consider_path(package_value):
                for nested in value.values():
                    _ingest_program_entries(nested)
                return

            builder = _ensure_builder(package_value)
            name_value = value.get("title") or value.get("display_name") or value.get("name")
            if isinstance(name_value, str) and name_value.strip():
                builder["title"] = name_value.strip()

            cli_value = value.get("cli_stub")
            if cli_value is None:
                cli_value = value.get("cli")
            if isinstance(cli_value, Mapping):
                cli_value = cli_value.get("stub") or cli_value.get("enabled")
            cli_flag = _as_bool(cli_value)
            builder["cli_stub"] = builder.get("cli_stub", False) or (True if cli_flag is None else cli_flag)

            tests_value = value.get("tests") or value.get("test_dirs") or value.get("test_paths")
            if isinstance(tests_value, Mapping):
                tests_value = tests_value.get("paths") or tests_value.get("dirs") or tests_value.get("names")
        if isinstance(tests_value, str) and _should_consider_path(tests_value):
            _record_tests(tests_value)
        elif isinstance(tests_value, Sequence) and not isinstance(tests_value, (str, bytes, bytearray)):
            for candidate in tests_value:
                if isinstance(candidate, str) and _should_consider_path(candidate):
                    _record_tests(candidate)

            for nested_key in ("programs", "packages", "apps"):
                nested = value.get(nested_key)
                if nested is not None:
                    _ingest_program_entries(nested)
            return

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                _ingest_program_entries(item)
            return

        if isinstance(value, str) and _should_consider_path(value):
            builder = _ensure_builder(value)
            builder["cli_stub"] = builder.get("cli_stub", False) or True

    for key in ("program", "programs", "applications", "apps", "packages", "plans"):
        _ingest_program_entries(config.get(key))

    scaffold_cfg = config.get("scaffold")
    if isinstance(scaffold_cfg, Mapping):
        for key in ("programs", "packages", "apps"):
            _ingest_program_entries(scaffold_cfg.get(key))

    iteration_cfg = config.get("iteration")
    if isinstance(iteration_cfg, Mapping):
        goal_text = iteration_cfg.get("goal")
        if isinstance(goal_text, str):
            for match in _PATH_TOKEN_RE.finditer(goal_text):
                token = match.group(0)
                if goal_text[match.end() : match.end() + 1] == ".":
                    continue
                if not _should_consider_path(token):
                    continue
                if token.startswith("src/"):
                    builder = _ensure_builder(token[4:])
                    builder["cli_stub"] = builder.get("cli_stub", False) or True
                elif token.startswith("tests/"):
                    _record_tests(token[6:])

    for text in _iter_strings(config):
        for match in _PATH_TOKEN_RE.finditer(text):
            token = match.group(0)
            if text[match.end() : match.end() + 1] == ".":
                continue
            if not _should_consider_path(token):
                continue
            if token.startswith("src/"):
                builder = _ensure_builder(token[4:])
                builder["cli_stub"] = builder.get("cli_stub", False) or True
            elif token.startswith("tests/"):
                _record_tests(token[6:])

    if not builders:
        return []

    specs: list[_ProgramSpec] = []
    for key, data in builders.items():
        path_parts = [segment for segment in key.split("/") if segment] or ["app"]
        package_path = Path("/".join(path_parts))
        package_name = path_parts[-1]
        title = data.get("title") or package_name.replace("_", " ").title()
        cli_stub = bool(data.get("cli_stub") or True)
        test_candidates = tests_map.get(package_name, set())
        if not test_candidates:
            test_candidates = {package_name}
        test_dirs = tuple(sorted(Path(candidate) for candidate in test_candidates))
        specs.append(
            _ProgramSpec(
                package_path=package_path,
                package_name=package_name,
                title=str(title),
                cli_stub=cli_stub,
                test_dirs=test_dirs,
            )
        )

    specs.sort(key=lambda spec: spec.package_path.as_posix())
    return specs


def _materialise_program_spec(
    *,
    spec: _ProgramSpec,
    src_root: Path,
    tests_root: Path,
    track: Callable[[Path, bool], None],
) -> None:
    """Create packages and tests on disk for a single program specification."""
    cursor = src_root
    package_segments = list(spec.package_path.parts)
    for index, segment in enumerate(package_segments):
        cursor = cursor / segment
        track(cursor, _ensure_directory(cursor))
        description = spec.package_name if index == len(package_segments) - 1 else segment
        track(cursor / "__init__.py", _ensure_package_init(cursor, description=description))

    for test_dir in spec.test_dirs:
        target = tests_root / test_dir
        track(target, _ensure_directory(target))
        track(target / "__init__.py", _ensure_tests_init(target, description=test_dir.as_posix()))


def ensure_program_framework(
    repo_root: Path | str,
    *,
    config: Mapping[str, Any] | Path | str | None = None,
    fallback_package: str | None = None,
    fallback_title: str | None = None,
) -> list[Path]:
    """Ensure shared folders exist for programs described in config."""

    root = Path(repo_root).resolve()
    config_mapping = _load_config_mapping(root, config)
    specs = _extract_program_specs(config_mapping)
    if not specs and fallback_package:
        normalised = _normalise_program_path(fallback_package, fallback=fallback_package)
        package_path = Path(normalised or "app")
        package_name = package_path.parts[-1]
        title = fallback_title or package_name.replace("_", " ").title()
        specs = [
            _ProgramSpec(
                package_path=package_path,
                package_name=package_name,
                title=title,
                cli_stub=True,
                test_dirs=(Path(package_name),),
            )
        ]
    if not specs:
        return []

    created: list[Path] = []

    def _track(path: Path, created_flag: bool) -> None:
        if not created_flag:
            return
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        if relative not in created:
            created.append(relative)

    src_root = root / "src"
    _track(src_root, _ensure_directory(src_root))
    tests_root = root / "tests"
    _track(tests_root, _ensure_directory(tests_root))
    _track(tests_root / "__init__.py", _ensure_tests_init(tests_root, description="tests"))

    for spec in specs:
        _materialise_program_spec(
            spec=spec,
            src_root=src_root,
            tests_root=tests_root,
            track=_track,
        )

    return created


def ensure_project_scaffold(
    repo_root: Path | str,
    *,
    project_name: str | None = None,
    precompliant: bool = True,
) -> list[Path]:
    """Create a minimal scaffold if the repository is empty."""

    root = Path(repo_root).resolve()
    config_mapping = _load_config_mapping(root, None)
    skip_root_scaffold = _should_skip_root_scaffold(config_mapping)
    has_python = _has_existing_python_project(root)
    created: list[Path] = []

    project_title = (project_name or root.name or "Project").strip() or "Project"
    fallback_package = _package_name_from(project_title)
    slug = _slugify_project(project_title)

    should_seed_root = not skip_root_scaffold and not has_python

    if should_seed_root:
        if precompliant:
            created = _create_precompliant_scaffold(
                root=root,
                title=project_title,
                slug=slug,
                package=fallback_package,
            )
        else:
            created = _create_legacy_scaffold(
                root=root,
                title=project_title,
                slug=slug,
                package=fallback_package,
            )
    elif skip_root_scaffold:
        pyproject_created = _ensure_text_file(
            root / "pyproject.toml",
            content=_MINIMAL_PYPROJECT_TEMPLATE.format(
                project_slug=slug,
                project_title=project_title,
                package_name=fallback_package,
            ),
        )
        if pyproject_created:
            created.append(root / "pyproject.toml")

    if not has_python:
        program_paths = ensure_program_framework(
            root,
            config=config_mapping,
            fallback_package=fallback_package if skip_root_scaffold else None,
            fallback_title=project_title if skip_root_scaffold else None,
        )
        for path in program_paths:
            candidate = root / path
            if candidate not in created:
                created.append(candidate)

    normalised: list[Path] = []
    for path in created:
        if hasattr(path, "is_absolute") and path.is_absolute():
            try:
                normalised.append(path.relative_to(root))
            except ValueError:
                normalised.append(path)
        else:
            normalised.append(path)
    return normalised


def _create_precompliant_scaffold(
    *,
    root: Path,
    title: str,
    slug: str,
    package: str,
) -> list[Path]:
    """Create the pre-compliant scaffold variant."""

    created: list[Path] = []

    files: dict[Path, str] = {
        root / "pyproject.toml": _MINIMAL_PYPROJECT_TEMPLATE.format(
            project_slug=slug,
            project_title=title,
            package_name=package,
        ),
        root / ".gitignore": (
            "/data\n"
            "__pycache__/\n"
            ".pytest_cache/\n"
            "*.py[cod]\n"
            ".coverage\n"
        ),
        root / "README.md": _PRECOMPLIANT_README_TEMPLATE.format(
            project_title=title,
        ),
        root / "requirements.txt": _REQUIREMENTS_TEMPLATE,
        root / "config.yaml": _PRECOMPLIANT_CONFIG_TEMPLATE.format(
            project_slug=slug,
            project_title=title,
        ),
        root / "policy" / "capsule.txt": _PRECOMPLIANT_CAPSULE_TEMPLATE,
    }

    directories: Iterable[Path] = [
        root / "src",
        root / "src" / package,
        root / "tests",
        root / "policy",
    ]

    for directory in directories:
        if _ensure_directory(directory):
            created.append(directory)

    if _ensure_package_init(root / "src" / package, description=package):
        created.append(Path("src") / package / "__init__.py")
    if _ensure_tests_init(root / "tests", description="tests"):
        created.append(Path("tests") / "__init__.py")
    if _ensure_tests_init(root / "tests" / package, description=package):
        created.append(Path("tests") / package / "__init__.py")

    for path, content in files.items():
        if _ensure_text_file(path, content=content):
            created.append(path)

    return created


def _create_legacy_scaffold(
    *,
    root: Path,
    title: str,
    slug: str,
    package: str,
) -> list[Path]:
    """Create the legacy scaffold variant (without policy/config extras)."""

    created: list[Path] = []

    files: dict[Path, str] = {
        root / "pyproject.toml": _MINIMAL_PYPROJECT_TEMPLATE.format(
            project_slug=slug,
            project_title=title,
            package_name=package,
        ),
        root / ".gitignore": "/data\n",
        root / "README.md": _LEGACY_README_TEMPLATE.format(
            project_title=title,
        ),
        root / "requirements.txt": _REQUIREMENTS_TEMPLATE,
    }

    directories: Iterable[Path] = [
        root / "src",
        root / "src" / package,
        root / "tests",
    ]

    for directory in directories:
        if _ensure_directory(directory):
            created.append(directory)

    if _ensure_package_init(root / "src" / package, description=package):
        created.append(Path("src") / package / "__init__.py")
    if _ensure_tests_init(root / "tests", description="tests"):
        created.append(Path("tests") / "__init__.py")
    if _ensure_tests_init(root / "tests" / package, description=package):
        created.append(Path("tests") / package / "__init__.py")

    for path, content in files.items():
        if _ensure_text_file(path, content=content):
            created.append(path)

    return created


def _should_consider_path(raw: str | None) -> bool:
    """Return True for candidate paths worth scaffolding from configuration hints."""
    if not isinstance(raw, str):
        return False
    candidate = raw.strip()
    if not candidate:
        return False
    last_segment = candidate.split("/")[-1]
    if "." in last_segment:
        return False
    return True


def _should_skip_root_scaffold(config: Mapping[str, Any]) -> bool:
    """Detect guidance that explicitly forbids modifying repository scaffolding."""
    context = config.get("context")
    if not isinstance(context, Mapping):
        return False
    guidance = context.get("guidance")
    if guidance is None:
        return False
    for entry in _iter_strings(guidance):
        lowered = entry.lower()
        if "do not create or modify repository root scaffolding files" in lowered:
            return True
    return False


__all__ = ["ensure_program_framework", "ensure_project_scaffold"]
