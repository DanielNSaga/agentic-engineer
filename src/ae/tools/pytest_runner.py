"""Pytest execution helpers for the automated coding loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Mapping, Sequence

import hashlib
import json
import os
import re
import subprocess
import sys

PytestStatus = Literal["passed", "failed", "error", "no-tests"]


@dataclass(slots=True)
class PytestResult:
    """Structured summary of a pytest invocation."""

    command: tuple[str, ...]
    cwd: Path
    exit_code: int
    status: PytestStatus
    collected: int | None
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.status == "passed" or (self.status == "no-tests")


def _merge_env(extra: Mapping[str, str] | None) -> Dict[str, str]:
    """Merge provided environment overrides with the current process state."""
    env: Dict[str, str] = os.environ.copy()
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


_COLLECT_RE = re.compile(r"collected\s+(\d+)\s+item")
_SUMMARY_RE = re.compile(
    r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected)"
)
_PROGRESS_RE = re.compile(r"^\s*([.FEsSkxX!P]+)\s*\[[^\]]+\]\s*$")
_PROGRESS_SYMBOLS = frozenset(".FEsSkxX!P")
_OPTS_EXPECT_VALUE = {
    "-c",
    "-k",
    "-m",
    "-n",
    "-o",
    "-p",
    "--basetemp",
    "--confcutdir",
    "--cov",
    "--cov-config",
    "--cov-report",
    "--durations",
    "--html",
    "--junitxml",
    "--log-cli-level",
    "--log-level",
    "--looponfail-command",
    "--maxfail",
    "--result-log",
    "--rootdir",
    "--self-contained-html",
}


def _normalise_pytest_path(token: str, workdir: Path) -> str:
    """Normalise filesystem tokens into repository-relative pytest paths."""
    cleaned = token.replace("\\", "/")
    trailing_slash = cleaned.endswith("/")
    cleaned = cleaned.rstrip("/")
    if cleaned.startswith("../"):
        return token
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    candidate = (workdir / cleaned).resolve()
    if candidate.exists():
        result = cleaned
    else:
        parts = [part for part in cleaned.split("/") if part]
        result = cleaned
        for index in range(1, len(parts)):
            trimmed = "/".join(parts[index:])
            if not trimmed:
                continue
            probe = (workdir / trimmed).resolve()
            if probe.exists():
                result = trimmed
                break
    if trailing_slash and result:
        result = f"{result}/"
    return result or token


def _normalise_pytest_token(token: str, workdir: Path) -> str:
    """Resolve pytest node tokens containing optional test selectors."""
    if "::" in token:
        path_part, remainder = token.split("::", 1)
        normalised = _normalise_pytest_path(path_part, workdir)
        if normalised != path_part:
            return f"{normalised}::{remainder}"
        return token
    normalised = _normalise_pytest_path(token, workdir)
    return normalised if normalised != token else token


def _normalise_pytest_args(args: Sequence[str], workdir: Path) -> tuple[str, ...]:
    """Apply consistent path normalisation across pytest command arguments."""
    result: list[str] = []
    pending_option: str | None = None
    for token in args:
        if pending_option:
            if pending_option in _OPTS_EXPECT_VALUE:
                result.append(_normalise_pytest_token(token, workdir) if not token.startswith("-") else token)
            else:
                result.append(token)
            pending_option = None
            continue

        if token in _OPTS_EXPECT_VALUE:
            result.append(token)
            pending_option = token
            continue

        if token.startswith("--"):
            name, sep, value = token.partition("=")
            if sep and value:
                if name in _OPTS_EXPECT_VALUE:
                    normalised = _normalise_pytest_token(value, workdir)
                    result.append(f"{name}={normalised}")
                else:
                    result.append(token)
                continue
            if name in _OPTS_EXPECT_VALUE:
                result.append(token)
                pending_option = name
                continue
            result.append(token)
            continue

        if token.startswith("-"):
            result.append(token)
            continue

        if (workdir / token).resolve().exists():
            result.append(token)
            continue

        normalised = _normalise_pytest_token(token, workdir)
        result.append(normalised)
    return tuple(result)


def _parse_collected(stdout: str, stderr: str) -> int | None:
    """Extract the number of collected tests from pytest output streams."""
    text = "\n".join(part for part in (stdout, stderr) if part)
    match = _COLLECT_RE.search(text)
    if match:
        return int(match.group(1))

    matches = _SUMMARY_RE.findall(text)
    if not matches:
        for line in text.splitlines():
            progress = _PROGRESS_RE.match(line)
            if progress:
                symbols = progress.group(1)
                total = sum(1 for char in symbols if char in _PROGRESS_SYMBOLS)
                if total:
                    return total
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and all(char in _PROGRESS_SYMBOLS for char in stripped):
                return len(stripped)
        return None
    return sum(int(amount) for amount, _ in matches)


def _status_from_exit_code(exit_code: int) -> PytestStatus:
    """Translate pytest's exit codes into the consolidated status enum."""
    if exit_code == 0:
        return "passed"
    if exit_code == 5:
        return "no-tests"
    if exit_code == 1:
        return "failed"
    return "error"


_DEPENDENCY_SENTINEL = Path(".agentic-engineer") / "deps-installed.json"


def _discover_requirements(root: Path) -> list[Path]:
    """Locate requirement files that should be installed prior to pytest runs."""
    candidates: list[Path] = []
    primary_names = (
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-dev.in",
        "requirements-test.txt",
    )
    for name in primary_names:
        path = root / name
        if path.is_file():
            candidates.append(path)

    src_root = root / "src"
    if src_root.is_dir():
        for package_dir in src_root.iterdir():
            if not package_dir.is_dir():
                continue
            requirement_path = package_dir / "requirements.txt"
            if requirement_path.is_file():
                candidates.append(requirement_path)

    # Drop empty files and duplicates while preserving order.
    seen: set[Path] = set()
    filtered: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            if path.read_text(encoding="utf-8").strip():
                filtered.append(path)
        except OSError:
            continue
    return filtered


def _requirements_fingerprint(root: Path, requirements: Sequence[Path]) -> str:
    """Build a reproducible fingerprint for the set of requirement files."""
    digest = hashlib.sha256()
    for path in sorted(requirements):
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _ensure_dependencies_installed(
    root: Path,
    env: Mapping[str, str],
    *,
    force: bool = False,
) -> tuple[bool, str | None]:
    """Install pip dependencies when the recorded fingerprint changes."""
    requirements = _discover_requirements(root)
    if not requirements:
        return True, None

    sentinel_path = root / _DEPENDENCY_SENTINEL
    fingerprint = _requirements_fingerprint(root, requirements)

    if not force and sentinel_path.is_file():
        try:
            data = json.loads(sentinel_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        if data.get("fingerprint") == fingerprint:
            return True, None

    for requirement in requirements:
        install_cmd = (
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirement),
        )
        result = subprocess.run(
            install_cmd,
            cwd=root,
            env=dict(env),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            try:
                display_path = requirement.relative_to(root)
            except ValueError:
                display_path = requirement
            detail = (result.stderr or "").strip() or (result.stdout or "").strip()
            message = f"pip install -r {display_path} failed with exit code {result.returncode}"
            if detail:
                message = f"{message}: {detail}"
            return False, message

    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "files": [
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            for path in requirements
        ],
    }
    try:
        sentinel_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass
    return True, None


def ensure_requirements_installed(
    root: Path,
    *,
    env: Mapping[str, str] | None = None,
    force: bool = False,
) -> tuple[bool, str | None]:
    """Install runtime requirements for ``root`` when they are out of date."""

    env_vars = _merge_env(env)
    return _ensure_dependencies_installed(root, env_vars, force=force)


def run_pytest(
    args: Sequence[str] | None = None,
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> PytestResult:
    """Execute pytest and return a structured result."""

    workdir = Path(cwd or Path.cwd()).resolve()
    raw_args = tuple(args or ("-q",))
    invocation = ("pytest", *_normalise_pytest_args(raw_args, workdir))

    env_vars = _merge_env(env)
    install_ok, install_error = _ensure_dependencies_installed(workdir, env_vars)
    if not install_ok:
        message = install_error or "Failed to install repository requirements."
        return PytestResult(
            command=invocation,
            cwd=workdir,
            exit_code=2,
            status="error",
            collected=None,
            stdout="",
            stderr=message,
        )
    src_dir = workdir / "src"
    if src_dir.is_dir():
        src_entry = str(src_dir)
        current_pythonpath = env_vars.get("PYTHONPATH")
        if current_pythonpath:
            parts = current_pythonpath.split(os.pathsep)
            if src_entry not in parts:
                env_vars["PYTHONPATH"] = os.pathsep.join([src_entry, current_pythonpath])
        else:
            env_vars["PYTHONPATH"] = src_entry

    process = subprocess.run(
        invocation,
        cwd=workdir,
        env=env_vars,
        capture_output=True,
        text=True,
        check=False,
    )

    collected = _parse_collected(process.stdout, process.stderr)
    status = _status_from_exit_code(process.returncode)

    return PytestResult(
        command=invocation,
        cwd=workdir,
        exit_code=process.returncode,
        status=status,
        collected=collected,
        stdout=process.stdout,
        stderr=process.stderr,
    )


__all__ = ["PytestResult", "PytestStatus", "ensure_requirements_installed", "run_pytest"]
