"""Static output parsing driven by pyproject.toml configuration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for earlier versions
    import tomli as tomllib  # type: ignore[assignment]

from .snippets import StaticFinding

__all__ = ["resolve_static_parser"]


@dataclass(frozen=True)
class _RegexPattern:
    """Configuration describing how to extract findings with regex patterns."""

    regex: re.Pattern[str]
    path_group: str
    line_group: str | None
    message_template: str
    default_line: int | None
    mode: str
    scope: str
    filters: Mapping[str, set[str]] | None

    def apply(
        self,
        *,
        text: str,
        lines: Sequence[str],
        limit: int,
        seen: set[tuple[str, int, str]],
    ) -> list[StaticFinding]:
        findings: list[StaticFinding] = []

        def _coerce_line(raw: str | None) -> int | None:
            if raw is None:
                return None
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None

        if self.scope == "text":
            candidates = list(self.regex.finditer(text))
        else:
            candidates = []
            func = self.regex.match if self.mode == "match" else self.regex.search
            for raw_line in lines:
                match = func(raw_line)
                if match is not None:
                    candidates.append((match, raw_line))
            # Normalise to tuple form for later logic.
            candidates = [(match, raw_line) for match, raw_line in candidates]  # type: ignore[assignment]

        for item in candidates:
            if len(findings) >= limit:
                break

            if self.scope == "text":
                match = item  # type: ignore[assignment]
                line_text = match.group(0)
            else:
                match, line_text = item  # type: ignore[assignment]

            data = match.groupdict()
            if self.filters:
                include = True
                for key, allowed in self.filters.items():
                    value = data.get(key, "")
                    if value.lower() not in allowed:
                        include = False
                        break
                if not include:
                    continue

            path = data.get(self.path_group, "")
            if not path:
                continue

            line_value = None
            if self.line_group:
                line_value = _coerce_line(data.get(self.line_group))
            if line_value is None:
                line_value = self.default_line
            if line_value is None:
                continue

            values: MutableMapping[str, Any] = {**data}
            values.setdefault("line", line_value)
            values.setdefault("line_text", line_text)
            path = _normalise_path(path)
            message = self.message_template.format_map(_SafeDict(values))
            key = (path, line_value, message)
            if key in seen:
                continue
            findings.append(
                StaticFinding(
                    path=path,
                    line_start=line_value,
                    line_end=line_value,
                    message=message,
                )
            )
            seen.add(key)
        return findings


@dataclass(frozen=True)
class _JsonPattern:
    """Configuration describing how to extract findings from JSON payloads."""

    records_key: str | None
    path_key: str
    line_key: str | None
    message_template: str
    filters: Mapping[str, set[str]] | None

    def apply(
        self,
        *,
        text: str,
        limit: int,
        seen: set[tuple[str, int, str]],
    ) -> list[StaticFinding]:
        findings: list[StaticFinding] = []
        if not text.strip():
            return findings
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return findings

        records: Iterable[Mapping[str, Any]] = []
        if isinstance(payload, list):
            records = (entry for entry in payload if isinstance(entry, Mapping))
        elif isinstance(payload, Mapping):
            if self.records_key:
                records_candidate = _dig_nested(payload, self.records_key)
                if isinstance(records_candidate, list):
                    records = (entry for entry in records_candidate if isinstance(entry, Mapping))
            else:
                records = [payload]

        for record in records:
            if len(findings) >= limit:
                break

            flat = _flatten(record)
            if self.filters:
                include = True
                for key, allowed in self.filters.items():
                    value = str(_fallback_get(record, flat, key)).lower()
                    if value not in allowed:
                        include = False
                        break
                if not include:
                    continue

            raw_path_value = _fallback_get(record, flat, self.path_key)
            path_value = _string(raw_path_value)
            if not path_value:
                continue
            line_value_raw = _fallback_get(record, flat, self.line_key) if self.line_key else None
            line_value = _coerce_int(line_value_raw)
            if line_value is None:
                continue

            message = self.message_template.format_map(_SafeDict(flat))
            path = _normalise_path(path_value)
            key = (path, line_value, message)
            if key in seen:
                continue
            findings.append(
                StaticFinding(
                    path=path,
                    line_start=line_value,
                    line_end=line_value,
                    message=message,
                )
            )
            seen.add(key)
        return findings


@dataclass(frozen=True)
class _ParserDefinition:
    """Collection of regex/JSON patterns bound to a named parser."""

    name: str
    identifiers: set[str]
    max_findings: int
    regex_patterns: tuple[_RegexPattern, ...]
    json_patterns: tuple[_JsonPattern, ...]

    def matches(self, hints: Iterable[str]) -> bool:
        tokens = {hint.lower() for hint in hints if hint}
        return bool(self.identifiers & tokens)

    def parse_output(self, output: str | Sequence[str] | Iterable[str]) -> list[StaticFinding]:
        text, lines = _coerce_output(output)
        max_results = max(self.max_findings, 1)
        findings: list[StaticFinding] = []
        seen: set[tuple[str, int, str]] = set()

        for pattern in self.json_patterns:
            if len(findings) >= max_results:
                break
            findings.extend(
                pattern.apply(text=text, limit=max_results - len(findings), seen=seen)
            )
        if findings:
            return findings[:max_results]

        for pattern in self.regex_patterns:
            if len(findings) >= max_results:
                break
            findings.extend(
                pattern.apply(
                    text=text,
                    lines=lines,
                    limit=max_results - len(findings),
                    seen=seen,
                )
            )
            if findings:
                break

        return findings[:max_results]


@dataclass(frozen=True)
class _ParserRegistry:
    """Lookup table that matches hints to parser definitions."""

    parsers: tuple[_ParserDefinition, ...]

    def resolve(self, hints: Iterable[str]) -> _ParserDefinition | None:
        for parser in self.parsers:
            if parser.matches(hints):
                return parser
        return None


def resolve_static_parser(
    repo_root: Path | None,
    hints: Sequence[str] | None,
) -> Callable[[str | Sequence[str] | Iterable[str]], list[StaticFinding]] | None:
    """Return a parser callable configured by ``pyproject.toml`` for the given hints."""

    if not hints:
        return None

    base = (repo_root or Path.cwd()).resolve()
    pyproject = base / "pyproject.toml"
    registry = _load_registry(pyproject)
    if registry is None:
        return None
    parser = registry.resolve(hints)
    if parser is None:
        return None
    return parser.parse_output


@lru_cache(maxsize=16)
def _load_registry(pyproject_path: Path) -> _ParserRegistry | None:
    """Load and cache parser definitions from ``pyproject.toml``."""
    if not pyproject_path.exists():
        return None
    try:
        with pyproject_path.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return None

    section = (
        document.get("tool", {})
        .get("agentic_engineer", {})
        .get("static_parsers")
    )
    if not isinstance(section, Mapping):
        return None

    parsers: list[_ParserDefinition] = []
    for name, config in section.items():
        if not isinstance(config, Mapping) or name in {"version", "defaults"}:
            continue
        parser = _build_parser_definition(name, config)
        if parser is not None:
            parsers.append(parser)

    return _ParserRegistry(parsers=tuple(parsers))


def _build_parser_definition(name: str, config: Mapping[str, Any]) -> _ParserDefinition | None:
    """Construct a parser definition from a raw configuration mapping."""
    identifiers = _normalise_identifiers(config.get("identifiers"), default=name)
    if not identifiers:
        return None

    max_findings = _coerce_int(config.get("max_findings"), default=25)
    regex_patterns: list[_RegexPattern] = []
    json_patterns: list[_JsonPattern] = []

    patterns = config.get("patterns")
    if isinstance(patterns, list):
        for entry in patterns:
            if not isinstance(entry, Mapping):
                continue
            kind = str(entry.get("type") or "regex").lower()
            if kind == "regex":
                spec = _build_regex_pattern(entry)
                if spec is not None:
                    regex_patterns.append(spec)
            elif kind == "json":
                spec = _build_json_pattern(entry)
                if spec is not None:
                    json_patterns.append(spec)

    if not regex_patterns and not json_patterns:
        return None

    return _ParserDefinition(
        name=name,
        identifiers=identifiers,
        max_findings=max_findings,
        regex_patterns=tuple(regex_patterns),
        json_patterns=tuple(json_patterns),
    )


def _build_regex_pattern(config: Mapping[str, Any]) -> _RegexPattern | None:
    """Convert a regex pattern configuration into an executable spec."""
    pattern_text = config.get("pattern")
    if not isinstance(pattern_text, str) or not pattern_text:
        return None

    flags_value = config.get("flags")
    flags = 0
    if isinstance(flags_value, Sequence) and not isinstance(flags_value, (str, bytes)):
        for flag in flags_value:
            if not isinstance(flag, str):
                continue
            flag_upper = flag.upper()
            flags |= getattr(re, flag_upper, 0)

    try:
        regex = re.compile(pattern_text, flags=flags)
    except re.error:
        return None

    path_group = config.get("path_group")
    if not isinstance(path_group, str) or not path_group:
        return None

    line_group = config.get("line_group")
    if isinstance(line_group, str) and not line_group:
        line_group = None
    elif not isinstance(line_group, str):
        line_group = None

    default_line = _coerce_int(config.get("default_line"), default=None)

    message_template = config.get("message_template")
    if not isinstance(message_template, str) or not message_template:
        message_template = "{message}"

    mode = str(config.get("mode") or "match").lower()
    if mode not in {"match", "search"}:
        mode = "match"

    scope = str(config.get("scope") or "line").lower()
    if scope not in {"line", "text"}:
        scope = "line"

    filters = _build_filters(config.get("filters"))

    return _RegexPattern(
        regex=regex,
        path_group=path_group,
        line_group=line_group,
        message_template=message_template,
        default_line=default_line,
        mode=mode,
        scope=scope,
        filters=filters,
    )


def _build_json_pattern(config: Mapping[str, Any]) -> _JsonPattern | None:
    """Convert a JSON pattern configuration into an executable spec."""
    path_key = config.get("path_key")
    if not isinstance(path_key, str) or not path_key:
        return None

    line_key = config.get("line_key")
    if isinstance(line_key, str) and not line_key:
        line_key = None
    elif not isinstance(line_key, str):
        line_key = None

    message_template = config.get("message_template")
    if not isinstance(message_template, str) or not message_template:
        message_template = "{message}"

    records_key = config.get("records_key")
    if isinstance(records_key, str) and not records_key:
        records_key = None
    elif not isinstance(records_key, str):
        records_key = None

    filters = _build_filters(config.get("filters"))

    return _JsonPattern(
        records_key=records_key,
        path_key=path_key,
        line_key=line_key,
        message_template=message_template,
        filters=filters,
    )


def _build_filters(value: Any) -> Mapping[str, set[str]] | None:
    """Normalise filter configuration into lowercase string sets."""
    if not isinstance(value, Mapping):
        return None
    filters: dict[str, set[str]] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            continue
        if isinstance(raw, str):
            filters[key] = {raw.lower()}
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            candidates = {str(item).lower() for item in raw if isinstance(item, (str, int, float))}
            if candidates:
                filters[key] = candidates
    return filters or None


def _normalise_identifiers(value: Any, *, default: str) -> set[str]:
    """Normalise identifier hints used to match parser definitions."""
    identifiers: set[str] = set()
    if isinstance(value, str):
        identifiers.add(value.lower())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        identifiers = {str(item).lower() for item in value if isinstance(item, (str, int, float))}
    if not identifiers and default:
        identifiers.add(default.lower())
    return identifiers


def _normalise_path(path: str) -> str:
    """Return a consistent, forward-slash path for findings."""
    normalized = path.strip()
    if "\\" in normalized:
        normalized = normalized.replace("\\", "/")
    return normalized


def _coerce_int(value: Any, *, default: int | None = None) -> int | None:
    """Parse integers from strings while respecting a default fallback."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _string(value: Any) -> str:
    """Best-effort conversion of values to trimmed strings."""
    return str(value).strip() if isinstance(value, (str, int, float)) else ""


def _coerce_output(output: str | Sequence[str] | Iterable[str]) -> tuple[str, list[str]]:
    """Normalise process output into a joined string and per-line list."""
    if isinstance(output, str):
        text = output
        lines = output.splitlines()
    elif isinstance(output, (list, tuple)):
        lines = [str(line) for line in output]
        text = "\n".join(lines)
    else:
        lines = [str(line) for line in output]
        text = "\n".join(lines)
    return text, lines


def _flatten(data: Mapping[str, Any], *, prefix: str | None = None) -> dict[str, Any]:
    """Flatten nested mappings into dotted-key dictionaries."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        combined = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=combined))
        else:
            flat[combined] = value
            if key not in flat:
                flat[key] = value
    return flat


def _dig_nested(data: Mapping[str, Any], path: str | None) -> Any:
    """Traverse ``data`` following a dotted path, returning ``None`` when absent."""
    if path is None:
        return None
    current: Any = data
    for segment in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(segment)
        else:
            return None
    return current


def _fallback_get(
    nested: Mapping[str, Any],
    flat: Mapping[str, Any],
    key: str | None,
) -> Any:
    """Lookup ``key`` in nested data, falling back to flattened keys."""
    if key is None:
        return None
    value = _dig_nested(nested, key)
    if value is not None:
        return value
    return flat.get(key)


class _SafeDict(dict):
    """`str.format_map` helper that tolerates missing keys."""

    def __missing__(self, key: str) -> str:
        return ""
