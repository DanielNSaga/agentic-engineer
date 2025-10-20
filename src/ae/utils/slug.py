"""Utilities for generating consistent, length-limited slugs."""

from __future__ import annotations

import hashlib
import re
from typing import Pattern

_LOWERCASE_PATTERN: Pattern[str] = re.compile(r"[^a-z0-9_.-]+")
_MIXED_CASE_PATTERN: Pattern[str] = re.compile(r"[^A-Za-z0-9_.-]+")
_HYPHEN_COLLAPSE = re.compile(r"-{2,}")


def slugify(
    value: str | None,
    *,
    fallback: str = "item",
    max_length: int = 80,
    lowercase: bool = True,
) -> str:
    """Normalize ``value`` into a filesystem-friendly slug."""
    source = (value or "").strip()
    if not source:
        source = fallback

    processed_fallback = (fallback or "").strip() or "item"
    if lowercase:
        source = source.lower()
        processed_fallback = processed_fallback.lower()

    pattern = _LOWERCASE_PATTERN if lowercase else _MIXED_CASE_PATTERN

    slug = _normalize(source, pattern)
    fallback_slug = _normalize(processed_fallback, pattern) or ("item" if lowercase else "Item")

    if not slug:
        slug = fallback_slug

    if len(slug) > max_length:
        slug = abbreviate_slug(slug, fallback=fallback_slug, max_length=max_length)

    return slug


def abbreviate_slug(segment: str, *, fallback: str = "item", max_length: int = 80) -> str:
    """Trim ``segment`` to ``max_length`` while preserving uniqueness via hashing."""
    slug = segment.strip("-")
    fallback_slug = fallback.strip("-") or "item"

    if not slug:
        slug = fallback_slug

    if len(slug) <= max_length:
        return slug

    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:8]
    prefix_length = max(max_length - len(digest) - 1, 1)
    prefix = slug[:prefix_length].rstrip("-")
    if not prefix:
        prefix = slug[:prefix_length]
    return f"{prefix}-{digest}"


def _normalize(value: str, pattern: Pattern[str]) -> str:
    slug = pattern.sub("-", value)
    slug = _HYPHEN_COLLAPSE.sub("-", slug)
    return slug.strip("-")
