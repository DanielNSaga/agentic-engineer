"""
Planning utilities for synthesizing iteration-zero artifacts.
"""

from importlib import import_module
from typing import Any

__all__ = ["PlanningArtifacts", "bootstrap_initial_plan"]


def __getattr__(name: str) -> Any:
    """Lazily import planning helpers to avoid eager bootstrap dependencies."""
    if name in __all__:
        module = import_module("ae.planning.bootstrap")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
