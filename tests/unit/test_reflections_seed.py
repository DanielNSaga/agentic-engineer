from __future__ import annotations

from ae.memory.reflections import PLANNER_PREFLIGHT_SCOPE, get_planner_preflight_seed
from ae.memory.store import MemoryStore


def test_reflection_store_seeds_planner_rules(tmp_path) -> None:
    db_path = tmp_path / "ae.sqlite"
    with MemoryStore(db_path) as store:
        reflections = store.reflections.get_top_reflections(PLANNER_PREFLIGHT_SCOPE, limit=10)

    expected_ids = {entry.id for entry in get_planner_preflight_seed()}
    actual_ids = {entry.id for entry in reflections}
    assert expected_ids == actual_ids

    for reflection in reflections:
        context = reflection.context
        assert isinstance(context, dict)
        assert "task" in context
        assert "applies_when" in context
