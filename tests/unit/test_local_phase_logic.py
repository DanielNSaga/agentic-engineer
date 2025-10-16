from __future__ import annotations

from ae.phases.local import LocalPhaseLogic


def test_extract_assertions_handles_equality() -> None:
    logs = "AssertionError: assert 'hi' == 'hello'"
    pairs = LocalPhaseLogic._extract_assertions(logs)
    assert ("hi", "hello") in pairs


def test_extract_assertions_handles_membership_with_empty_output() -> None:
    logs = "AssertionError: assert 'added' in ''"
    pairs = LocalPhaseLogic._extract_assertions(logs)
    assert ("", "added") in pairs


def test_extract_assertions_handles_membership_expression() -> None:
    logs = "AssertionError: assert 'added' in captured.out.lower()"
    pairs = LocalPhaseLogic._extract_assertions(logs)
    assert ("captured.out.lower()", "added") in pairs

