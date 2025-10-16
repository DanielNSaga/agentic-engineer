from __future__ import annotations

from pathlib import Path

from ae.tools.coverage_map import CoverageMap


def test_coverage_map_records_and_queries() -> None:
    coverage = CoverageMap()

    coverage.record("tests/test_alpha.py::test_alpha", ["src/foo.py", "src/bar.py"])
    coverage.record("tests/test_beta.py::test_beta", ["src/foo.py"])

    assert coverage.touched_by("src/foo.py") == {
        "tests/test_alpha.py::test_alpha",
        "tests/test_beta.py::test_beta",
    }

    affected = coverage.affected_tests(["src/bar.py"])
    assert affected == {"tests/test_alpha.py::test_alpha"}


def test_coverage_map_persistence_roundtrip(tmp_path: Path) -> None:
    coverage = CoverageMap()
    coverage.record("tests/test_gamma.py::test_gamma", ["src/baz.py"])

    path = tmp_path / "coverage.json"
    coverage.dump(path)

    loaded = CoverageMap.load(path)
    assert loaded.touched_by("src/baz.py") == {"tests/test_gamma.py::test_gamma"}
