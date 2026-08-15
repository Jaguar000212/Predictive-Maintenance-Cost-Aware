"""Run-record tests.

The record exists so a number in the report can be traced back to the code and
settings that produced it. These tests target the ways that trace can silently
break.
"""

from __future__ import annotations

import json

import pytest

from pdm.config import default_config
from pdm.eval.results import GitState, ResultsWriter, RunRecord, environment


def _record(**overrides) -> RunRecord:
    defaults = {
        "name": "unit_test",
        "config": default_config().to_dict(),
        "metrics": {"summary": {"pr_auc": {"mean": 0.5}}},
        "seeds": [0, 1],
        "git": GitState(sha="a" * 40, branch="main", dirty=False),
    }
    return RunRecord(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# Code identity
# ---------------------------------------------------------------------------
def test_clean_tree_identifies_the_code():
    assert GitState(sha="a" * 40, branch="main", dirty=False).identifies_the_code


def test_dirty_tree_does_not_identify_the_code():
    """A SHA names a commit that does not contain the uncommitted changes."""
    assert not GitState(sha="a" * 40, branch="main", dirty=True).identifies_the_code


def test_missing_git_does_not_identify_the_code():
    assert not GitState(sha=None, branch=None, dirty=False).identifies_the_code


def test_capture_reads_the_real_repository():
    state = GitState.capture(default_config().paths.repo_root)
    assert state.sha is not None and len(state.sha) == 40
    assert state.branch is not None


def test_capture_outside_a_repo_returns_nulls_rather_than_raising(tmp_path):
    state = GitState.capture(tmp_path)
    assert state.sha is None or isinstance(state.sha, str)


# ---------------------------------------------------------------------------
# Record contents
# ---------------------------------------------------------------------------
def test_record_carries_everything_needed_to_reproduce():
    payload = _record().to_dict()
    for key in ("name", "created_utc", "git", "seeds", "environment", "config", "metrics"):
        assert key in payload, f"missing {key}"
    assert payload["reproducible"] is True
    assert payload["config"]["cv"]["random_state"] == 42


def test_record_flags_itself_unreproducible_when_the_tree_was_dirty():
    payload = _record(git=GitState(sha="b" * 40, branch="main", dirty=True)).to_dict()
    assert payload["reproducible"] is False


def test_record_serialises_to_valid_json():
    payload = json.loads(_record().to_json())
    assert payload["name"] == "unit_test"
    assert isinstance(payload["environment"], dict)


def test_environment_captures_versions_that_move_results():
    versions = environment()
    for module in ("python", "numpy", "pandas", "sklearn"):
        assert module in versions


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def test_writer_creates_the_directory_and_names_the_file_by_run_and_time(tmp_path):
    out = tmp_path / "results"
    path = ResultsWriter(out).write(_record())

    assert path.exists()
    assert path.parent == out
    assert path.name.startswith("unit_test__")
    assert path.suffix == ".json"


def test_repeated_runs_accumulate_rather_than_overwrite(tmp_path):
    """Results are the audit trail; a rerun must not erase its predecessor."""
    writer = ResultsWriter(tmp_path)
    first = writer.write(_record(created_utc="2026-08-15T10:00:00+00:00"))
    second = writer.write(_record(created_utc="2026-08-15T11:00:00+00:00"))

    assert first != second
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_writing_from_a_dirty_tree_warns(tmp_path):
    """Silent recording of unreproducible runs is how untraceable figures happen."""
    dirty = _record(git=GitState(sha="c" * 40, branch="main", dirty=True))
    with pytest.warns(UserWarning, match="does not identify the code"):
        ResultsWriter(tmp_path).write(dirty)


def test_clean_tree_writes_without_warning(tmp_path, recwarn):
    ResultsWriter(tmp_path).write(_record())
    assert not [w for w in recwarn if "identify the code" in str(w.message)]


def test_written_json_round_trips(tmp_path):
    path = ResultsWriter(tmp_path).write(_record())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seeds"] == [0, 1]
    assert payload["metrics"]["summary"]["pr_auc"]["mean"] == 0.5
