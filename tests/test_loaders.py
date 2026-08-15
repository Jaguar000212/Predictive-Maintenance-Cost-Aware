"""Loader tests against synthetic fixtures.

None of these touch the real dataset -- that is the point. Injecting `path` /
`data_dir` and a schema is what makes the loaders testable, and each test here
targets one of the four silent-corruption paths the loaders exist to block.
"""

from __future__ import annotations

import pytest

from pdm.config import AI4ISchema, CMAPSSSchema
from pdm.loaders import (
    AI4ILoader,
    CMAPSSLifetimeBuilder,
    CMAPSSLoader,
    CMAPSSRULLoader,
    DatasetLoader,
    DataValidationError,
)

AI4I_HEADER = (
    "UDI,Product ID,Type,Air temperature [K],Process temperature [K],"
    "Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure,TWF,HDF,PWF,OSF,RNF"
)
AI4I_ROWS = [
    "1,M14860,M,298.1,308.6,1551,42.8,0,0,0,0,0,0,0",
    "2,L47181,L,298.2,308.7,1408,46.3,3,1,0,1,0,0,0",
    "3,H29424,H,298.1,308.5,1498,49.4,5,0,0,0,0,0,0",
]


@pytest.fixture
def ai4i_csv(tmp_path):
    """A tiny AI4I file written WITH a UTF-8 BOM, exactly like the published one."""
    path = tmp_path / "ai4i_fixture.csv"
    path.write_text("\n".join([AI4I_HEADER, *AI4I_ROWS]) + "\n", encoding="utf-8-sig")
    return path


def _cmapss_rows(units: dict[int, int]) -> str:
    """Build trajectory text: {unit: n_cycles}. Trailing space on every line,
    exactly like the NASA distribution."""
    lines = []
    for unit, n_cycles in units.items():
        for cycle in range(1, n_cycles + 1):
            values = [str(unit), str(cycle)] + [f"{0.1 * i:.4f}" for i in range(24)]
            lines.append(" ".join(values) + "  ")
    return "\n".join(lines) + "\n"


@pytest.fixture
def cmapss_dir(tmp_path):
    """train: units 1-2 run to failure. test: units 1-2 truncated. Plus a RUL file."""
    (tmp_path / "train_FIX.txt").write_text(_cmapss_rows({1: 5, 2: 8}), encoding="utf-8")
    (tmp_path / "test_FIX.txt").write_text(_cmapss_rows({1: 3, 2: 4}), encoding="utf-8")
    (tmp_path / "RUL_FIX.txt").write_text("10\n20\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def fixture_schema():
    return CMAPSSSchema(subset="FIX", expected_train_units=2, expected_test_units=2)


# ---------------------------------------------------------------------------
# Shared contract
# ---------------------------------------------------------------------------
def test_loaders_implement_the_base_contract():
    for cls in (AI4ILoader, CMAPSSLoader, CMAPSSRULLoader):
        assert issubclass(cls, DatasetLoader)


def test_missing_file_names_the_path_and_explains_why():
    loader = AI4ILoader(path="does/not/exist.csv")
    with pytest.raises(DataValidationError, match="gitignored"):
        loader.load()


def test_validation_errors_are_prefixed_with_the_filename(ai4i_csv):
    """The base class adds file context so no subclass has to remember to."""
    schema = AI4ISchema(expected_rows=999)
    with pytest.raises(DataValidationError, match=r"^ai4i_fixture\.csv: expected 999 rows"):
        AI4ILoader(schema, path=ai4i_csv).load()


# ---------------------------------------------------------------------------
# AI4I
# ---------------------------------------------------------------------------
def test_bom_is_stripped_and_columns_renamed(ai4i_csv):
    df = AI4ILoader(AI4ISchema(expected_rows=None), path=ai4i_csv).load()
    assert list(df.columns) == list(AI4ISchema().columns)
    assert "udi" in df.columns
    assert not any(c.startswith("﻿") for c in df.columns)
    assert len(df) == 3


def test_pandas_tolerates_the_bom_under_either_encoding(ai4i_csv):
    """Documents measured behaviour, so nobody re-derives it from folklore.

    pandas 2.3 strips the UTF-8 BOM under both engines whatever encoding is
    declared, so utf-8-sig is defensive here rather than required. It still
    matters outside pandas: `open(path, encoding="utf-8")` on this file yields
    a first column named "\\ufeffUDI". If a future pandas stops stripping it,
    this test flips and the missing-column check catches the load.
    """
    schema = AI4ISchema(encoding="utf-8", expected_rows=None)
    df = AI4ILoader(schema, path=ai4i_csv).load()
    assert "udi" in df.columns

    with open(ai4i_csv, encoding="utf-8") as fh:
        assert fh.readline().startswith("﻿")
    with open(ai4i_csv, encoding="utf-8-sig") as fh:
        assert fh.readline().startswith("UDI")


def test_missing_columns_are_reported_with_what_was_found(tmp_path):
    path = tmp_path / "wrong_header.csv"
    path.write_text("id,thing\n1,x\n", encoding="utf-8-sig")
    with pytest.raises(DataValidationError, match="missing expected columns"):
        AI4ILoader(AI4ISchema(expected_rows=None), path=path).load()


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "dupe.csv"
    path.write_text("\n".join([AI4I_HEADER, AI4I_ROWS[0], AI4I_ROWS[0]]) + "\n", encoding="utf-8-sig")
    with pytest.raises(DataValidationError, match="'udi' is not unique"):
        AI4ILoader(AI4ISchema(expected_rows=None), path=path).load()


def test_non_binary_target_is_rejected(tmp_path):
    # Same as AI4I_ROWS[0] but machine_failure (column 9) is 2.
    bad = "1,M14860,M,298.1,308.6,1551,42.8,0,2,0,0,0,0,0"
    assert bad.split(",")[8] == "2"
    path = tmp_path / "bad_target.csv"
    path.write_text(f"{AI4I_HEADER}\n{bad}\n", encoding="utf-8-sig")
    with pytest.raises(DataValidationError, match="'machine_failure' is not binary"):
        AI4ILoader(AI4ISchema(expected_rows=None), path=path).load()


def test_validation_can_be_switched_off(ai4i_csv):
    df = AI4ILoader(AI4ISchema(expected_rows=999), path=ai4i_csv, validate=False).load()
    assert len(df) == 3


# ---------------------------------------------------------------------------
# C-MAPSS
# ---------------------------------------------------------------------------
def test_trailing_whitespace_does_not_create_phantom_columns(cmapss_dir, fixture_schema):
    """sep=" " would parse 28 columns here, two of them all-NaN."""
    df = CMAPSSLoader("train", fixture_schema, data_dir=cmapss_dir).load()
    assert df.shape[1] == 26
    assert not df.isna().any().any()


def test_sensor_numbering_starts_at_column_index_five(cmapss_dir, fixture_schema):
    df = CMAPSSLoader("train", fixture_schema, data_dir=cmapss_dir).load()
    assert list(df.columns)[:5] == ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    assert list(df.columns)[5] == "sensor_1"
    assert list(df.columns)[-1] == "sensor_21"


def test_wrong_expected_unit_count_is_caught(cmapss_dir):
    schema = CMAPSSSchema(subset="FIX", expected_train_units=99, expected_test_units=2)
    with pytest.raises(DataValidationError, match="expected 99 units"):
        CMAPSSLoader("train", schema, data_dir=cmapss_dir).load()


def test_non_contiguous_unit_ids_are_caught(tmp_path, fixture_schema):
    """The RUL join is positional, which is only safe for a contiguous 1..N block."""
    (tmp_path / "train_FIX.txt").write_text(_cmapss_rows({1: 3, 7: 3}), encoding="utf-8")
    with pytest.raises(DataValidationError, match="not a contiguous 1..N block"):
        CMAPSSLoader("train", fixture_schema, data_dir=tmp_path).load()


def test_bad_split_name_is_rejected(fixture_schema):
    with pytest.raises(ValueError, match="split must be"):
        CMAPSSLoader("validation", fixture_schema)


# ---------------------------------------------------------------------------
# Lifetime table: censoring semantics
# ---------------------------------------------------------------------------
def test_lifetime_table_separates_censoring_time_from_lifetime(cmapss_dir, fixture_schema):
    life = CMAPSSLifetimeBuilder(fixture_schema, data_dir=cmapss_dir).build()

    train = life[life["split"] == "train"].set_index("unit")
    test = life[life["split"] == "test"].set_index("unit")

    # Train runs to failure: duration IS the lifetime, event observed.
    assert (train["event"] == 1).all()
    assert train.loc[1, "duration"] == 5
    assert train.loc[2, "duration"] == 8
    assert (train["duration"] == train["true_duration"]).all()

    # Test is right-censored: duration is a CENSORING time, and the true
    # lifetime is only recoverable by adding the withheld RUL.
    assert (test["event"] == 0).all()
    assert test.loc[1, "duration"] == 3
    assert test.loc[1, "true_duration"] == 3 + 10  # RUL line 1
    assert test.loc[2, "true_duration"] == 4 + 20  # RUL line 2
    assert (test["true_duration"] > test["duration"]).all()


def test_rul_is_matched_to_units_by_position(cmapss_dir, fixture_schema):
    rul = CMAPSSRULLoader(fixture_schema, data_dir=cmapss_dir).load_series()
    assert list(rul.index) == [1, 2]
    assert rul.loc[1] == 10
    assert rul.loc[2] == 20


def test_wrong_rul_length_is_caught(tmp_path, cmapss_dir, fixture_schema):
    (cmapss_dir / "RUL_FIX.txt").write_text("10\n20\n30\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="expected 2 RUL values, got 3"):
        CMAPSSRULLoader(fixture_schema, data_dir=cmapss_dir).load_series()
