"""Physics feature tests.

The formulas are simple, so the point of these tests is not "does subtraction
work" -- it is pinning the one place a unit mistake would be invisible: the
2*pi/60 conversion in `power_w`. Drop or misplace that term and the column is
still monotonic in true power, PR-AUC can still look reasonable, and nothing
raises. Only a hand-computed value catches it.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from pdm.config import AI4ISchema
from pdm.features.physics import PhysicsFeatures


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "air_temp_k": [300.0, 295.0],
            "process_temp_k": [310.0, 305.5],
            "rot_speed_rpm": [1500.0, 1800.0],
            "torque_nm": [40.0, 25.0],
            "tool_wear_min": [10.0, 200.0],
            "type": ["L", "M"],
        }
    )


# ---------------------------------------------------------------------------
# Formulas, against hand-computed values
# ---------------------------------------------------------------------------
def test_temp_diff_is_process_minus_air():
    out = PhysicsFeatures().fit_transform(_frame())
    assert out["temp_diff"].tolist() == pytest.approx([10.0, 10.5])


def test_power_w_matches_the_2pi_over_60_conversion():
    """The formula the PWF band (3500-9000 W) is defined against.

    Row 0: torque=40 Nm, rpm=1500 -> 40 * (2*pi*1500/60) = 40 * 157.0796... W.
    Computed independently of the implementation to catch a transcription bug.
    """
    out = PhysicsFeatures().fit_transform(_frame())
    expected_0 = 40.0 * (2 * math.pi * 1500.0 / 60.0)
    expected_1 = 25.0 * (2 * math.pi * 1800.0 / 60.0)
    assert out["power_w"].tolist() == pytest.approx([expected_0, expected_1])
    assert expected_0 == pytest.approx(6283.19, abs=0.01)


def test_power_w_without_the_conversion_would_land_outside_the_pwf_band():
    """Guards against the silent failure mode described in the module docstring.

    Skipping 2*pi/60 (i.e. torque * rpm directly) produces a number that is
    still monotonic in true power -- which is exactly why the bug is silent --
    but for these inputs it is nowhere near the 3500-9000 W PWF band, while the
    correctly-converted value is inside it.
    """
    row = _frame().iloc[[0]]
    correct = PhysicsFeatures().fit_transform(row)["power_w"].iloc[0]
    wrong_units = row["torque_nm"].iloc[0] * row["rot_speed_rpm"].iloc[0]

    assert 3500.0 <= correct <= 9000.0
    assert not (3500.0 <= wrong_units <= 9000.0)


def test_wear_strain_is_tool_wear_times_torque():
    out = PhysicsFeatures().fit_transform(_frame())
    assert out["wear_strain"].tolist() == pytest.approx([10.0 * 40.0, 200.0 * 25.0])


# ---------------------------------------------------------------------------
# Transformer behaviour
# ---------------------------------------------------------------------------
def test_original_columns_are_preserved_not_replaced():
    out = PhysicsFeatures().fit_transform(_frame())
    for col in ("air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min", "type"):
        assert col in out.columns


def test_missing_column_raises_rather_than_silently_producing_nan():
    df = _frame().drop(columns=["torque_nm"])
    with pytest.raises(ValueError, match="torque_nm"):
        PhysicsFeatures().fit_transform(df)


def test_stateless_across_different_frames():
    """No cross-row statistic is fit, so a fold boundary has nothing to leak.

    Fitting on one frame and transforming a completely different one must give
    the same per-row answer as fitting and transforming that frame directly --
    proof there is no stored mean, scale, or other train-derived state.
    """
    transformer = PhysicsFeatures().fit(_frame())
    transformed_via_fit_elsewhere = transformer.transform(_frame())
    transformed_directly = PhysicsFeatures().fit_transform(_frame())
    pd.testing.assert_frame_equal(transformed_via_fit_elsewhere, transformed_directly)


def test_uses_schema_column_names_not_hardcoded_strings():
    """A renamed raw column should move the formula with it, not break silently."""
    schema = AI4ISchema(
        air_temp_col="air_t",
        process_temp_col="proc_t",
        rot_speed_col="rpm",
        torque_col="torque",
        tool_wear_col="wear",
    )
    df = pd.DataFrame(
        {"air_t": [300.0], "proc_t": [310.0], "rpm": [1500.0], "torque": [40.0], "wear": [10.0]}
    )
    out = PhysicsFeatures(schema).fit_transform(df)
    assert out["temp_diff"].iloc[0] == pytest.approx(10.0)


def test_get_feature_names_out_appends_engineered_names():
    names = PhysicsFeatures().get_feature_names_out(["air_temp_k", "torque_nm"])
    assert list(names) == ["air_temp_k", "torque_nm", "temp_diff", "power_w", "wear_strain"]


def test_engineered_features_are_recorded_on_the_schema():
    """So a results JSON shows which engineered columns a run used."""
    assert AI4ISchema().engineered_features == ("temp_diff", "power_w", "wear_strain")
