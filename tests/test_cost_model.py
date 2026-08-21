"""Tests for the Layer 4 cost arithmetic (docs/DECISIONS.md D11)."""

from __future__ import annotations

import pandas as pd
import pytest

from pdm.config import CostConfig
from pdm.decision.cost_model import cost_curve, cost_per_row, expected_cost, optimal_operating_point

COST = CostConfig()  # default D11 ratio: 10 / 1 / 0.5


def test_expected_cost_hand_computed():
    """3 missed failures, 4 false alarms, 2 caught -- by hand:
    3*10 + 4*1 + 2*0.5 = 30 + 4 + 1 = 35.
    """
    counts = {"tn": 91, "fp": 4, "fn": 3, "tp": 2}
    assert expected_cost(counts, COST) == pytest.approx(35.0)


def test_true_negatives_are_free():
    counts_with_tn = {"tn": 1000, "fp": 0, "fn": 0, "tp": 0}
    assert expected_cost(counts_with_tn, COST) == 0.0


def test_cost_per_row_divides_by_every_row_including_tn():
    counts = {"tn": 91, "fp": 4, "fn": 3, "tp": 2}  # n = 100
    assert cost_per_row(counts, COST) == pytest.approx(0.35)


def test_cost_per_row_matches_the_stated_dummy_expectation():
    """The number stated before running anything: a constant-negative model
    (never fires) at AI4I's ~3.39% base rate has fn = base rate, fp = tp = 0,
    so cost per row = 0.0339 * 10 = 0.339.
    """
    counts = {"tn": 9661, "fp": 0, "fn": 339, "tp": 0}  # 10,000 rows, 3.39% positive
    assert cost_per_row(counts, COST) == pytest.approx(0.339, abs=1e-6)


def test_missing_keys_are_rejected_not_defaulted():
    """A caller passing a plain confusion_matrix().ravel() without labels=[0, 1]
    could easily transpose fp/fn -- requiring the named keys explicitly, rather
    than accepting a bare 4-tuple, is the guard against that."""
    with pytest.raises(KeyError, match="missing required keys"):
        expected_cost({"tn": 1, "fp": 1}, COST)


def test_zero_rows_is_an_error_not_a_zero_cost():
    with pytest.raises(ValueError, match="zero rows"):
        cost_per_row({"tn": 0, "fp": 0, "fn": 0, "tp": 0}, COST)


def test_unconfigured_cost_still_raises_through_this_module():
    """The 'no cost before the decision' guard (now only tripped by an explicit
    override) must still fire from inside cost_model, not just from CostConfig
    used directly."""
    unset = CostConfig(missed_failure=None)
    with pytest.raises(ValueError, match="pending project decision"):
        expected_cost({"tn": 1, "fp": 0, "fn": 0, "tp": 0}, unset)


# ---------------------------------------------------------------------------
# cost_curve / optimal_operating_point
# ---------------------------------------------------------------------------
def _fake_sweep() -> pd.DataFrame:
    """A tiny 3-threshold sweep standing in for MetricSuite.sweep()'s output.

    Low threshold: fires on everything -> many false alarms, no misses.
    Mid threshold: a balanced point.
    High threshold: fires on nothing -> no false alarms, all misses.
    """
    return pd.DataFrame(
        [
            {"threshold": 0.1, "tn": 50, "fp": 50, "fn": 0, "tp": 10},
            {"threshold": 0.5, "tn": 90, "fp": 10, "fn": 4, "tp": 6},
            {"threshold": 0.9, "tn": 100, "fp": 0, "fn": 10, "tp": 0},
        ]
    )


def test_cost_curve_adds_a_cost_per_row_column():
    curve = cost_curve(_fake_sweep(), COST)
    assert "cost_per_row" in curve.columns
    assert len(curve) == 3

    # threshold 0.5 by hand: (4*10 + 10*1 + 6*0.5) / 110 = (40+10+3)/110
    row = curve.loc[curve["threshold"] == 0.5].iloc[0]
    assert row["cost_per_row"] == pytest.approx((40 + 10 + 3) / 110)


def test_optimal_operating_point_picks_the_minimum():
    curve = cost_curve(_fake_sweep(), COST)
    best = optimal_operating_point(curve)
    # threshold 0.1 (fire on everything): (0*10 + 50*1 + 10*0.5)/110 = 55/110 = 0.5
    # threshold 0.5:                     (4*10 + 10*1 + 6*0.5)/110 = 53/110 ~= 0.482
    # threshold 0.9 (fire on nothing):    (10*10 + 0 + 0)/110       = 100/110 ~= 0.909
    # 0.5 is the cheapest of the three.
    assert best["threshold"] == pytest.approx(0.5)


def test_cost_curve_rejects_a_sweep_missing_confusion_counts():
    with pytest.raises(KeyError, match="missing columns"):
        cost_curve(pd.DataFrame({"threshold": [0.5]}), COST)


def test_optimal_operating_point_rejects_a_curve_without_cost():
    with pytest.raises(KeyError, match="cost_per_row"):
        optimal_operating_point(_fake_sweep())
