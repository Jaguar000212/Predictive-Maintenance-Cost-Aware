"""Tests for the Layer 4 cost arithmetic (docs/DECISIONS.md D11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from pdm.config import CostConfig, CVConfig
from pdm.decision.cost_model import (
    cost_curve,
    cost_per_row,
    cross_validated_cost_curve,
    expected_cost,
    optimal_operating_point,
)

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


# ---------------------------------------------------------------------------
# cross_validated_cost_curve -- the honest version
#
# Fixtures mirror tests/test_cv.py's `noise` and `signal` (same seeds), since
# this module reuses that one's leakage-guard logic rather than re-deriving
# it, and its correctness rests on exactly the same "no test-fold row ever
# reaches a fit" property.
# ---------------------------------------------------------------------------
@pytest.fixture
def noise():
    """400 rows, 10% positives, features carrying no signal whatsoever."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(400, 5)), columns=[f"f{i}" for i in range(5)])
    y = (rng.random(400) < 0.10).astype(int)
    return X, y


@pytest.fixture
def signal():
    """400 rows where one feature genuinely separates the classes."""
    rng = np.random.default_rng(7)
    y = (rng.random(400) < 0.20).astype(int)
    X = pd.DataFrame(
        {"informative": y * 2.5 + rng.normal(size=400), "noise": rng.normal(size=400)},
    )
    return X, y


def _small_cv_config() -> CVConfig:
    return CVConfig(n_splits=5, n_repeats=2, random_state=0)


def test_cv_cost_curve_cannot_beat_the_two_trivial_strategies_on_pure_noise(noise):
    """Leakage guard, cost-model version of test_cv.py's PR-AUC analogue.

    An unlimited-depth tree memorises noise perfectly in-sample. With no real
    signal, the best any honestly-scored threshold can do is match whichever
    trivial strategy the cost ratio favours -- "always fire" (pay false_alarm
    on every negative) or "never fire" (pay missed_failure on every
    positive). If the honestly cross-validated optimum beats that floor by
    much, folds are leaking into each other.
    """
    X, y = noise
    result = cross_validated_cost_curve(
        lambda: DecisionTreeClassifier(random_state=0), X, y, cost=COST, cv_config=_small_cv_config()
    )
    base_rate = y.mean()
    floor = min(base_rate * COST.missed_failure, (1 - base_rate) * COST.false_alarm)
    assert result.optimal["cost_per_row_mean"] > floor * 0.8, (
        "honest CV beat the no-signal trivial-strategy floor by more than fold noise "
        "should allow -- check for a leakage path before trusting this number"
    )


def test_cv_cost_curve_matches_the_hand_computed_dummy_expectation(signal):
    """A constant-negative model never fires, at any threshold in the grid
    (predict_proba is always 0, and the lowest grid point is 0.01 > 0), so
    cost per row is the same closed-form number as the pure-arithmetic dummy
    check: base_rate * missed_failure.
    """
    X, y = signal
    result = cross_validated_cost_curve(
        lambda: DummyClassifier(strategy="constant", constant=0),
        X,
        y,
        cost=COST,
        cv_config=_small_cv_config(),
    )
    base_rate = y.mean()
    expected = base_rate * COST.missed_failure
    assert result.optimal["cost_per_row_mean"] == pytest.approx(expected, abs=0.02)
    # The curve should be flat: a model whose prediction never changes with
    # the threshold has the same cost at every threshold, within one fold.
    one_fold = result.fold_curves[(result.fold_curves["fold"] == 0) & (result.fold_curves["repeat"] == 0)]
    assert one_fold["cost_per_row"].nunique() == 1


def test_real_signal_beats_the_dummy_once_scored_honestly(signal):
    """Complement to the leakage test: real signal must still show up."""
    X, y = signal
    dummy = cross_validated_cost_curve(
        lambda: DummyClassifier(strategy="constant", constant=0),
        X,
        y,
        cost=COST,
        cv_config=_small_cv_config(),
    )
    tree = cross_validated_cost_curve(
        lambda: DecisionTreeClassifier(random_state=0, max_depth=3),
        X,
        y,
        cost=COST,
        cv_config=_small_cv_config(),
    )
    assert tree.optimal["cost_per_row_mean"] < dummy.optimal["cost_per_row_mean"]


def test_fold_count_matches_the_cv_config(noise):
    X, y = noise
    cv_config = CVConfig(n_splits=4, n_repeats=3, random_state=0)
    result = cross_validated_cost_curve(
        lambda: DummyClassifier(strategy="prior"), X, y, cost=COST, cv_config=cv_config
    )
    assert result.n_fits == 12 == cv_config.n_fits
    assert sorted(result.fold_curves["fold"].unique()) == [0, 1, 2, 3]
    assert sorted(result.fold_curves["repeat"].unique()) == [0, 1, 2]
    # Every fold, every threshold in the 99-point default grid.
    assert len(result.fold_curves) == 12 * 99


def test_same_cv_config_gives_the_same_fold_curves(noise):
    """Comparing models scored on different splits confounds model with split
    -- the same reason CrossValidator.run() is deterministic given a seed."""
    X, y = noise
    a = cross_validated_cost_curve(
        lambda: DummyClassifier(strategy="prior"), X, y, cost=COST, cv_config=_small_cv_config()
    )
    b = cross_validated_cost_curve(
        lambda: DummyClassifier(strategy="prior"), X, y, cost=COST, cv_config=_small_cv_config()
    )
    pd.testing.assert_frame_equal(a.fold_curves, b.fold_curves)


def test_a_fresh_estimator_is_built_for_every_fold(noise):
    X, y = noise
    built = []

    def factory():
        estimator = DummyClassifier(strategy="prior")
        built.append(estimator)
        return estimator

    result = cross_validated_cost_curve(factory, X, y, cost=COST, cv_config=_small_cv_config())
    assert len(built) == result.n_fits == 10
    assert len({id(e) for e in built}) == 10


def test_passing_an_instance_instead_of_a_factory_is_rejected(noise):
    X, y = noise
    with pytest.raises(TypeError, match="zero-argument callable"):
        cross_validated_cost_curve(DummyClassifier(strategy="prior"), X, y, cost=COST)


def test_non_binary_target_is_rejected(noise):
    X, _ = noise
    with pytest.raises(ValueError, match="y must be binary"):
        cross_validated_cost_curve(
            lambda: DummyClassifier(strategy="prior"), X, np.arange(400) % 3, cost=COST
        )


def test_unconfigured_cost_is_rejected_before_any_fold_runs(noise):
    """The 'no cost before the decision' guard must fire before spending time
    fitting folds, not partway through."""
    X, y = noise
    with pytest.raises(ValueError, match="pending project decision"):
        cross_validated_cost_curve(
            lambda: DummyClassifier(strategy="prior"), X, y, cost=CostConfig(missed_failure=None)
        )


def test_summary_is_indexed_by_the_full_threshold_grid(noise):
    X, y = noise
    result = cross_validated_cost_curve(
        lambda: DummyClassifier(strategy="prior"), X, y, cost=COST, cv_config=_small_cv_config()
    )
    summary = result.summary()
    assert len(summary) == 99  # MetricConfig's default grid
    assert {"threshold", "cost_per_row_mean", "cost_per_row_std"} <= set(summary.columns)
