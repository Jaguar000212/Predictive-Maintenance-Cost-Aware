"""Calibration diagnostics tests.

The Brier decomposition is the genuinely new content here, and it has a real
subtlety (see calibration.py's module docstring): the classic identity
`reliability - resolution + uncertainty` reconstructs the score you get by
replacing each row's prediction with its bucket's mean -- not the raw,
per-row Brier score. Both get checked independently here rather than assumed
equal.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend -- these tests never display a figure

import numpy as np
import pandas as pd
import pytest

from pdm.eval.calibration import brier_decomposition, calibration_curve, plot_reliability_diagram
from pdm.eval.metrics import brier as raw_brier


# ---------------------------------------------------------------------------
# calibration_curve, against a hand-computed example
# ---------------------------------------------------------------------------
def test_calibration_curve_matches_hand_computed_buckets():
    """Two clean clusters -- pred=0.1 (1/4 true positives) and pred=0.8 (3/4)
    -- so quantile splitting at the midpoint has an unambiguous answer.
    """
    y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8])
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])

    curve = calibration_curve(y_true, y_prob, n_bins=2, strategy="quantile")

    assert curve.n_bins_used == 2
    low, high = curve.bins.sort_values("mean_predicted").itertuples(index=False)
    assert (
        low.count == 4
        and low.mean_predicted == pytest.approx(0.1)
        and low.observed_frequency == pytest.approx(0.25)
    )
    assert (
        high.count == 4
        and high.mean_predicted == pytest.approx(0.8)
        and high.observed_frequency == pytest.approx(0.75)
    )


def test_max_gap_reports_the_worst_bucket():
    y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8])
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    curve = calibration_curve(y_true, y_prob, n_bins=2, strategy="quantile")
    # |0.1-0.25| = 0.15, |0.8-0.75| = 0.05 -> the worse bucket is 0.15
    assert curve.max_gap() == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Brier decomposition -- the exact identity, checked independently
# ---------------------------------------------------------------------------
def test_binned_brier_matches_direct_row_level_computation():
    """Independent check of the Murphy identity: recompute the
    "replace-by-bucket-mean" Brier score directly at the row level, without
    going through reliability/resolution/uncertainty at all.
    """
    rng = np.random.default_rng(5)
    n = 500
    y_prob = rng.uniform(0, 1, n)
    y_true = (rng.random(n) < y_prob).astype(int)
    n_bins = 8

    decomp = brier_decomposition(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    bin_id = pd.qcut(y_prob, q=n_bins, duplicates="drop")
    bucket_mean = pd.Series(y_prob).groupby(bin_id, observed=True).transform("mean")
    direct_binned_brier = float(np.mean((bucket_mean.to_numpy() - y_true) ** 2))

    assert decomp.binned_brier == pytest.approx(direct_binned_brier, abs=1e-10)
    assert decomp.reconstructed_brier == pytest.approx(direct_binned_brier, abs=1e-10)


def test_binned_and_raw_brier_are_not_the_same_number_in_general():
    """The module docstring's central warning: with real within-bucket
    variation in continuous predictions, binned_brier != metrics.brier.
    Both must still be small, sane numbers in the same ballpark.
    """
    rng = np.random.default_rng(5)
    n = 500
    y_prob = rng.uniform(0, 1, n)
    y_true = (rng.random(n) < y_prob).astype(int)

    decomp = brier_decomposition(y_true, y_prob, n_bins=8, strategy="quantile")

    assert decomp.brier == pytest.approx(raw_brier(y_true, y_prob))
    assert abs(decomp.binned_brier - decomp.brier) > 1e-6  # a real, non-zero gap exists
    assert abs(decomp.binned_brier - decomp.brier) < 0.01  # but it is small, not a different score


def test_hand_computed_decomposition_on_the_two_cluster_example():
    """Same data as the calibration_curve hand-computation, carried through
    to reliability/resolution/uncertainty by hand.

    base_rate = 0.5; reliability = 0.5*(0.1-0.25)^2 + 0.5*(0.8-0.75)^2 = 0.0125
    resolution = 0.5*(0.25-0.5)^2 + 0.5*(0.75-0.5)^2 = 0.0625
    uncertainty = 0.5*0.5 = 0.25
    binned_brier = 0.0125 - 0.0625 + 0.25 = 0.2
    """
    y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8])
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])

    decomp = brier_decomposition(y_true, y_prob, n_bins=2, strategy="quantile")

    assert decomp.reliability == pytest.approx(0.0125)
    assert decomp.resolution == pytest.approx(0.0625)
    assert decomp.uncertainty == pytest.approx(0.25)
    assert decomp.binned_brier == pytest.approx(0.2)
    # No within-bucket dispersion in this constructed example (every row's
    # own prediction already equals its bucket's mean), so here -- and only
    # here -- binned and raw agree exactly.
    assert decomp.brier == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------
def test_constant_predictions_collapse_to_a_single_bucket():
    """A model with zero resolution (predicts one number for everyone) must
    show that as n_bins_used == 1, not raise on a degenerate quantile split.
    """
    y_true = np.array([0, 1, 0, 1, 0])
    y_prob = np.full(5, 0.4)

    curve = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    assert curve.n_bins_used == 1
    assert curve.bins["count"].iloc[0] == 5

    decomp = brier_decomposition(y_true, y_prob, n_bins=10, strategy="quantile")
    assert decomp.resolution == pytest.approx(0.0, abs=1e-12)  # one bucket cannot separate anything


def test_quantile_strategy_keeps_buckets_populated_under_imbalance():
    """The reason quantile is the default: at a low base rate, equal-width
    bins leave most of the [0, 1] range empty.
    """
    rng = np.random.default_rng(6)
    y_prob = np.concatenate([rng.uniform(0, 0.05, 966), rng.uniform(0.3, 0.9, 34)])
    y_true = np.concatenate([np.zeros(966), (rng.random(34) < 0.6).astype(int)])

    quantile_curve = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    uniform_curve = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    assert uniform_curve.n_bins_used < quantile_curve.n_bins_used


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="length mismatch"):
        calibration_curve(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_rejects_probabilities_outside_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibration_curve(np.array([0, 1]), np.array([0.1, 1.5]))


def test_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="binary"):
        calibration_curve(np.array([0, 2]), np.array([0.1, 0.9]))


def test_rejects_non_positive_n_bins():
    with pytest.raises(ValueError, match="n_bins"):
        calibration_curve(np.array([0, 1]), np.array([0.1, 0.9]), n_bins=0)


def test_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        calibration_curve(np.array([0, 1]), np.array([0.1, 0.9]), strategy="bogus")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def test_plot_reliability_diagram_returns_axes_with_both_lines():
    y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8])
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    curve = calibration_curve(y_true, y_prob, n_bins=2, strategy="quantile")

    ax = plot_reliability_diagram(curve, label="test model")

    assert len(ax.lines) == 2  # the diagonal reference line, plus the curve itself
    assert ax.get_xlabel() == "mean predicted probability"
    assert ax.get_ylabel() == "observed frequency"
