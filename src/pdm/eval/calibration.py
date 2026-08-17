"""Calibration diagnostics: reliability curves and the Brier decomposition.

PR-AUC asks whether a model *ranks* failures above non-failures. It says
nothing about whether a predicted "70% chance of failure" means anything --
a model that outputs 0.9 for real failures and 0.8 for everything else ranks
perfectly while every number is wrong. Calibration is that second question:
among rows the model called p% likely to fail, did about p% of them actually
fail? A **reliability curve** answers it directly: bucket predictions, and
for each bucket compare the mean predicted probability to the observed
failure rate in that bucket. Perfect calibration is the diagonal.

Layer 4 computes expected costs directly from predicted probabilities, so a
model can look excellent on PR-AUC and still produce confidently wrong
currency figures if it is not calibrated. That is what this module exists to
catch, and why Brier score is a locked primary metric.

## The exact identity, and where it stops being exact

Murphy's (1973) decomposition splits the Brier score into three parts:

    reliability    mean squared gap between predicted and observed
                   frequency, per bucket -- calibration error (lower better)
    resolution     how much bucket-observed-frequencies vary from the
                   overall base rate -- ability to separate cases at all
                   (higher better; a model that always predicts the base
                   rate has resolution 0)
    uncertainty    the base rate's own variance, o*(1-o). Fixed by the data,
                   not the model -- the irreducible part of the score.

`reliability - resolution + uncertainty` reconstructs the Brier score
EXACTLY only when every prediction inside a bucket is replaced by that
bucket's mean prediction first. Our predictions are continuous, not
literally identical within a bucket, so the exact identity holds against a
*binned* Brier score (`BrierDecomposition.binned_brier`), not against the
raw per-row score `pdm.eval.metrics.brier` computes (`BrierDecomposition.
brier`). The two converge as bucket count grows; for a finite bucket count
there is a real, non-zero gap. Both are reported rather than treating them
as one number, because asserting an inexact identity as exact is exactly the
kind of thing that looks like a correct report and is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import brier


def _validate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    if y_true.shape != y_prob.shape:
        raise ValueError(f"length mismatch: y_true has {y_true.shape[0]}, y_prob has {y_prob.shape[0]}")
    if y_true.size == 0:
        raise ValueError("empty input")

    labels = set(np.unique(y_true).tolist())
    if not labels <= {0, 1}:
        raise ValueError(f"y_true must be binary 0/1; found {sorted(labels)}")
    if y_prob.min() < 0.0 or y_prob.max() > 1.0:
        raise ValueError(f"y_prob outside [0, 1]: min={y_prob.min()}, max={y_prob.max()}")
    return y_true, y_prob


@dataclass
class CalibrationCurve:
    """Per-bucket reliability data.

    One row per bucket: how many predictions landed there, what they
    predicted on average, and what fraction of them actually failed. A
    perfectly calibrated model has `mean_predicted == observed_frequency` in
    every row.
    """

    bins: pd.DataFrame  # columns: count, mean_predicted, observed_frequency
    strategy: str
    n_bins_requested: int

    @property
    def n_bins_used(self) -> int:
        """May be less than requested: `quantile` collapses duplicate bin
        edges, which happens when many predictions share a value -- e.g.
        Naive Bayes assigning near-identical probabilities to a cluster of
        rows, or a degenerate all-identical-prediction model collapsing to
        a single bucket outright.
        """
        return len(self.bins)

    def max_gap(self) -> float:
        """Largest |mean_predicted - observed_frequency| across buckets --
        the single worst-calibrated region, in probability points.
        """
        return float((self.bins["mean_predicted"] - self.bins["observed_frequency"]).abs().max())


def calibration_curve(y_true, y_prob, n_bins: int = 10, strategy: str = "quantile") -> CalibrationCurve:
    """Bucket predictions and compare mean prediction to observed frequency per bucket.

    `strategy="quantile"` (equal-count buckets) is the sane default at AI4I's
    3.39% base rate: with `strategy="uniform"` (equal-width buckets), nearly
    every prediction falls in the first bucket or two and the rest sit empty,
    leaving a reliability curve with almost all its information in one point.
    """
    y_true, y_prob = _validate_probabilities(y_true, y_prob)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if strategy not in ("uniform", "quantile"):
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")

    if np.unique(y_prob).size == 1:
        # A model that predicts one number for everyone (e.g. dummy_prior, or
        # a degenerate real one) has no distinct values to carve into bins.
        # Both `pd.cut` and `pd.qcut` respond to that by returning all-NaN
        # bin labels rather than raising -- which `groupby` then silently
        # drops, returning an EMPTY curve for a perfectly well-defined case
        # (one bucket, zero resolution). That silent-empty-result is exactly
        # the failure mode this project watches for, so it is handled
        # explicitly here instead of falling through to it.
        summary = pd.DataFrame(
            {
                "count": [y_true.size],
                "mean_predicted": [float(y_prob[0])],
                "observed_frequency": [float(y_true.mean())],
            }
        )
        return CalibrationCurve(bins=summary, strategy=strategy, n_bins_requested=n_bins)

    if strategy == "uniform":
        bin_id = pd.cut(y_prob, bins=n_bins, include_lowest=True)
    else:
        bin_id = pd.qcut(y_prob, q=n_bins, duplicates="drop")

    frame = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "bin": bin_id})
    summary = (
        frame.groupby("bin", observed=True)
        .agg(
            count=("y_true", "size"), mean_predicted=("y_prob", "mean"), observed_frequency=("y_true", "mean")
        )
        .reset_index(drop=True)
    )

    return CalibrationCurve(bins=summary, strategy=strategy, n_bins_requested=n_bins)


@dataclass
class BrierDecomposition:
    """Murphy's (1973) three-way split of the (binned) Brier score.

    See the module docstring for exactly what `binned_brier` versus `brier`
    means and why they are not the same number.
    """

    reliability: float
    resolution: float
    uncertainty: float
    binned_brier: float
    brier: float
    n_bins_used: int

    @property
    def reconstructed_brier(self) -> float:
        """reliability - resolution + uncertainty. Equals `binned_brier`
        exactly (up to floating point); provided so the identity is visibly
        checkable rather than only asserted in a docstring.
        """
        return self.reliability - self.resolution + self.uncertainty


def brier_decomposition(y_true, y_prob, n_bins: int = 10, strategy: str = "quantile") -> BrierDecomposition:
    """Split the Brier score into calibration error, resolution, and uncertainty.

    Uses the same bucketing as `calibration_curve` -- a reliability diagram
    and its accompanying decomposition must never disagree about what a
    "bucket" is.
    """
    y_true, y_prob = _validate_probabilities(y_true, y_prob)
    curve = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    n = y_true.size
    base_rate = float(y_true.mean())
    weights = curve.bins["count"] / n

    reliability = float(
        (weights * (curve.bins["mean_predicted"] - curve.bins["observed_frequency"]) ** 2).sum()
    )
    resolution = float((weights * (curve.bins["observed_frequency"] - base_rate) ** 2).sum())
    uncertainty = base_rate * (1.0 - base_rate)
    binned_brier = reliability - resolution + uncertainty

    return BrierDecomposition(
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        binned_brier=binned_brier,
        brier=brier(y_true, y_prob),
        n_bins_used=curve.n_bins_used,
    )


def plot_reliability_diagram(curve: CalibrationCurve, ax=None, label: str | None = None):
    """Render a reliability diagram onto a matplotlib Axes.

    Draws mean predicted probability against observed frequency per bucket,
    plus the y=x perfect-calibration reference line. Returns the Axes so a
    caller can overlay several curves (e.g. GNB against BLR) on one figure,
    or save it -- this function only draws; where a figure gets saved is a
    reporting decision, not this module's.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    lo = min(0.0, float(curve.bins["mean_predicted"].min()))
    hi = max(1.0, float(curve.bins["mean_predicted"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="grey", label="perfect calibration")
    ax.plot(curve.bins["mean_predicted"], curve.bins["observed_frequency"], marker="o", label=label)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    if label is not None:
        ax.legend()
    return ax
