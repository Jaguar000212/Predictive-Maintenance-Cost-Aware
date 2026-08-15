"""Classification metrics for the AI4I layers.

Accuracy is deliberately not implemented. At a 3.39% base rate a
constant-negative model scores 96.61%, so the metric is actively misleading
here. It is absent rather than discouraged so it cannot be called by accident.

Two families of metric live here and they are not interchangeable:

  threshold-free   pr_auc, brier         consume PROBABILITIES, score the whole
                                         ranking / calibration
  threshold-bound  recall, precision,    consume a hard 0/1 decision, and only
                   f_beta                exist relative to a chosen threshold

Passing predicted labels where probabilities are expected returns a plausible
number rather than raising, which is why `classification_metrics` takes the
threshold as a required argument and validates the probability range.

Cost per 1000h is NOT here. It belongs in `pdm.decision` alongside the cost
constants, because it needs inputs this module has no business knowing about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix

from ..config import MetricConfig

# Keys produced by `classification_metrics`, in report order.
METRIC_NAMES = (
    "pr_auc",
    "recall",
    "precision",
    "f2",
    "brier",
)


def _validate(
    y_true: np.ndarray, y_score: np.ndarray, *, probabilities: bool
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    if y_true.shape != y_score.shape:
        raise ValueError(f"length mismatch: y_true has {y_true.shape[0]}, y_score has {y_score.shape[0]}")
    if y_true.size == 0:
        raise ValueError("empty input")

    labels = set(np.unique(y_true).tolist())
    if not labels <= {0, 1}:
        raise ValueError(f"y_true must be binary 0/1; found {sorted(labels)}")

    if probabilities and (y_score.min() < 0.0 or y_score.max() > 1.0):
        raise ValueError(
            f"y_prob outside [0, 1]: min={y_score.min()}, max={y_score.max()}. "
            "Decision-function scores are not probabilities -- pass predict_proba()[:, 1]."
        )
    return y_true, y_score


# ---------------------------------------------------------------------------
# Threshold-free
# ---------------------------------------------------------------------------
def pr_auc(y_true, y_prob) -> float:
    """Area under the precision-recall curve, as average precision.

    Uses `average_precision_score`, NOT `auc(recall, precision)`. Both are
    called "PR-AUC" in the literature and they disagree: trapezoidal
    interpolation between operating points assumes the PR curve is linear
    between them, which it is not, and the result is optimistically biased.
    Average precision is the step-wise sum and is the intended quantity.

    Baseline: a model with no discriminative power scores the positive base
    rate (~0.0339 on AI4I), not 0.5. Unlike ROC-AUC, the floor moves with
    prevalence, so PR-AUC is only comparable across sets with the same base
    rate.
    """
    y_true, y_prob = _validate(y_true, y_prob, probabilities=True)
    if y_true.sum() == 0:
        raise ValueError("PR-AUC is undefined with no positive samples")
    return float(average_precision_score(y_true, y_prob))


def brier(y_true, y_prob) -> float:
    """Mean squared error between predicted probability and outcome. Lower is better.

    This is the calibration metric. A model can rank well (high PR-AUC) while
    its probabilities are systematically shifted, and Brier is what exposes it.
    That matters because Layer 4 computes expected costs directly from these
    probabilities: a miscalibrated model produces confident, plausible, wrong
    currency figures with no error raised.

    Reference point: predicting the constant 0 gives Brier == the positive base
    rate (0.0339 on AI4I). Any model scoring worse than its own base rate is
    worse than useless.
    """
    y_true, y_prob = _validate(y_true, y_prob, probabilities=True)
    return float(brier_score_loss(y_true, y_prob))


# ---------------------------------------------------------------------------
# Threshold-bound
# ---------------------------------------------------------------------------
def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Return tn/fp/fn/tp as plain ints.

    `labels=[0, 1]` is required: without it, a fold or a dummy that predicts a
    single class returns a 1x1 matrix and unpacking raises -- or worse, silently
    misassigns the counts.
    """
    y_true, y_pred = _validate(y_true, y_pred, probabilities=False)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def recall(y_true, y_pred) -> float:
    """TP / (TP + FN). Share of real failures caught.

    Undefined with no positives in y_true; that is an error, not a 0.
    """
    c = confusion_counts(y_true, y_pred)
    denom = c["tp"] + c["fn"]
    if denom == 0:
        raise ValueError("recall is undefined with no positive samples in y_true")
    return c["tp"] / denom


def precision(y_true, y_pred) -> float:
    """TP / (TP + FP). Share of alarms that were real.

    Genuinely undefined when the model raises no alarms at all (0/0). Returns
    0.0 by convention so downstream aggregation does not propagate NaN -- but
    0.0 here means "never fired", not "always wrong". `confusion_counts` is the
    way to tell those apart.
    """
    c = confusion_counts(y_true, y_pred)
    denom = c["tp"] + c["fp"]
    if denom == 0:
        return 0.0
    return c["tp"] / denom


def f_beta(y_true, y_pred, beta: float = 2.0) -> float:
    """Weighted harmonic mean of precision and recall.

    beta = 2 weights recall beta^2 = 4x precision, which encodes "a missed
    failure costs about four times a false alarm".

    That 4:1 is a FIXED ASSUMPTION baked into the metric, and it is a stand-in
    for the real cost ratio that Layer 4 will supply. When F2 and the cost model
    disagree about which model wins, the cost model is authoritative -- F2 is a
    proxy that happens to be comparable with published work.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = (beta**2 * p) + r
    if denom == 0:
        # Both precision and recall are zero: no true positives at all.
        return 0.0
    return (1 + beta**2) * p * r / denom


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_prob, threshold: float, beta: float = 2.0) -> dict[str, float]:
    """Every metric at one operating point, plus the counts behind them.

    `threshold` is required and has no default. 0.5 is meaningless at a 3.39%
    base rate, and threshold choice is the variable this project is studying --
    a silent default would hide it.

    Returns the threshold-free metrics (identical for every threshold), the
    threshold-bound metrics, the raw confusion counts, and the base rate needed
    to interpret PR-AUC and Brier.
    """
    y_true, y_prob = _validate(y_true, y_prob, probabilities=True)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}")

    y_pred = (y_prob >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)

    out: dict[str, float] = {
        "pr_auc": pr_auc(y_true, y_prob),
        "recall": recall(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "f2": f_beta(y_true, y_pred, beta=beta),
        "brier": brier(y_true, y_prob),
        "threshold": float(threshold),
        "beta": float(beta),
        "base_rate": float(y_true.mean()),
        "n": int(y_true.size),
        "n_positive": int(y_true.sum()),
    }
    out.update({k: float(v) for k, v in counts.items()})
    return out


class MetricSuite:
    """The metric primitives bound to one `MetricConfig`.

    The primitives above stay free functions on purpose: they are pure maths over
    arrays with no state, and wrapping them in objects would add indirection
    without adding anything. What genuinely needs to be configurable is the
    *settings* -- beta, and the threshold grid the cost curve is built over --
    so those live in a config object and this class composes the functions
    against it.

    Carrying the config also means a run can record exactly which beta and which
    grid produced its numbers.
    """

    def __init__(self, config: MetricConfig | None = None) -> None:
        self.config = config or MetricConfig()

    def evaluate(self, y_true, y_prob, threshold: float | None = None) -> dict[str, float]:
        """Metrics at one operating point.

        `threshold` falls back to `config.report_threshold` -- an explicit,
        recorded setting rather than a hidden literal.
        """
        t = self.config.report_threshold if threshold is None else threshold
        return classification_metrics(y_true, y_prob, threshold=t, beta=self.config.beta)

    def sweep(self, y_true, y_prob) -> pd.DataFrame:
        """Metrics across the whole threshold grid, one row per threshold.

        This is the table the cost curve is built from, and the evidence for
        (or against) the claim that threshold moves outcomes more than algorithm
        choice does.
        """
        rows = [self.evaluate(y_true, y_prob, threshold=t) for t in self.config.thresholds]
        return pd.DataFrame(rows)
