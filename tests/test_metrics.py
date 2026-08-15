"""Metric tests against hand-computed values.

Every expected number here is derived by hand in the comment above it. Testing
a metric against another library's implementation of the same metric only proves
the two agree, not that either is right.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sklearn.metrics import auc, precision_recall_curve

from pdm.eval import metrics as M


# ---------------------------------------------------------------------------
# Threshold-bound metrics, hand-computed
# ---------------------------------------------------------------------------
def test_perfect_separation():
    # y=[0,0,0,0,1], p=[.1,.2,.3,.4,.9] at t=0.5 -> pred=[0,0,0,0,1]
    # tp=1 fp=0 fn=0 tn=4  =>  recall=1, precision=1, f2=1
    y = [0, 0, 0, 0, 1]
    p = [0.1, 0.2, 0.3, 0.4, 0.9]
    m = M.classification_metrics(y, p, threshold=0.5)

    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 0, 0, 4)
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["f2"] == 1.0
    assert m["pr_auc"] == 1.0
    # brier = (.01 + .04 + .09 + .16 + .01) / 5 = 0.31 / 5 = 0.062
    assert m["brier"] == pytest.approx(0.062)


def test_f2_weights_recall_four_to_one():
    # y=[1,1,0,0], p=[.9,.1,.1,.1] at t=0.5 -> pred=[1,0,0,0]
    # tp=1 fp=0 fn=1 tn=2  =>  precision=1.0, recall=0.5
    # f2 = 5 * 1.0 * 0.5 / (4 * 1.0 + 0.5) = 2.5 / 4.5
    m = M.classification_metrics([1, 1, 0, 0], [0.9, 0.1, 0.1, 0.1], threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert m["f2"] == pytest.approx(2.5 / 4.5)
    # F1 would be 2*1*.5/(1+.5) = 0.667; F2 is lower because it punishes the
    # missed positive harder. Guards against beta being silently 1.
    assert m["f2"] < 2 * 1.0 * 0.5 / (1.0 + 0.5)


def test_f2_equals_precision_equals_recall():
    # y=[0,0,1,1], p=[.1,.6,.4,.9] at t=0.5 -> pred=[0,1,0,1]
    # tp=1 fp=1 fn=1 tn=1  =>  precision=recall=0.5, so f_beta=0.5 for any beta
    m = M.classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f2"] == pytest.approx(0.5)


def test_threshold_boundary_is_inclusive():
    # p == threshold must count as a positive prediction (>=, not >).
    m = M.classification_metrics([1, 0], [0.5, 0.4], threshold=0.5)
    assert m["tp"] == 1


def test_recall_is_monotone_non_increasing_in_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    recalls = [M.classification_metrics(y, p, threshold=t)["recall"] for t in np.linspace(0, 1, 21)]
    assert all(a >= b for a, b in itertools.pairwise(recalls))


# ---------------------------------------------------------------------------
# Constant-negative dummy: the Week 1 harness-validation criterion
# ---------------------------------------------------------------------------
def test_constant_negative_dummy_hits_base_rate():
    # 1 positive in 4 => base rate 0.25.
    # Predicting p=0 everywhere: recall=0, precision undefined->0, f2=0.
    # brier = mean((0 - y)^2) = mean(y) = 0.25  == base rate
    # pr_auc with all-equal scores = base rate = 0.25
    y = [0, 0, 0, 1]
    p = [0.0, 0.0, 0.0, 0.0]
    m = M.classification_metrics(y, p, threshold=0.5)

    assert m["recall"] == 0.0
    assert m["precision"] == 0.0  # never fired, not "always wrong"
    assert m["f2"] == 0.0
    assert (m["tp"], m["fp"]) == (0, 0)
    assert m["brier"] == pytest.approx(0.25)
    assert m["pr_auc"] == pytest.approx(0.25)
    assert m["base_rate"] == pytest.approx(0.25)


def test_constant_negative_on_real_ai4i_base_rate():
    """The exact numbers CLAUDE.md predicts for the Week 1 gate."""
    from pdm.loaders import load_ai4i

    y = load_ai4i()["machine_failure"].to_numpy()
    p = np.zeros_like(y, dtype=float)
    m = M.classification_metrics(y, p, threshold=0.5)

    assert m["base_rate"] == pytest.approx(0.0339)
    assert m["recall"] == 0.0
    assert m["pr_auc"] == pytest.approx(0.0339, abs=1e-4)
    assert m["brier"] == pytest.approx(0.0339, abs=1e-4)


# ---------------------------------------------------------------------------
# PR-AUC definition
# ---------------------------------------------------------------------------
def test_pr_auc_is_average_precision_not_trapezoidal():
    """Guard the definition: the two disagree, and only one is intended.

    `auc(recall, precision)` interpolates linearly between operating points,
    which overstates the area. If someone swaps the implementation, this fails.
    """
    rng = np.random.default_rng(7)
    y = (rng.random(300) < 0.05).astype(int)
    p = np.clip(0.3 * y + rng.random(300) * 0.7, 0, 1)

    prec, rec, _ = precision_recall_curve(y, p)
    trapezoidal = auc(rec, prec)

    assert M.pr_auc(y, p) != pytest.approx(trapezoidal, abs=1e-6)


def test_pr_auc_floor_is_base_rate_not_half():
    """Unlike ROC-AUC, a non-discriminative model scores the prevalence."""
    rng = np.random.default_rng(3)
    y = (rng.random(20_000) < 0.03).astype(int)
    p = rng.random(20_000)  # pure noise, no signal
    assert M.pr_auc(y, p) == pytest.approx(y.mean(), abs=0.01)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_rejects_scores_outside_unit_interval():
    # Decision-function output passed where predict_proba was expected.
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        M.classification_metrics([0, 1], [-1.4, 2.7], threshold=0.5)


def test_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        M.classification_metrics([0, 1, 1], [0.1, 0.9], threshold=0.5)


def test_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="must be binary"):
        M.classification_metrics([0, 1, 2], [0.1, 0.5, 0.9], threshold=0.5)


def test_rejects_threshold_outside_unit_interval():
    with pytest.raises(ValueError, match=r"threshold must lie in \[0, 1\]"):
        M.classification_metrics([0, 1], [0.1, 0.9], threshold=1.5)


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="empty input"):
        M.classification_metrics([], [], threshold=0.5)


def test_pr_auc_and_recall_raise_without_positives():
    with pytest.raises(ValueError, match="no positive samples"):
        M.pr_auc([0, 0, 0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="no positive samples"):
        M.recall([0, 0, 0], [0, 0, 0])


def test_confusion_counts_survives_single_class_prediction():
    """labels=[0,1] guard: a fold predicting one class must not reshape the matrix."""
    c = M.confusion_counts([0, 0, 1, 1], [0, 0, 0, 0])
    assert c == {"tn": 2, "fp": 0, "fn": 2, "tp": 0}


# ---------------------------------------------------------------------------
# Accuracy must not exist
# ---------------------------------------------------------------------------
def test_accuracy_is_not_implemented():
    """Locked decision: accuracy is banned. Absent, not merely discouraged."""
    assert not hasattr(M, "accuracy")
    assert "accuracy" not in M.METRIC_NAMES
    assert "accuracy" not in M.classification_metrics([0, 1], [0.1, 0.9], threshold=0.5)
