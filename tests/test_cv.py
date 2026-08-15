"""Cross-validation tests.

Most of these exist to prove a negative: that no test-fold information reaches
a fit. Leakage raises nothing and produces ~0.99 scores, so the harness has to
be tested against a known-answer case where leakage would be visible.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pdm.config import CVConfig, MetricConfig
from pdm.eval.cv import CrossValidator, CVResult, compare, factory_from
from pdm.eval.metrics import MetricSuite

# Rows observed by a transformer's fit(), recorded across folds.
_FIT_ROWS: list[int] = []


class RowRecorder(BaseEstimator, TransformerMixin):
    """Passthrough transformer that records how many rows its fit() saw."""

    def fit(self, X, y=None):
        _FIT_ROWS.append(len(X))
        return self

    def transform(self, X):
        return X


@pytest.fixture(autouse=True)
def _clear_recorder():
    _FIT_ROWS.clear()
    yield
    _FIT_ROWS.clear()


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


def _small_cv(**kwargs) -> CrossValidator:
    return CrossValidator(CVConfig(n_splits=5, n_repeats=2, random_state=0), **kwargs)


# ---------------------------------------------------------------------------
# Leakage: the whole point of the module
# ---------------------------------------------------------------------------
def test_transformers_are_fitted_on_train_rows_only(noise):
    """A scaler fitted on all 400 rows would leak test-fold distribution."""
    X, y = noise
    factory = lambda: Pipeline(
        [("record", RowRecorder()), ("scale", StandardScaler()), ("clf", DummyClassifier(strategy="prior"))]
    )
    _small_cv().run(factory, X, y)

    # 5 splits => each fit sees 4/5 of 400 = 320 rows. Never 400.
    assert len(_FIT_ROWS) == 10
    assert set(_FIT_ROWS) == {320}
    assert 400 not in _FIT_ROWS


def test_a_fresh_estimator_is_built_for_every_fold(noise):
    """Reuse would carry fitted state -- and thus test-fold information -- forward."""
    X, y = noise
    built = []

    def factory():
        estimator = DummyClassifier(strategy="prior")
        built.append(estimator)
        return estimator

    result = _small_cv().run(factory, X, y)

    assert len(built) == result.n_fits == 10
    assert len({id(e) for e in built}) == 10  # ten distinct objects


def test_no_signal_data_scores_at_the_base_rate(noise):
    """The load-bearing harness test.

    An unlimited-depth tree memorises its training set perfectly. Scored on
    random labels with honest folds it must land near the base rate; if train
    and test ever overlapped it would score near 1.0 instead.
    """
    X, y = noise
    result = _small_cv().run(lambda: DecisionTreeClassifier(random_state=0), X, y)

    base_rate = y.mean()
    assert result.mean("pr_auc") == pytest.approx(base_rate, abs=0.08)
    assert result.mean("pr_auc") < 0.30, "PR-AUC far above base rate on noise implies leakage"


def test_real_signal_is_still_detected(signal):
    """Complement to the leakage test: the harness must not flatten everything."""
    X, y = signal
    result = _small_cv().run(lambda: DecisionTreeClassifier(random_state=0, max_depth=3), X, y)
    assert result.mean("pr_auc") > 0.70


def test_constant_negative_dummy_reproduces_the_week_one_gate(signal):
    X, y = signal
    result = _small_cv().run(lambda: DummyClassifier(strategy="constant", constant=0), X, y)

    assert result.mean("recall") == 0.0
    assert result.mean("f2") == 0.0
    assert result.mean("pr_auc") == pytest.approx(y.mean(), abs=0.02)
    assert result.mean("brier") == pytest.approx(y.mean(), abs=0.02)


# ---------------------------------------------------------------------------
# Fold mechanics
# ---------------------------------------------------------------------------
def test_fold_count_matches_the_config(noise):
    X, y = noise
    cv = CrossValidator(CVConfig(n_splits=4, n_repeats=3, random_state=0))
    result = cv.run(lambda: DummyClassifier(strategy="prior"), X, y)

    assert result.n_fits == 12 == cv.config.n_fits
    assert sorted(result.fold_metrics["fold"].unique()) == [0, 1, 2, 3]
    assert sorted(result.fold_metrics["repeat"].unique()) == [0, 1, 2]


def test_folds_are_stratified(noise):
    """At a low base rate an unstratified fold can contain no positives at all."""
    X, y = noise
    result = _small_cv().run(lambda: DummyClassifier(strategy="prior"), X, y)

    assert (result.fold_metrics["n_positive"] > 0).all()
    # Every fold's prevalence sits close to the overall base rate.
    prevalence = result.fold_metrics["base_rate"]
    assert prevalence.max() - prevalence.min() < 0.05


def test_same_config_gives_the_same_folds(noise):
    """Comparing models scored on different splits confounds model with split."""
    X, y = noise
    a = _small_cv().run(lambda: DummyClassifier(strategy="prior"), X, y)
    b = _small_cv().run(lambda: DummyClassifier(strategy="prior"), X, y)
    pd.testing.assert_frame_equal(a.fold_metrics, b.fold_metrics)


def test_metric_config_flows_through_to_every_fold(noise):
    X, y = noise
    cv = CrossValidator(CVConfig(n_splits=5, n_repeats=1), MetricSuite(MetricConfig(beta=1.0)))
    result = cv.run(lambda: DummyClassifier(strategy="prior"), X, y)
    assert (result.fold_metrics["beta"] == 1.0).all()
    assert result.metric_config.beta == 1.0


# ---------------------------------------------------------------------------
# Positive-class column
# ---------------------------------------------------------------------------
def test_positive_class_column_is_located_by_label_not_position():
    """`[:, 1]` silently scores the negative class when classes_ is [1, 0]."""
    flipped = SimpleNamespace(
        classes_=np.array([1, 0]),
        predict_proba=lambda X: np.array([[0.9, 0.1], [0.8, 0.2]]),
    )
    assert list(CrossValidator._positive_class_proba(flipped, None)) == [0.9, 0.8]

    standard = SimpleNamespace(
        classes_=np.array([0, 1]),
        predict_proba=lambda X: np.array([[0.9, 0.1], [0.8, 0.2]]),
    )
    assert list(CrossValidator._positive_class_proba(standard, None)) == [0.1, 0.2]


def test_missing_positive_class_is_reported():
    only_negatives = SimpleNamespace(classes_=np.array([0]), predict_proba=lambda X: np.array([[1.0], [1.0]]))
    with pytest.raises(ValueError, match="never saw the positive class"):
        CrossValidator._positive_class_proba(only_negatives, None)


def test_estimator_without_predict_proba_is_rejected():
    class NoProba(BaseEstimator):
        def fit(self, X, y):
            return self

    with pytest.raises(TypeError, match="no predict_proba"):
        CrossValidator._positive_class_proba(NoProba(), None)


# ---------------------------------------------------------------------------
# Misuse
# ---------------------------------------------------------------------------
def test_passing_an_instance_instead_of_a_factory_is_rejected(noise):
    X, y = noise
    with pytest.raises(TypeError, match="zero-argument callable"):
        _small_cv().run(DummyClassifier(strategy="prior"), X, y)


def test_factory_from_wraps_an_instance_safely(noise):
    X, y = noise
    result = _small_cv().run(factory_from(DummyClassifier(strategy="prior")), X, y)
    assert result.n_fits == 10


def test_non_binary_target_is_rejected(noise):
    X, _ = noise
    with pytest.raises(ValueError, match="y must be binary"):
        _small_cv().run(lambda: DummyClassifier(strategy="prior"), X, np.arange(400) % 3)


def test_unknown_metric_name_is_reported(noise):
    X, y = noise
    result = _small_cv().run(lambda: DummyClassifier(strategy="prior"), X, y)
    with pytest.raises(KeyError, match="no metric 'accuracy'"):
        result.mean("accuracy")


# ---------------------------------------------------------------------------
# Reporting and the falsification rule
# ---------------------------------------------------------------------------
def test_summary_reports_spread_and_excludes_fold_indices(noise):
    X, y = noise
    summary = _small_cv().run(lambda: DummyClassifier(strategy="prior"), X, y).summary()

    assert {"mean", "std", "min", "max"} == set(summary.columns)
    assert "pr_auc" in summary.index
    assert "fold" not in summary.index and "repeat" not in summary.index


def test_result_serialises_for_the_results_record(noise):
    X, y = noise
    payload = _small_cv().run(lambda: DummyClassifier(strategy="prior"), X, y).to_dict()

    assert payload["n_fits"] == 10
    assert payload["cv"]["random_state"] == 0
    assert "pr_auc" in payload["summary"]


def _fabricate(name: str, values: list[float]) -> CVResult:
    return CVResult(
        estimator_name=name,
        fold_metrics=pd.DataFrame({"pr_auc": values}),
        cv_config=CVConfig(),
        metric_config=MetricConfig(),
    )


def test_compare_flags_a_difference_larger_than_the_cv_spread():
    """Falsification: boosting beats the tree by more than the CV spread."""
    baseline = _fabricate("tree", [0.50, 0.51, 0.49, 0.50])
    challenger = _fabricate("boosting", [0.70, 0.71, 0.69, 0.70])

    verdict = compare(challenger, baseline)
    assert verdict["difference"] == pytest.approx(0.20, abs=1e-9)
    assert verdict["exceeds_cv_sd"] is True


def test_compare_does_not_flag_a_difference_inside_the_spread():
    """The hypothesis-consistent case: models converge."""
    baseline = _fabricate("tree", [0.50, 0.60, 0.40, 0.55])
    challenger = _fabricate("boosting", [0.52, 0.62, 0.42, 0.57])

    verdict = compare(challenger, baseline)
    assert verdict["difference"] == pytest.approx(0.02, abs=1e-9)
    assert verdict["exceeds_cv_sd"] is False


def test_compare_uses_the_larger_standard_deviation():
    """The conservative choice; the smaller SD would over-declare significance."""
    baseline = _fabricate("tree", [0.50, 0.50, 0.50, 0.50])  # sd 0
    challenger = _fabricate("boosting", [0.40, 0.60, 0.45, 0.65])  # sd ~0.12
    verdict = compare(challenger, baseline)
    assert verdict["cv_sd_used"] == pytest.approx(challenger.std("pr_auc"))


def test_compare_rejects_runs_with_different_fold_counts():
    with pytest.raises(ValueError, match="fold counts differ"):
        compare(_fabricate("a", [0.5, 0.5]), _fabricate("b", [0.5, 0.5, 0.5]))
