"""Cross-validation.

The single property this module exists to guarantee: **nothing fitted on a test
fold ever influences a prediction for that fold.** Leakage does not raise; it
produces ~0.99 scores that look like success. Three structural choices enforce it:

1. Estimators arrive as a **factory** -- a zero-argument callable returning a
   fresh, unfitted estimator. Passing an instance invites reuse across folds,
   where a fitted scaler or warm-started ensemble carries test-fold information
   forward. A factory makes per-fold freshness impossible to forget.
2. Every transform (scaling, resampling, tuning) belongs **inside** the object
   the factory returns, so `fit` sees train-fold rows only.
3. `run` touches the test fold exactly once per fold, through `predict_proba`.

Folds are stratified because at a 3.39% base rate an unstratified split can hand
a fold too few positives to score.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from ..config import CVConfig, MetricConfig
from .metrics import MetricSuite

EstimatorFactory = Callable[[], Any]


def factory_from(estimator: Any) -> EstimatorFactory:
    """Wrap an estimator instance as a factory, via `sklearn.clone`.

    Convenience for the common case. `clone` copies hyperparameters and drops
    all fitted state, so each fold still gets an unfitted object.
    """
    return lambda: clone(estimator)


def _take(X: Any, idx: np.ndarray) -> Any:
    """Row-select from a DataFrame or an array without changing its type."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    return np.asarray(X)[idx]


@dataclass
class CVResult:
    """Per-fold metrics plus the configuration that produced them.

    `fold_metrics` has one row per fit -- `n_splits * n_repeats` of them -- so
    the spread is inspectable rather than pre-aggregated.
    """

    estimator_name: str
    fold_metrics: pd.DataFrame
    cv_config: CVConfig
    metric_config: MetricConfig
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_fits(self) -> int:
        return len(self.fold_metrics)

    def mean(self, metric: str) -> float:
        self._require_metric(metric)
        return float(self.fold_metrics[metric].mean())

    def std(self, metric: str) -> float:
        """Sample standard deviation across folds (ddof=1).

        Descriptive spread, NOT a standard error. Repeated k-fold folds share
        rows, so they are not independent samples and this understates true
        uncertainty. Treat "difference exceeds one SD" as a decision rule, not a
        significance test.
        """
        self._require_metric(metric)
        return float(self.fold_metrics[metric].std(ddof=1))

    def summary(self) -> pd.DataFrame:
        """mean / std / min / max per metric, one row per metric."""
        numeric = self.fold_metrics.select_dtypes("number").drop(
            columns=[c for c in ("fold", "repeat") if c in self.fold_metrics], errors="ignore"
        )
        return pd.DataFrame(
            {
                "mean": numeric.mean(),
                "std": numeric.std(ddof=1),
                "min": numeric.min(),
                "max": numeric.max(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record: config in, metrics out, nothing implicit."""
        return {
            "estimator": self.estimator_name,
            "n_fits": self.n_fits,
            "cv": {
                "n_splits": self.cv_config.n_splits,
                "n_repeats": self.cv_config.n_repeats,
                "random_state": self.cv_config.random_state,
                "shuffle": self.cv_config.shuffle,
            },
            "metrics": {
                "beta": self.metric_config.beta,
                "report_threshold": self.metric_config.report_threshold,
            },
            "summary": self.summary().to_dict(orient="index"),
            **self.extra,
        }

    def _require_metric(self, metric: str) -> None:
        if metric not in self.fold_metrics.columns:
            raise KeyError(f"no metric {metric!r}; available: {sorted(self.fold_metrics.columns)}")


class CrossValidator:
    """Runs repeated stratified k-fold and scores each fold.

    Holds configuration only -- no fitted state -- so one instance can evaluate
    many estimators against an identical fold sequence. That identical sequence
    matters: comparing models scored on different splits confounds the model
    difference with the split difference.
    """

    def __init__(self, cv_config: CVConfig | None = None, metrics: MetricSuite | None = None) -> None:
        self.config = cv_config or CVConfig()
        self.metrics = metrics or MetricSuite()

    def splitter(self) -> RepeatedStratifiedKFold:
        return RepeatedStratifiedKFold(
            n_splits=self.config.n_splits,
            n_repeats=self.config.n_repeats,
            random_state=self.config.random_state,
        )

    def run(self, estimator_factory: EstimatorFactory, X: Any, y: Any, name: str | None = None) -> CVResult:
        """Fit and score one estimator across every fold.

        `estimator_factory` must be a zero-argument callable returning a fresh,
        unfitted estimator -- typically `lambda: Pipeline([...])`. Use
        `factory_from(estimator)` to wrap an existing instance.
        """
        if not callable(estimator_factory):
            raise TypeError(
                "estimator_factory must be a zero-argument callable returning a fresh "
                "estimator, e.g. `lambda: Pipeline([...])`. Passing a fitted instance "
                "would reuse it across folds and leak test-fold state. Wrap an instance "
                "with `factory_from(estimator)`."
            )

        y_arr = np.asarray(y).ravel()
        if set(np.unique(y_arr).tolist()) - {0, 1}:
            raise ValueError(f"y must be binary 0/1; found {sorted(set(y_arr.tolist()))}")

        rows: list[dict[str, float]] = []
        estimator_name = name
        splitter = self.splitter()

        for i, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y_arr)), y_arr)):
            estimator = estimator_factory()
            if estimator_name is None:
                estimator_name = type(estimator).__name__

            estimator.fit(_take(X, train_idx), y_arr[train_idx])
            proba = self._positive_class_proba(estimator, _take(X, test_idx))

            fold_metrics = self.metrics.evaluate(y_arr[test_idx], proba)
            fold_metrics["fold"] = i % self.config.n_splits
            fold_metrics["repeat"] = i // self.config.n_splits
            rows.append(fold_metrics)

        return CVResult(
            estimator_name=estimator_name or "unknown",
            fold_metrics=pd.DataFrame(rows),
            cv_config=self.config,
            metric_config=self.metrics.config,
        )

    @staticmethod
    def _positive_class_proba(estimator: Any, X_test: Any) -> np.ndarray:
        """Probability of the positive class, with the column located by label.

        Indexing `[:, 1]` blindly is wrong whenever `classes_` is not `[0, 1]`
        in that order -- it silently scores the negative class and inverts every
        metric. Looking the column up by label costs nothing and cannot invert.
        """
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(
                f"{type(estimator).__name__} has no predict_proba. This project scores "
                "Brier and PR-AUC on calibrated probabilities, so decision_function "
                "output is not a substitute."
            )
        proba = estimator.predict_proba(X_test)
        classes = list(getattr(estimator, "classes_", [0, 1]))
        if 1 not in classes:
            raise ValueError(
                f"estimator never saw the positive class (classes_={classes}); " "the fold cannot be scored"
            )
        return np.asarray(proba)[:, classes.index(1)]


def compare(challenger: CVResult, baseline: CVResult, metric: str = "pr_auc") -> dict[str, Any]:
    """The project's falsification rule, as code.

    The hypothesis is falsified if a more complex model beats a depth-limited
    tree by more than the cross-validation spread. This reports the difference
    against the larger of the two standard deviations -- the conservative choice,
    since using the smaller would declare differences significant too easily.

    Caveat that belongs in the write-up: repeated k-fold folds reuse rows, so
    these standard deviations are not standard errors and this is a decision
    rule rather than a significance test. A paired test with a Nadeau-Bengio
    style correction would be the statistically defensible version.
    """
    if challenger.n_fits != baseline.n_fits:
        raise ValueError(
            f"fold counts differ ({challenger.n_fits} vs {baseline.n_fits}); "
            "compare only runs scored on the same fold sequence"
        )

    difference = challenger.mean(metric) - baseline.mean(metric)
    spread = max(challenger.std(metric), baseline.std(metric))
    return {
        "metric": metric,
        "challenger": challenger.estimator_name,
        "baseline": baseline.estimator_name,
        "challenger_mean": challenger.mean(metric),
        "baseline_mean": baseline.mean(metric),
        "difference": difference,
        "challenger_sd": challenger.std(metric),
        "baseline_sd": baseline.std(metric),
        "cv_sd_used": spread,
        "exceeds_cv_sd": bool(abs(difference) > spread),
        "n_fits": challenger.n_fits,
    }


def compare_many(results: Sequence[CVResult], baseline: CVResult, metric: str = "pr_auc") -> pd.DataFrame:
    """`compare` across several challengers, one row each."""
    return pd.DataFrame([compare(r, baseline, metric) for r in results])
