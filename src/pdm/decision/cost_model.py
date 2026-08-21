"""Layer 4: expected cost, in the currency units fixed by `CostConfig`.

`pdm.eval.metrics` deliberately does not compute a cost figure -- it says so in
its own docstring. This module is the "inputs it has no business knowing
about": the missed-failure : false-alarm : inspection ratio decided in
`docs/DECISIONS.md` D11.

**Unit note.** CLAUDE.md's primary-metrics table names "cost per 1000h", but
AI4I is 10,000 discrete machining processes with no timestamp and no stated
process duration -- there is no justified row-to-hour conversion anywhere in
this project yet. Converting today's per-row cost into an hourly rate without
that conversion being an explicit, recorded decision would be exactly the kind
of thing that runs fine and prints a confident, wrong number. So everything
here is **cost per classification decision** (equivalently, cost per row) --
correct and unit-unambiguous on its own -- and the hours conversion is left as
a named follow-up rather than smuggled in silently. See the `horizon_hours`
field on `CostConfig` for where that conversion will eventually plug in.

Two ways to get a cost curve, and they are not interchangeable:

  cost_curve()               a single array of predictions -> one curve.
                              Correct arithmetic; worthless as a result if
                              those predictions were scored on rows the model
                              was fitted on (in-sample).
  cross_validated_cost_curve()  the honest version. Re-fits per fold like
                              CrossValidator.run() and sweeps thresholds on
                              held-out predictions only, then reports mean +/-
                              std across folds -- this is what threshold
                              optimisation and the policy table must use.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CostConfig, CVConfig, MetricConfig
from ..eval.cv import CrossValidator, _take
from ..eval.metrics import MetricSuite

# The four classification outcomes, and which CostConfig field prices each one.
# True negatives are free: nothing happened, correctly.
_COST_FIELD_BY_OUTCOME = {
    "tp": "inspection",
    "fp": "false_alarm",
    "fn": "missed_failure",
}


def expected_cost(counts: Mapping[str, float], cost: CostConfig) -> float:
    """Total expected cost over a batch of classification decisions.

    `counts` must have keys "tp", "fp", "fn" (and may have "tn", which is
    priced at 0 and ignored). This is exactly the shape returned by
    `pdm.eval.metrics.confusion_counts` and embedded in every row produced by
    `classification_metrics` / `MetricSuite.evaluate` / `.sweep()`, so this
    function composes directly with existing CV output -- no new plumbing
    needed to call it.
    """
    cost.validate()
    missing = [k for k in ("tp", "fp", "fn") if k not in counts]
    if missing:
        raise KeyError(f"counts missing required keys {missing}; got {sorted(counts)}")

    return sum(counts[outcome] * getattr(cost, field) for outcome, field in _COST_FIELD_BY_OUTCOME.items())


def cost_per_row(counts: Mapping[str, float], cost: CostConfig) -> float:
    """Expected cost divided by the number of decisions it was computed over.

    This is the number that is comparable across models and thresholds
    regardless of how many rows each was evaluated on -- `expected_cost` alone
    is not, since it scales with n.
    """
    n = counts.get("tn", 0) + counts.get("fp", 0) + counts.get("fn", 0) + counts.get("tp", 0)
    if n <= 0:
        raise ValueError(f"counts imply zero rows: {dict(counts)}")
    return expected_cost(counts, cost) / n


def cost_curve(sweep: pd.DataFrame, cost: CostConfig) -> pd.DataFrame:
    """Attach a `cost_per_row` column to a threshold sweep.

    `sweep` is the output of `MetricSuite.sweep()` -- one row per threshold,
    already carrying `tn`/`fp`/`fn`/`tp` for that operating point. This is
    where the project's central claim becomes checkable: does `cost_per_row`
    move more across this table's rows (threshold) than it does between two
    such tables for different models (algorithm)?
    """
    cost.validate()
    required = {"tn", "fp", "fn", "tp", "threshold"}
    missing = required - set(sweep.columns)
    if missing:
        raise KeyError(
            f"sweep is missing columns {sorted(missing)} -- pass the output of "
            "MetricSuite.sweep(), not an arbitrary DataFrame"
        )

    out = sweep.copy()
    out["cost_per_row"] = [
        cost_per_row(row, cost) for row in sweep[["tn", "fp", "fn", "tp"]].to_dict("records")
    ]
    return out


def optimal_operating_point(sweep_with_cost: pd.DataFrame) -> pd.Series:
    """The row (threshold and every metric at it) with the lowest `cost_per_row`.

    Ties (a flat minimum across several thresholds) resolve to the first,
    i.e. the lowest such threshold -- `idxmin` is stable in that sense. A wide
    flat region is itself a finding worth reporting, not just a tie to break.
    """
    if "cost_per_row" not in sweep_with_cost.columns:
        raise KeyError("sweep_with_cost has no 'cost_per_row' column -- pass the output of cost_curve()")
    if sweep_with_cost.empty:
        raise ValueError("sweep_with_cost is empty")
    return sweep_with_cost.loc[sweep_with_cost["cost_per_row"].idxmin()]


# ---------------------------------------------------------------------------
# Threshold optimisation on real (cross-validated) predictions
# ---------------------------------------------------------------------------
@dataclass
class CVCostResult:
    """The cost curve, evaluated honestly: every point comes from a fold this
    estimator's fit never saw.

    `fold_curves` is long-form -- one row per (repeat, fold, threshold), each
    carrying that fold's `cost_per_row` alongside every metric
    `MetricSuite.sweep()` already computes (recall, precision, f2, brier,
    tn/fp/fn/tp). Never read a single fold's curve as *the* answer -- go
    through `summary()`, which applies the same fold-to-fold spread this
    project's CV rule (`CLAUDE.md`) requires of every other metric.
    """

    estimator_name: str
    fold_curves: pd.DataFrame
    cv_config: CVConfig
    metric_config: MetricConfig
    cost_config: CostConfig

    @property
    def n_fits(self) -> int:
        return int(self.fold_curves[["repeat", "fold"]].drop_duplicates().shape[0])

    def summary(self) -> pd.DataFrame:
        """Mean and std of `cost_per_row` at each threshold, across every fold.

        This is the curve to report or plot. A single fold's curve is one
        noisy draw -- exactly the reasoning that ruled out a single 80/20
        split for every other metric in this project.
        """
        grouped = self.fold_curves.groupby("threshold")["cost_per_row"]
        return pd.DataFrame(
            {"cost_per_row_mean": grouped.mean(), "cost_per_row_std": grouped.std(ddof=1)}
        ).reset_index()

    @property
    def optimal(self) -> pd.Series:
        """The threshold with the lowest mean cost per row, across folds."""
        summary = self.summary()
        return summary.loc[summary["cost_per_row_mean"].idxmin()]


def cross_validated_cost_curve(
    estimator_factory,
    X,
    y,
    cost: CostConfig | None = None,
    cv_config: CVConfig | None = None,
    metric_config: MetricConfig | None = None,
    name: str | None = None,
) -> CVCostResult:
    """The cost-optimal threshold, found honestly.

    The in-sample check documented in README.md's Layer 4 section (fit on
    every row, score the same rows) is fine for confirming the arithmetic and
    worthless as an actual result -- an unregularised model can memorise its
    way to a near-zero apparent cost. This is the real version: it re-splits
    `X`/`y` exactly the way `CrossValidator.run()` does (same `cv_config` ->
    the same fold sequence, which is what makes a later model-vs-model
    comparison fair rather than confounded by different splits) and computes
    the full threshold sweep on each fold's held-out predictions only, never
    on rows that fold's fit saw.

    `estimator_factory` has the same contract as `CrossValidator.run()`: a
    zero-argument callable returning a fresh, unfitted estimator.
    """
    cost = cost or CostConfig()
    cost.validate()
    cv_config = cv_config or CVConfig()
    metrics = MetricSuite(metric_config)

    if not callable(estimator_factory):
        raise TypeError(
            "estimator_factory must be a zero-argument callable returning a fresh "
            "estimator -- see CrossValidator.run()'s docstring for why."
        )

    y_arr = np.asarray(y).ravel()
    if set(np.unique(y_arr).tolist()) - {0, 1}:
        raise ValueError(f"y must be binary 0/1; found {sorted(set(y_arr.tolist()))}")

    splitter = CrossValidator(cv_config).splitter()
    estimator_name = name
    fold_frames: list[pd.DataFrame] = []

    for i, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y_arr)), y_arr)):
        estimator = estimator_factory()
        if estimator_name is None:
            estimator_name = type(estimator).__name__

        estimator.fit(_take(X, train_idx), y_arr[train_idx])
        # Reused, not re-derived: which predict_proba column is the positive
        # class is exactly the kind of thing that silently drifts wrong if
        # written twice (see CrossValidator's own docstring on this point).
        proba = CrossValidator._positive_class_proba(estimator, _take(X, test_idx))

        sweep = metrics.sweep(y_arr[test_idx], proba)
        curve = cost_curve(sweep, cost)
        curve["fold"] = i % cv_config.n_splits
        curve["repeat"] = i // cv_config.n_splits
        fold_frames.append(curve)

    return CVCostResult(
        estimator_name=estimator_name or "unknown",
        fold_curves=pd.concat(fold_frames, ignore_index=True),
        cv_config=cv_config,
        metric_config=metrics.config,
        cost_config=cost,
    )
