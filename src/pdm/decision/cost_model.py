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
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ..config import CostConfig

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
