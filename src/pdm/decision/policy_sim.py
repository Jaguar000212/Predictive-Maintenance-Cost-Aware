"""Layer 4: policy simulation -- the cost of the two named policies D6 left
open ("report both ceilings and both policies... and let the cost curve
decide"), plus the trivial bookends any policy has to beat to be worth
building.

Two families of policy here, and they answer different questions:

  trivial policies       never_alarm / always_alarm. What doing nothing, or
                         stopping every machine, costs -- the two bookends
                         any real policy must beat to be worth building.
  oracle ceiling         "if every deterministic-boundary failure (HDF, PWF,
  policies               OSF) were recognised perfectly, what would it cost?"
                         Computed directly from the AI4I mode flags -- ground
                         truth, not a prediction -- so this is an upper bound,
                         not something deployable: the flags are recorded AT
                         failure time, not available beforehand. Exactly the
                         same caveat `eda.AI4IRecallCeilingAnalysis` already
                         states for the recall-only version of this number.

The "extended" oracle policy adds the one piece of this that IS deployable:
TWF's failure threshold is drawn per-tool from U[200, 240] min and is not
observable, so there is no rule that flags only the tools about to fail --
recovering those failures means flagging every row with tool_wear at or
above `DeterminismConfig.wear_band_start_min`, whether or not that specific
tool ever fails. That is D6's "fixed precision cost", made concrete: this
module counts exactly how many of AI4I's 10,000 rows that band sweeps up,
and what they cost against the D11 ratio.

How a real, trained model (`cross_validated_cost_curve` in `cost_model.py`)
compares to these ceilings is the headroom question -- assembled alongside
this module's output in `README.md`, not computed here: this module only
knows about the oracle policies and the raw AI4I frame.
"""

from __future__ import annotations

import pandas as pd

from ..config import AI4ISchema, CostConfig, DeterminismConfig
from .cost_model import cost_per_row


def _counts_from_alarm(alarm: pd.Series, is_failure: pd.Series) -> dict[str, int]:
    return {
        "tn": int((~alarm & ~is_failure).sum()),
        "fp": int((alarm & ~is_failure).sum()),
        "fn": int((~alarm & is_failure).sum()),
        "tp": int((alarm & is_failure).sum()),
    }


def never_alarm_counts(df: pd.DataFrame, schema: AI4ISchema | None = None) -> dict[str, int]:
    """The status-quo bookend: no monitoring at all."""
    schema = schema or AI4ISchema()
    is_failure = df[schema.target] == 1
    return _counts_from_alarm(pd.Series(False, index=df.index), is_failure)


def always_alarm_counts(df: pd.DataFrame, schema: AI4ISchema | None = None) -> dict[str, int]:
    """The other bookend: stop and inspect every machine, every time."""
    schema = schema or AI4ISchema()
    is_failure = df[schema.target] == 1
    return _counts_from_alarm(pd.Series(True, index=df.index), is_failure)


def oracle_ceiling_counts(
    df: pd.DataFrame,
    schema: AI4ISchema | None = None,
    determinism: DeterminismConfig | None = None,
    *,
    extended: bool,
) -> dict[str, int]:
    """tn/fp/fn/tp for one of the two named oracle ceiling policies.

    `extended=False` ("strict physics"): alarm iff >=1 deterministic mode
    flag (HDF/PWF/OSF) is set. Ground truth, not a prediction -- see the
    module docstring's caveat.

    `extended=True` ("strict physics + high-wear band"): also alarms on
    every row with `tool_wear_min >= wear_band_start_min`, which is the one
    part of this policy that IS a real, deployable rule (a feature value
    available before any failure, not a post-hoc flag). This is what buys
    the 12.7-point recall gap documented in `docs/DECISIONS.md` D6 -- and
    what makes it cost something: most rows swept up by the wear band did
    not actually fail.
    """
    schema = schema or AI4ISchema()
    determinism = determinism or DeterminismConfig()
    determinism.validate_against(schema.mode_flags)

    is_failure = df[schema.target] == 1
    alarm = df[list(determinism.deterministic)].sum(axis=1) > 0
    if extended:
        alarm = alarm | (df[schema.tool_wear_col] >= determinism.wear_band_start_min)

    return _counts_from_alarm(alarm, is_failure)


def policy_table(
    df: pd.DataFrame,
    cost: CostConfig | None = None,
    schema: AI4ISchema | None = None,
    determinism: DeterminismConfig | None = None,
) -> pd.DataFrame:
    """One row per policy: counts, recall, precision, and cost per row.

    Four rows -- the two trivial bookends and the two oracle ceiling
    policies -- so a report can read the table and see immediately how much
    of the missed-failure cost the strict boundary already buys, and whether
    also buying the wear band is worth its false-alarm cost under the D11
    ratio. A real model's honest CV cost (`cost_model.cross_validated_cost_curve`)
    belongs alongside this table, not inside it -- this function only knows
    the oracle policies and the raw frame.
    """
    cost = cost or CostConfig()
    cost.validate()
    schema = schema or AI4ISchema()
    determinism = determinism or DeterminismConfig()

    rows = [
        ("never_alarm", never_alarm_counts(df, schema)),
        ("strict_physics_ceiling", oracle_ceiling_counts(df, schema, determinism, extended=False)),
        (
            "strict_physics_plus_wear_band",
            oracle_ceiling_counts(df, schema, determinism, extended=True),
        ),
        ("always_alarm", always_alarm_counts(df, schema)),
    ]

    records = []
    for name, counts in rows:
        n_alarms = counts["fp"] + counts["tp"]
        n_positive = counts["tp"] + counts["fn"]
        records.append(
            {
                "policy": name,
                **counts,
                "n_alarms": n_alarms,
                "recall": counts["tp"] / n_positive if n_positive else float("nan"),
                "precision": counts["tp"] / n_alarms if n_alarms else 0.0,
                "cost_per_row": cost_per_row(counts, cost),
            }
        )
    return pd.DataFrame.from_records(records)
