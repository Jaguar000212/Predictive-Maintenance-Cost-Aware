"""Tests for the Layer 4 policy simulation (docs/DECISIONS.md D6, D12)."""

from __future__ import annotations

import pandas as pd
import pytest

from pdm.config import AI4ISchema, CostConfig, DeterminismConfig, ExperimentConfig
from pdm.decision.policy_sim import (
    always_alarm_counts,
    never_alarm_counts,
    oracle_ceiling_counts,
    policy_table,
)
from pdm.loaders import load_ai4i

SCHEMA = AI4ISchema()
DETERMINISM = DeterminismConfig()
COST = CostConfig()  # D11: 10 / 1 / 0.5


def _row(flags: dict[str, int], failure: int, wear: float) -> dict[str, object]:
    base = {m: 0 for m in SCHEMA.mode_flags}
    base.update(flags)
    base[SCHEMA.target] = failure
    base[SCHEMA.tool_wear_col] = wear
    return base


# 10 hand-picked rows -- see the module docstring's arithmetic in
# test_cv_cost_curve_matches... style comments below for how each was chosen.
# 6 positives (1,2,3,4,5,9), 4 negatives (6,7,8,10).
_ROWS = [
    _row({"HDF": 1}, 1, 50),  # 1: strict catches via HDF
    _row({"PWF": 1}, 1, 30),  # 2: strict catches via PWF
    _row({"OSF": 1}, 1, 60),  # 3: strict catches via OSF
    _row({"TWF": 1}, 1, 210),  # 4: only the wear band recovers this one
    _row({"RNF": 1}, 1, 40),  # 5: irreducible under both policies
    _row({}, 0, 20),  # 6: ordinary negative, low wear
    _row({}, 0, 220),  # 7: negative, but in the wear band -> false alarm
    _row({}, 0, 205),  # 8: negative, in the wear band -> false alarm
    _row({"HDF": 1}, 1, 215),  # 9: HDF alone already catches this; wear is incidental
    _row({}, 0, 199),  # 10: just BELOW the band -- boundary check
]


@pytest.fixture
def toy_df() -> pd.DataFrame:
    return pd.DataFrame(_ROWS)


# ---------------------------------------------------------------------------
# Trivial bookends
# ---------------------------------------------------------------------------
def test_never_alarm_counts(toy_df):
    counts = never_alarm_counts(toy_df, SCHEMA)
    assert counts == {"tn": 4, "fp": 0, "fn": 6, "tp": 0}


def test_always_alarm_counts(toy_df):
    counts = always_alarm_counts(toy_df, SCHEMA)
    assert counts == {"tn": 0, "fp": 4, "fn": 0, "tp": 6}


# ---------------------------------------------------------------------------
# Oracle ceiling policies -- hand-computed on the toy frame above
# ---------------------------------------------------------------------------
def test_strict_ceiling_matches_hand_computed_counts(toy_df):
    """Alarm iff HDF|PWF|OSF. Rows 1,2,3,9 fire (all real failures, so
    precision is 1.0); rows 4 (TWF-only) and 5 (RNF-only) are missed."""
    counts = oracle_ceiling_counts(toy_df, SCHEMA, DETERMINISM, extended=False)
    assert counts == {"tn": 4, "fp": 0, "fn": 2, "tp": 4}


def test_extended_ceiling_recovers_the_wear_band_row_but_not_rnf(toy_df):
    """Adding wear >= 200 recovers row 4 (TWF, wear=210) but not row 5
    (RNF-only, wear=40 -- irreducible under any wear-based rule) -- and it
    flags rows 7 and 8 (negatives sitting in the wear band) as false alarms.
    Row 10 (wear=199) must stay unflagged: this is the >= boundary check.
    """
    counts = oracle_ceiling_counts(toy_df, SCHEMA, DETERMINISM, extended=True)
    assert counts == {"tn": 2, "fp": 2, "fn": 1, "tp": 5}


def test_wear_band_boundary_is_inclusive_at_the_start(toy_df):
    """wear_band_start_min=200 must include 200, not just values above it --
    row 8 (wear=205) and row 4 (wear=210) both count; row 10 (wear=199)
    must not. Perturbing the frame's only row exactly at 200 isolates this."""
    df = toy_df.copy()
    df.loc[len(df)] = _row({}, 0, DETERMINISM.wear_band_start_min)  # wear == 200 exactly
    counts = oracle_ceiling_counts(df, SCHEMA, DETERMINISM, extended=True)
    # One more negative row, and it must be flagged (an added false alarm),
    # not silently excluded by an off-by-one on the boundary.
    assert counts["fp"] == 3


def test_extended_ceiling_never_loses_a_strict_catch():
    """The extended policy ORs in the wear band -- it must never *un-flag*
    anything the strict policy already caught."""
    strict = oracle_ceiling_counts(_toy(), SCHEMA, DETERMINISM, extended=False)
    extended = oracle_ceiling_counts(_toy(), SCHEMA, DETERMINISM, extended=True)
    assert extended["tp"] >= strict["tp"]
    assert extended["fn"] <= strict["fn"]


def _toy() -> pd.DataFrame:
    return pd.DataFrame(_ROWS)


def test_unknown_determinism_split_is_rejected(toy_df):
    """oracle_ceiling_counts must reuse DeterminismConfig's own guard rather
    than silently dropping a mode from the arithmetic."""
    bad = DeterminismConfig(deterministic=("HDF",), semi_deterministic=("PWF", "TWF"))
    with pytest.raises(ValueError, match="no determinism classification"):
        oracle_ceiling_counts(toy_df, SCHEMA, bad, extended=False)


# ---------------------------------------------------------------------------
# policy_table
# ---------------------------------------------------------------------------
def test_policy_table_has_one_row_per_named_policy(toy_df):
    table = policy_table(toy_df, cost=COST, schema=SCHEMA, determinism=DETERMINISM)
    assert list(table["policy"]) == [
        "never_alarm",
        "strict_physics_ceiling",
        "strict_physics_plus_wear_band",
        "always_alarm",
    ]
    assert {"tn", "fp", "fn", "tp", "n_alarms", "recall", "precision", "cost_per_row"} <= set(table.columns)


def test_policy_table_cost_matches_hand_computation(toy_df):
    """never_alarm: 6 missed failures * 10 / 10 rows = 6.0.
    strict: (2*10 + 0*1 + 4*0.5) / 10 = 22/10 = 2.2.
    extended: (1*10 + 2*1 + 5*0.5) / 10 = 14.5/10 = 1.45.
    always_alarm: (0 + 4*1 + 6*0.5) / 10 = 7/10 = 0.7.
    """
    table = policy_table(toy_df, cost=COST, schema=SCHEMA, determinism=DETERMINISM).set_index("policy")
    assert table.loc["never_alarm", "cost_per_row"] == pytest.approx(6.0)
    assert table.loc["strict_physics_ceiling", "cost_per_row"] == pytest.approx(2.2)
    assert table.loc["strict_physics_plus_wear_band", "cost_per_row"] == pytest.approx(1.45)
    assert table.loc["always_alarm", "cost_per_row"] == pytest.approx(0.7)


def test_policy_table_recall_matches_the_ceiling_definitions(toy_df):
    table = policy_table(toy_df, cost=COST, schema=SCHEMA, determinism=DETERMINISM).set_index("policy")
    assert table.loc["strict_physics_ceiling", "recall"] == pytest.approx(4 / 6)
    assert table.loc["strict_physics_plus_wear_band", "recall"] == pytest.approx(5 / 6)


def test_policy_table_rejects_an_unconfigured_cost(toy_df):
    with pytest.raises(ValueError, match="pending project decision"):
        policy_table(toy_df, cost=CostConfig(missed_failure=None), schema=SCHEMA, determinism=DETERMINISM)


# ---------------------------------------------------------------------------
# Real data -- pins the actual finding (docs/DECISIONS.md D12)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    return load_ai4i(ExperimentConfig().paths.data_raw / "AI4I_2020" / "ai4i2020.csv")


def test_strict_ceiling_matches_the_documented_84_66_percent(real_df):
    """CLAUDE.md's verified data fact: 287/339 failures trip a deterministic
    mode, with zero false positives (the only flag/label disagreement rows
    are RNF-only, per eda.AI4ILabelDisagreementAnalysis)."""
    counts = oracle_ceiling_counts(real_df, SCHEMA, DETERMINISM, extended=False)
    assert counts == {"tn": 9661, "fp": 0, "fn": 52, "tp": 287}


def test_extended_ceiling_matches_the_documented_97_35_percent(real_df):
    """CLAUDE.md's verified data fact: 330/339, recovering TWF's failures by
    flagging the whole wear band -- at a real, measured false-alarm cost of
    678 rows that never actually failed.
    """
    counts = oracle_ceiling_counts(real_df, SCHEMA, DETERMINISM, extended=True)
    assert counts == {"tn": 8983, "fp": 678, "fn": 9, "tp": 330}


def test_buying_the_extended_ceiling_costs_more_than_the_strict_one(real_df):
    """The headline Layer 4 finding for D6's open question (docs/DECISIONS.md
    D12): at the D11 ratio (10 : 1 : 0.5), catching TWF's extra 43 failures
    by flagging the whole wear band costs MORE than leaving them uncaught --
    678 new false alarms outweigh 43 avoided misses. The cost curve's answer
    to "should this project target 84.66% or 97.35% recall" is: neither
    ceiling for free: 84.66%, not 97.35%, is what a cost-minimising policy
    would buy under this ratio. Pinned so a change to D11's ratio that
    flips this conclusion is caught here, not just noticed by accident in
    the write-up.
    """
    table = policy_table(real_df, cost=COST, schema=SCHEMA, determinism=DETERMINISM).set_index("policy")
    strict = table.loc["strict_physics_ceiling", "cost_per_row"]
    extended = table.loc["strict_physics_plus_wear_band", "cost_per_row"]
    assert extended > strict, (
        f"the extended (wear-band) ceiling ({extended:.4f}) no longer costs more than the "
        f"strict ceiling ({strict:.4f}) -- D12's verdict has flipped; update docs/DECISIONS.md "
        "and README.md's policy table before treating this as anything other than a real change"
    )
    assert strict == pytest.approx(0.06635)
    assert extended == pytest.approx(0.0933)


# ---------------------------------------------------------------------------
# D11's sensitivity check: is D12's verdict robust to the exact ratio, or
# does it depend on 10:1 specifically?
# ---------------------------------------------------------------------------
def test_d12_verdict_is_not_robust_across_the_plausible_ratio_range(real_df):
    """D11 flagged this as recommended before treating its ratio as final.

    The oracle counts (287/52/0 strict; 330/9/678 extended) don't change with
    the ratio, so whether extended beats strict is purely a function of
    missed_failure (false_alarm and inspection held at D11's 1 / 0.5):
    extended wins once 43 * (missed_failure - 0.5) > 678, i.e.
    missed_failure > 678 / 43 + 0.5 ~= 16.267.

    At 5:1 -- the conservative end of the 5-10x literature range D11 cites --
    strict wins by an even wider margin than at 10:1. At 20:1, on the other
    side of the breakeven point, the verdict FLIPS: extended becomes
    cheaper. D12's conclusion ("target 84.66%, not 97.35%") holds at D11's
    chosen ratio and is not a knife-edge call (10 sits well below the ~16.3
    breakeven), but it is not true across the whole range CLAUDE.md's own
    literature citation (5-10x, extended informally up to 20x) would permit
    -- exactly the caveat a sensitivity check exists to surface.
    """
    breakeven = 678 / 43 + 0.5
    assert breakeven == pytest.approx(16.267, abs=0.001)

    def _verdict(missed_failure: float) -> tuple[float, float]:
        table = policy_table(
            real_df,
            cost=CostConfig(missed_failure=missed_failure, false_alarm=1.0, inspection=0.5),
            schema=SCHEMA,
            determinism=DETERMINISM,
        ).set_index("policy")
        return (
            table.loc["strict_physics_ceiling", "cost_per_row"],
            table.loc["strict_physics_plus_wear_band", "cost_per_row"],
        )

    strict_5, extended_5 = _verdict(5.0)
    assert strict_5 < extended_5, "at 5:1 the strict ceiling must still be the cheaper policy"

    strict_20, extended_20 = _verdict(20.0)
    assert extended_20 < strict_20, (
        f"at 20:1 the extended ceiling ({extended_20:.4f}) should now be cheaper than strict "
        f"({strict_20:.4f}) -- if this fails, the breakeven arithmetic above no longer matches "
        "policy_table()'s actual counts; investigate before trusting either number"
    )
