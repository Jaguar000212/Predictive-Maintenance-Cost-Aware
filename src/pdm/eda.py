"""Exploratory analyses. No modelling, no fitting, no train/test splitting.

Every function returns (report_text, {name: DataFrame}) so results can be both
printed and persisted without recomputation.
"""

from __future__ import annotations

import pandas as pd

from . import config as C
from .loaders import cmapss_lifetimes, load_ai4i, load_cmapss

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)


def _header(title: str) -> str:
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}\n"


# ---------------------------------------------------------------------------
# (1) C-MAPSS engine lifetimes
# ---------------------------------------------------------------------------
def cmapss_lifetime_summary(subset: str = C.CMAPSS_SUBSET) -> tuple[str, dict[str, pd.DataFrame]]:
    """describe() over the engine lifetimes, kept strictly separate by split.

    The train lifetimes are the real object of interest: 100 complete,
    uncensored run-to-failure durations. The test block is reported alongside
    only to make the censoring gap explicit.
    """
    life = cmapss_lifetimes(subset)
    train = life[life["split"] == "train"]
    test = life[life["split"] == "test"]

    train_desc = train["duration"].describe().to_frame("train_lifetime_cycles")
    test_desc = pd.concat(
        [
            test["duration"].describe().rename("test_observed_cycles"),
            test["true_duration"].describe().rename("test_true_lifetime_cycles"),
        ],
        axis=1,
    )

    censor_frac = (test["true_duration"] - test["duration"]) / test["true_duration"]

    lines = [
        _header(f"(1) C-MAPSS {subset} ENGINE LIFETIMES"),
        f"train trajectories : {len(train)} (uncensored, run to failure)",
        f"test  trajectories : {len(test)} (right-censored before failure)",
        "",
        "--- describe() on the train lifetimes (the 100 complete lifetimes) ---",
        train_desc.to_string(),
        "",
        "--- test split: observed (censoring) time vs true lifetime ---",
        "NOTE: 'test_observed_cycles' is a CENSORING TIME, not a lifetime.",
        "      Using it as a lifetime biases every fitted distribution downward.",
        test_desc.to_string(),
        "",
        "--- fraction of each test engine's life that is unobserved ---",
        censor_frac.describe().to_frame("censored_fraction").to_string(),
        "",
        f"NOTE: because RUL_{subset}.txt is supplied, the test set is effectively",
        "      DE-CENSORED. It is ground truth for evaluation, not a censored",
        "      training input. Fitting on true_duration and calling the result a",
        "      censored-Weibull fit would be circular.",
    ]

    # Zero-variance sensors: constant columns yield NaN under any manual
    # (x - mean) / std standardisation and NaN correlations, with no error.
    train_raw = load_cmapss("train", subset)
    stds = train_raw[C.CMAPSS_OP_COLUMNS + C.CMAPSS_SENSOR_COLUMNS].std()
    variance = (
        stds.rename("std")
        .to_frame()
        .assign(
            n_unique=[train_raw[c].nunique() for c in stds.index],
            constant=lambda d: d["n_unique"] == 1,
        )
        .sort_values("std")
    )
    constant_cols = variance.index[variance["constant"]].tolist()

    lines += [
        "",
        "--- zero-variance / degenerate columns in the train split ---",
        f"constant columns ({len(constant_cols)}): {constant_cols}",
        "HAZARD: constant columns produce NaN under manual (x-mean)/std scaling",
        "        and NaN correlations. Drop them before any scaling step.",
        variance.head(12).to_string(),
    ]

    return "\n".join(lines), {
        "cmapss_lifetimes": life,
        "cmapss_train_lifetime_describe": train_desc,
        "cmapss_test_describe": test_desc,
        "cmapss_train_column_variance": variance,
    }


# ---------------------------------------------------------------------------
# (2) AI4I failure counts
# ---------------------------------------------------------------------------
def ai4i_failure_counts(df: pd.DataFrame) -> tuple[str, dict[str, pd.DataFrame]]:
    """Aggregate label prevalence and per-mode flag counts."""
    n = len(df)
    n_fail = int(df[C.AI4I_TARGET].sum())

    rows = []
    for mode in C.AI4I_MODE_FLAGS:
        flag = df[mode] == 1
        rows.append(
            {
                "mode": mode,
                "flag_count": int(flag.sum()),
                "flag_and_labelled_failure": int((flag & (df[C.AI4I_TARGET] == 1)).sum()),
                "flag_but_not_labelled": int((flag & (df[C.AI4I_TARGET] == 0)).sum()),
                "pct_of_all_rows": round(100 * flag.sum() / n, 3),
                "pct_of_labelled_failures": round(100 * (flag & (df[C.AI4I_TARGET] == 1)).sum() / n_fail, 2),
            }
        )
    per_mode = pd.DataFrame(rows)

    # Row-level multi-mode structure. The flag counts above sum to MORE than
    # the number of failing rows, because one row can trip several modes.
    n_flags = df[C.AI4I_MODE_FLAGS].sum(axis=1)
    overlap = (
        n_flags[df[C.AI4I_TARGET] == 1]
        .value_counts()
        .sort_index()
        .rename_axis("n_modes_tripped")
        .to_frame("n_failure_rows")
    )

    lines = [
        _header("(2) AI4I 2020 FAILURE COUNTS"),
        f"rows                       : {n}",
        f"machine_failure == 1       : {n_fail}  ({100 * n_fail / n:.2f}%)",
        f"machine_failure == 0       : {n - n_fail}",
        f"imbalance ratio (neg:pos)  : {(n - n_fail) / n_fail:.1f} : 1",
        "",
        "--- per failure mode ---",
        per_mode.to_string(index=False),
        "",
        f"sum of per-mode flag counts        : {int(df[C.AI4I_MODE_FLAGS].sum().sum())}",
        f"distinct rows with >=1 flag set    : {int((n_flags > 0).sum())}",
        "The gap between those two numbers is double-counting from multi-mode rows.",
        "",
        "--- number of modes tripped, among labelled failures ---",
        overlap.to_string(),
    ]

    return "\n".join(lines), {"ai4i_per_mode_counts": per_mode, "ai4i_mode_overlap": overlap}


# ---------------------------------------------------------------------------
# (3) Flag / label disagreements
# ---------------------------------------------------------------------------
def ai4i_label_disagreements(df: pd.DataFrame) -> tuple[str, dict[str, pd.DataFrame]]:
    """Rows where the per-mode flags contradict the aggregate label.

    The generator's stated rule is: machine_failure = 1 if and only if at least
    one mode flag is 1. The published file violates this in both directions.
    Both directions are label noise and both cap achievable performance.
    """
    any_flag = df[C.AI4I_MODE_FLAGS].sum(axis=1) > 0
    any_flag_excl_rnf = df[[m for m in C.AI4I_MODE_FLAGS if m != "RNF"]].sum(axis=1) > 0
    is_fail = df[C.AI4I_TARGET] == 1

    # Direction A: a mode fired but the aggregate label says no failure.
    flag_no_label = df[any_flag & ~is_fail]
    # Direction B: labelled a failure but no mode explains it.
    label_no_flag = df[is_fail & ~any_flag]

    per_mode_a = pd.DataFrame(
        [
            {
                "mode": m,
                "flag_set_but_label_0": int(((df[m] == 1) & ~is_fail).sum()),
            }
            for m in C.AI4I_MODE_FLAGS
        ]
    )

    n_a_deterministic = int((any_flag_excl_rnf & ~is_fail).sum())

    if n_a_deterministic == 0:
        impact_a = [
            "  Direction A: all of these are RNF-only rows -- a random failure flag",
            "  that was never propagated into the aggregate label. RNF fires",
            "  independently of every feature, so these rows are feature-wise",
            "  INDISTINGUISHABLE from ordinary negatives. A physics-learning model",
            "  will not fire on them, so they do NOT cap precision. They are inert",
            "  label noise, not a scoring trap -- provided RNF stays out of the",
            "  feature matrix, which is the locked decision.",
        ]
    else:
        impact_a = [
            f"  Direction A: {n_a_deterministic} of these rows trip a DETERMINISTIC",
            "  rule (not RNF) yet carry a negative label. A model that learns the",
            "  physics will fire on them and be scored as a FALSE POSITIVE. That is",
            "  a hard ceiling on precision, and therefore on PR-AUC, that no",
            "  algorithm can beat. Quantify it before reading any precision number.",
        ]

    lines = [
        _header("(3) AI4I MODE-FLAG vs AGGREGATE-LABEL DISAGREEMENTS"),
        "Stated generator rule: machine_failure == 1  <=>  at least one mode flag == 1",
        "",
        f"A. any flag == 1 but machine_failure == 0 : {len(flag_no_label)} rows",
        f"B. machine_failure == 1 but all flags == 0: {len(label_no_flag)} rows",
        f"   total disagreeing rows                 : {len(flag_no_label) + len(label_no_flag)}",
        "",
        "--- direction A broken down by mode ---",
        per_mode_a.to_string(index=False),
        "",
        f"Direction A excluding RNF: {n_a_deterministic} rows",
        "",
        "IMPACT -- read this before interpreting any metric:",
        *impact_a,
        "",
        f"  Direction B: {len(label_no_flag)} positives with no feature-visible cause.",
        "  These are unrecoverable and cap RECALL. They are the only genuinely",
        "  irreducible positives in the dataset.",
        "",
        "--- direction A rows ---",
        (flag_no_label.to_string(index=False) if len(flag_no_label) else "  (none)"),
        "",
        "--- direction B rows ---",
        (label_no_flag.to_string(index=False) if len(label_no_flag) else "  (none)"),
    ]

    return "\n".join(lines), {
        "ai4i_flag_set_label_0": flag_no_label,
        "ai4i_label_1_no_flag": label_no_flag,
        "ai4i_direction_a_by_mode": per_mode_a,
    }


# ---------------------------------------------------------------------------
# (4) Theoretical recall ceiling
# ---------------------------------------------------------------------------
def ai4i_recall_ceiling(df: pd.DataFrame) -> tuple[str, dict[str, pd.DataFrame]]:
    """Recall ceiling = deterministic-mode failures / all labelled failures.

    Counted over DISTINCT ROWS, never over summed flags. A row tripping both
    HDF and PWF is one failure and one recoverable failure; summing flags would
    count it twice and inflate the ceiling above 1.0.

    Three definitions are reported because "deterministic" is a judgement call:
    the strict set (HDF/PWF/OSF), the strict set plus TWF, and the residual.
    """
    is_fail = df[C.AI4I_TARGET] == 1
    pos = df[is_fail]
    n_pos = len(pos)

    det = C.AI4I_DETERMINISTIC_MODES
    semi = C.AI4I_SEMI_DETERMINISTIC_MODES
    sto = C.AI4I_STOCHASTIC_MODES

    covered_strict = pos[det].sum(axis=1) > 0
    covered_with_twf = pos[det + semi].sum(axis=1) > 0
    no_flag = pos[C.AI4I_MODE_FLAGS].sum(axis=1) == 0
    rnf_only = (pos[sto].sum(axis=1) > 0) & (pos[det + semi].sum(axis=1) == 0)
    twf_only = (pos[semi].sum(axis=1) > 0) & (pos[det].sum(axis=1) == 0)

    ceiling = pd.DataFrame(
        [
            {
                "definition": "strict (HDF|PWF|OSF)",
                "recoverable_failures": int(covered_strict.sum()),
                "total_failures": n_pos,
                "recall_ceiling": round(covered_strict.sum() / n_pos, 4),
                "irreducible_share": round(1 - covered_strict.sum() / n_pos, 4),
            },
            {
                "definition": "strict + TWF",
                "recoverable_failures": int(covered_with_twf.sum()),
                "total_failures": n_pos,
                "recall_ceiling": round(covered_with_twf.sum() / n_pos, 4),
                "irreducible_share": round(1 - covered_with_twf.sum() / n_pos, 4),
            },
        ]
    )

    residual = pd.DataFrame(
        [
            {"bucket": "TWF only (semi-deterministic boundary)", "n_failure_rows": int(twf_only.sum())},
            {"bucket": "RNF only (0.1% coin flip)", "n_failure_rows": int(rnf_only.sum())},
            {"bucket": "no mode flag at all (unexplained)", "n_failure_rows": int(no_flag.sum())},
        ]
    )
    residual["pct_of_failures"] = (100 * residual["n_failure_rows"] / n_pos).round(2)

    strict_ceiling = covered_strict.sum() / n_pos

    lines = [
        _header("(4) THEORETICAL RECALL CEILING (AI4I)"),
        f"denominator = rows with machine_failure == 1 : {n_pos}",
        "numerator   = DISTINCT such rows tripping >=1 deterministic mode",
        "              (row-level, so multi-mode rows are counted once)",
        "",
        f"deterministic modes     : {det}",
        f"semi-deterministic      : {semi}  (wear threshold drawn from U[200,240] min)",
        f"purely stochastic       : {sto}  (0.1% independent chance, feature-independent)",
        "",
        ceiling.to_string(index=False),
        "",
        "--- what the strict ceiling leaves behind ---",
        residual.to_string(index=False),
        "",
        (
            f"HYPOTHESIS CHECK: strict ceiling = {strict_ceiling:.4f}, so "
            f"{100 * (1 - strict_ceiling):.2f}% of failures are not recoverable"
        ),
        "from the deterministic rules alone (project hypothesis states ~15%).",
        "",
        "CAVEATS -- this number is an upper bound under stated assumptions, not a",
        "measurement:",
        "  * It assumes a model can recover the generator's threshold rules exactly.",
        "    Real trees approximate axis-aligned thresholds well but not perfectly,",
        "    so achieved recall will sit below this.",
        "  * OSF's threshold depends on product `type`, so `type` must be in the",
        "    feature matrix for the OSF share of this ceiling to be reachable.",
        "  * PWF depends on torque x ANGULAR velocity. Rotational speed is given in",
        "    rpm, so the power feature needs the 2*pi/60 conversion. Without it the",
        "    boundary is still learnable but is no longer a clean axis-aligned cut.",
        "  * It is a ceiling on RECALL only. Direction-A rows from section (3) cap",
        "    precision independently.",
    ]

    return "\n".join(lines), {
        "ai4i_recall_ceiling": ceiling,
        "ai4i_ceiling_residual": residual,
    }


# ---------------------------------------------------------------------------
def run_all(subset: str = C.CMAPSS_SUBSET) -> tuple[str, dict[str, pd.DataFrame]]:
    """Run every EDA section and merge their outputs."""
    ai4i = load_ai4i()

    sections = [
        cmapss_lifetime_summary(subset),
        ai4i_failure_counts(ai4i),
        ai4i_label_disagreements(ai4i),
        ai4i_recall_ceiling(ai4i),
    ]

    text = "\n".join(s[0] for s in sections)
    tables: dict[str, pd.DataFrame] = {}
    for _, tabs in sections:
        tables.update(tabs)
    return text, tables
