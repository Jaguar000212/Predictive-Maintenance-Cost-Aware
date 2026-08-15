"""Exploratory analyses. No modelling, no fitting, no train/test splitting.

Each analysis is a class implementing `Analysis`: it receives its data through
the constructor and returns an `AnalysisResult`. Nothing here loads a file --
wiring loaders to analyses happens in the composition root (`scripts/run_eda.py`),
which is what lets a test run any analysis against a fixture frame.

`EDAReport` composes analyses in order. Adding a fifth section means writing one
class and appending it to a list; no existing analysis changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import AI4ISchema, CMAPSSSchema, DeterminismConfig, EDAConfig

_RULE = "=" * 78


def _header(title: str) -> str:
    return f"\n{_RULE}\n{title}\n{_RULE}\n"


@dataclass
class AnalysisResult:
    """One analysis's rendered text plus the tables behind it."""

    title: str
    text: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


class Analysis(ABC):
    """One section of the EDA report."""

    title: str = "untitled"

    @abstractmethod
    def run(self) -> AnalysisResult:
        """Compute the section and render it."""


# ---------------------------------------------------------------------------
# (1) C-MAPSS engine lifetimes
# ---------------------------------------------------------------------------
class CMAPSSLifetimeAnalysis(Analysis):
    """describe() over engine lifetimes, kept strictly separate by split.

    Train lifetimes are the real object of interest: complete, uncensored
    run-to-failure durations. The test block is reported alongside only to make
    the censoring gap explicit.
    """

    title = "C-MAPSS engine lifetimes"

    def __init__(
        self,
        lifetimes: pd.DataFrame,
        train_cycles: pd.DataFrame,
        schema: CMAPSSSchema | None = None,
        config: EDAConfig | None = None,
    ) -> None:
        self.lifetimes = lifetimes
        self.train_cycles = train_cycles
        self.schema = schema or CMAPSSSchema()
        self.config = config or EDAConfig()

    def _degenerate_columns(self) -> pd.DataFrame:
        """Flag constant columns.

        Defaults to `nunique`, not `std == 0`: two FD001 columns are constant
        only to floating-point tolerance (std 3e-18 and 5e-15), so an equality
        test on the standard deviation misses them and they survive into a
        scaler as NaN.
        """
        cols = list(self.schema.op_columns + self.schema.sensor_columns)
        stds = self.train_cycles[cols].std()
        n_unique = [self.train_cycles[c].nunique() for c in stds.index]

        table = stds.rename("std").to_frame().assign(n_unique=n_unique)
        if self.config.constant_detection == "nunique":
            table["constant"] = table["n_unique"] == 1
        else:
            table["constant"] = table["std"].abs() <= self.config.constant_std_tolerance
        return table.sort_values("std")

    def run(self) -> AnalysisResult:
        life = self.lifetimes
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

        variance = self._degenerate_columns()
        constant_cols = variance.index[variance["constant"]].tolist()
        subset = self.schema.subset

        lines = [
            _header(f"(1) C-MAPSS {subset} ENGINE LIFETIMES"),
            f"train trajectories : {len(train)} (uncensored, run to failure)",
            f"test  trajectories : {len(test)} (right-censored before failure)",
            "",
            f"--- describe() on the train lifetimes (the {len(train)} complete lifetimes) ---",
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
            "",
            "--- zero-variance / degenerate columns in the train split ---",
            f"constant columns ({len(constant_cols)}): {constant_cols}",
            "HAZARD: constant columns produce NaN under manual (x-mean)/std scaling",
            "        and NaN correlations. Drop them before any scaling step.",
            variance.head(self.config.variance_preview_rows).to_string(),
        ]

        return AnalysisResult(
            title=self.title,
            text="\n".join(lines),
            tables={
                "cmapss_lifetimes": life,
                "cmapss_train_lifetime_describe": train_desc,
                "cmapss_test_describe": test_desc,
                "cmapss_train_column_variance": variance,
            },
        )


# ---------------------------------------------------------------------------
# AI4I analyses
# ---------------------------------------------------------------------------
class AI4IAnalysis(Analysis):
    """Base for analyses over the AI4I frame. Holds the frame and its schema."""

    def __init__(self, df: pd.DataFrame, schema: AI4ISchema | None = None) -> None:
        self.df = df
        self.schema = schema or AI4ISchema()

    @property
    def modes(self) -> list[str]:
        return list(self.schema.mode_flags)

    @property
    def is_failure(self) -> pd.Series:
        return self.df[self.schema.target] == 1


class AI4IFailureCountAnalysis(AI4IAnalysis):
    """Aggregate label prevalence and per-mode flag counts."""

    title = "AI4I failure counts"

    def run(self) -> AnalysisResult:
        df, target = self.df, self.schema.target
        n = len(df)
        n_fail = int(df[target].sum())
        is_fail = self.is_failure

        per_mode = pd.DataFrame(
            [
                {
                    "mode": mode,
                    "flag_count": int((df[mode] == 1).sum()),
                    "flag_and_labelled_failure": int(((df[mode] == 1) & is_fail).sum()),
                    "flag_but_not_labelled": int(((df[mode] == 1) & ~is_fail).sum()),
                    "pct_of_all_rows": round(100 * (df[mode] == 1).sum() / n, 3),
                    "pct_of_labelled_failures": round(100 * ((df[mode] == 1) & is_fail).sum() / n_fail, 2),
                }
                for mode in self.modes
            ]
        )

        # Flag counts sum to more than the number of failing rows, because one
        # row can trip several modes.
        n_flags = df[self.modes].sum(axis=1)
        overlap = (
            n_flags[is_fail]
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
            f"sum of per-mode flag counts        : {int(df[self.modes].sum().sum())}",
            f"distinct rows with >=1 flag set    : {int((n_flags > 0).sum())}",
            "The gap between those two numbers is double-counting from multi-mode rows.",
            "",
            "--- number of modes tripped, among labelled failures ---",
            overlap.to_string(),
        ]

        return AnalysisResult(
            title=self.title,
            text="\n".join(lines),
            tables={"ai4i_per_mode_counts": per_mode, "ai4i_mode_overlap": overlap},
        )


class AI4ILabelDisagreementAnalysis(AI4IAnalysis):
    """Rows where the per-mode flags contradict the aggregate label.

    The generator's stated rule is: machine_failure = 1 if and only if at least
    one mode flag is 1. The published file violates this in both directions.
    """

    title = "AI4I flag/label disagreements"

    def run(self) -> AnalysisResult:
        df = self.df
        modes = self.modes
        is_fail = self.is_failure
        any_flag = df[modes].sum(axis=1) > 0
        non_random = [m for m in modes if m != "RNF"]
        any_flag_excl_rnf = df[non_random].sum(axis=1) > 0

        flag_no_label = df[any_flag & ~is_fail]  # direction A
        label_no_flag = df[is_fail & ~any_flag]  # direction B

        per_mode_a = pd.DataFrame(
            [{"mode": m, "flag_set_but_label_0": int(((df[m] == 1) & ~is_fail).sum())} for m in modes]
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

        return AnalysisResult(
            title=self.title,
            text="\n".join(lines),
            tables={
                "ai4i_flag_set_label_0": flag_no_label,
                "ai4i_label_1_no_flag": label_no_flag,
                "ai4i_direction_a_by_mode": per_mode_a,
            },
        )


class AI4IRecallCeilingAnalysis(AI4IAnalysis):
    """Recall ceiling = deterministic-mode failures / all labelled failures.

    Counted over DISTINCT ROWS, never over summed flags. A row tripping both HDF
    and PWF is one failure and one recoverable failure; summing flags counts it
    twice and can push the ceiling above 1.0.

    Which modes count as deterministic comes from `DeterminismConfig`, because
    it is a judgement call that moves the headline number by 12.7 points.
    """

    title = "AI4I recall ceiling"

    def __init__(
        self,
        df: pd.DataFrame,
        schema: AI4ISchema | None = None,
        determinism: DeterminismConfig | None = None,
    ) -> None:
        super().__init__(df, schema)
        self.determinism = determinism or DeterminismConfig()
        self.determinism.validate_against(self.schema.mode_flags)

    def run(self) -> AnalysisResult:
        d = self.determinism
        det, semi, sto = list(d.deterministic), list(d.semi_deterministic), list(d.stochastic)

        pos = self.df[self.is_failure]
        n_pos = len(pos)
        if n_pos == 0:
            raise ValueError("no labelled failures; the recall ceiling is undefined")

        covered_strict = pos[det].sum(axis=1) > 0
        covered_extended = pos[list(d.recoverable_extended)].sum(axis=1) > 0
        no_flag = pos[self.modes].sum(axis=1) == 0
        semi_only = (pos[semi].sum(axis=1) > 0) & (pos[det].sum(axis=1) == 0)
        sto_only = (pos[sto].sum(axis=1) > 0) & (pos[det + semi].sum(axis=1) == 0)

        ceiling = pd.DataFrame(
            [
                self._ceiling_row(f"strict ({'|'.join(det)})", covered_strict, n_pos),
                self._ceiling_row(f"strict + {'+'.join(semi)}", covered_extended, n_pos),
            ]
        )

        residual = pd.DataFrame(
            [
                {
                    "bucket": f"{'/'.join(semi)} only (semi-deterministic boundary)",
                    "n_failure_rows": int(semi_only.sum()),
                },
                {"bucket": f"{'/'.join(sto)} only (0.1% coin flip)", "n_failure_rows": int(sto_only.sum())},
                {"bucket": "no mode flag at all (unexplained)", "n_failure_rows": int(no_flag.sum())},
            ]
        )
        residual["pct_of_failures"] = (100 * residual["n_failure_rows"] / n_pos).round(2)

        strict_ceiling = covered_strict.sum() / n_pos

        lines = [
            _header("(4) THEORETICAL RECALL CEILING (AI4I)"),
            f"denominator = rows with {self.schema.target} == 1 : {n_pos}",
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

        return AnalysisResult(
            title=self.title,
            text="\n".join(lines),
            tables={"ai4i_recall_ceiling": ceiling, "ai4i_ceiling_residual": residual},
        )

    @staticmethod
    def _ceiling_row(label: str, covered: pd.Series, n_pos: int) -> dict[str, object]:
        n_covered = int(covered.sum())
        return {
            "definition": label,
            "recoverable_failures": n_covered,
            "total_failures": n_pos,
            "recall_ceiling": round(n_covered / n_pos, 4),
            "irreducible_share": round(1 - n_covered / n_pos, 4),
        }


# ---------------------------------------------------------------------------
# Report composition
# ---------------------------------------------------------------------------
class EDAReport:
    """Runs a sequence of analyses and renders them as one report.

    Holds no knowledge of which analyses exist -- they are injected. Display
    options are applied via a context manager rather than mutating pandas
    globals at import time, so importing this module changes nothing elsewhere.
    """

    def __init__(self, analyses: list[Analysis], display_width: int = 120, max_columns: int = 40) -> None:
        self.analyses = analyses
        self.display_width = display_width
        self.max_columns = max_columns

    def run(self) -> tuple[str, dict[str, pd.DataFrame]]:
        with pd.option_context("display.width", self.display_width, "display.max_columns", self.max_columns):
            results = [analysis.run() for analysis in self.analyses]

        text = "\n".join(r.text for r in results)
        tables: dict[str, pd.DataFrame] = {}
        for result in results:
            tables.update(result.tables)
        return text, tables

    def write(self, out_dir: Path | str) -> Path:
        text, tables = self.run()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        report_path = out / "eda_report.txt"
        report_path.write_text(text, encoding="utf-8")
        for name, table in tables.items():
            table.to_csv(out / f"{name}.csv", index=True)
        return report_path
