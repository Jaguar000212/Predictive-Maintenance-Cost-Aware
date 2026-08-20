"""Configuration objects.

Every setting that can change a result lives here as a field on a frozen
dataclass, never as a literal buried in code. Two properties matter:

* **Frozen.** Configuration mutated halfway through a run produces results that
  cannot be reproduced and raises nothing. `frozen=True` makes that a
  `FrozenInstanceError` at the point of the mistake.
* **Serialisable.** `ExperimentConfig.to_dict()` feeds the results JSON, so every
  recorded number carries the exact configuration that produced it. That is what
  makes the "no figure from an unrecorded run" rule enforceable.

Defaults reproduce the committed EDA results. Override by constructing a config
and passing it in -- do not edit the defaults to run a variant, or the recorded
history stops matching the code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathConfig:
    """Filesystem layout. Injectable so tests can point at fixtures."""

    repo_root: Path
    data_raw: Path
    reports_dir: Path
    results_dir: Path

    @classmethod
    def default(cls) -> PathConfig:
        # config.py -> pdm -> src -> repo root
        root = Path(__file__).resolve().parents[2]
        return cls(
            repo_root=root,
            data_raw=root / "data" / "raw",
            reports_dir=root / "reports",
            results_dir=root / "results",
        )

    @property
    def ai4i_csv(self) -> Path:
        return self.data_raw / "AI4I_2020" / "ai4i2020.csv"

    @property
    def cmapss_dir(self) -> Path:
        return self.data_raw / "CMAPSS_2008"

    @property
    def eda_dir(self) -> Path:
        return self.reports_dir / "eda"


# ---------------------------------------------------------------------------
# Dataset schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AI4ISchema:
    """Column names and structural expectations for AI4I 2020.

    The published CSV carries a UTF-8 BOM. `encoding="utf-8-sig"` is explicit
    rather than strictly required: pandas 2.3 strips the BOM under both the C
    and python engines whatever encoding is declared (measured, not assumed).
    It matters for anything reading the file outside pandas -- `open()`, the
    `csv` module, or another tool -- where plain utf-8 names the first column
    "\\ufeffUDI" and every lookup by "UDI" fails. Declaring it here keeps that
    true regardless of who reads the file next.
    """

    encoding: str = "utf-8-sig"
    expected_rows: int | None = 10_000
    target: str = "machine_failure"

    # (raw header, internal name). A tuple of pairs rather than a dict so the
    # dataclass stays genuinely immutable and hashable.
    column_pairs: tuple[tuple[str, str], ...] = (
        ("UDI", "udi"),
        ("Product ID", "product_id"),
        ("Type", "type"),
        ("Air temperature [K]", "air_temp_k"),
        ("Process temperature [K]", "process_temp_k"),
        ("Rotational speed [rpm]", "rot_speed_rpm"),
        ("Torque [Nm]", "torque_nm"),
        ("Tool wear [min]", "tool_wear_min"),
        ("Machine failure", "machine_failure"),
        ("TWF", "TWF"),
        ("HDF", "HDF"),
        ("PWF", "PWF"),
        ("OSF", "OSF"),
        ("RNF", "RNF"),
    )

    mode_flags: tuple[str, ...] = ("TWF", "HDF", "PWF", "OSF", "RNF")
    numeric_features: tuple[str, ...] = (
        "air_temp_k",
        "process_temp_k",
        "rot_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    )
    categorical_features: tuple[str, ...] = ("type",)
    identifier_columns: tuple[str, ...] = ("udi", "product_id")

    # Named single-column references for the physics feature formulas
    # (`features/physics.py`). Kept as explicit fields rather than positional
    # indices into `numeric_features`, so renaming a raw column cannot silently
    # point a formula at the wrong sensor.
    air_temp_col: str = "air_temp_k"
    process_temp_col: str = "process_temp_k"
    rot_speed_col: str = "rot_speed_rpm"
    torque_col: str = "torque_nm"
    tool_wear_col: str = "tool_wear_min"

    # Columns `PhysicsFeatures` adds. Recorded here, not just in the
    # transformer, so a results JSON shows which engineered columns a run used.
    engineered_features: tuple[str, ...] = ("temp_diff", "power_w", "wear_strain")

    @property
    def column_map(self) -> dict[str, str]:
        return dict(self.column_pairs)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(internal for _, internal in self.column_pairs)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Columns permitted in a feature matrix, before engineering.

        `type` is included and must stay: the OSF threshold is tier-dependent
        (L/M/H), so the OSF share of the recall ceiling is unreachable without it.
        """
        return self.numeric_features + self.categorical_features

    @property
    def excluded_from_features(self) -> tuple[str, ...]:
        """Never admissible as features: identifiers, the target, and mode flags.

        The mode flags are deterministic functions of the label's generator --
        leaving them in is leakage that scores ~0.99 and raises nothing.
        """
        return self.identifier_columns + (self.target,) + self.mode_flags


_CMAPSS_UNIT_COUNTS: dict[str, tuple[int, int]] = {
    "FD001": (100, 100),
    "FD002": (260, 259),
    "FD003": (100, 100),
    "FD004": (248, 249),
}


@dataclass(frozen=True)
class CMAPSSSchema:
    """Column layout and unit counts for one C-MAPSS subset.

    The bundled NASA readme describes the 26 columns as "sensor measurement
    1 ... 26". That is a typo in the original distribution: columns 1-5 are
    unit, cycle, and three operational settings, leaving 21 sensors in columns
    6-26. Taking the readme literally produces 26 sensor names for 21 sensors
    and silently mislabels every one of them.
    """

    subset: str = "FD001"
    n_op_settings: int = 3
    n_sensors: int = 21
    expected_train_units: int | None = 100
    expected_test_units: int | None = 100

    @classmethod
    def for_subset(cls, subset: str, **overrides: Any) -> CMAPSSSchema:
        """Build the schema for any subset, with the right unit counts."""
        if subset not in _CMAPSS_UNIT_COUNTS:
            raise ValueError(
                f"unknown C-MAPSS subset {subset!r}; expected one of {sorted(_CMAPSS_UNIT_COUNTS)}"
            )
        train, test = _CMAPSS_UNIT_COUNTS[subset]
        return cls(subset=subset, expected_train_units=train, expected_test_units=test, **overrides)

    def expected_units(self, split: str) -> int | None:
        if split == "train":
            return self.expected_train_units
        if split == "test":
            return self.expected_test_units
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    @property
    def id_columns(self) -> tuple[str, ...]:
        return ("unit", "cycle")

    @property
    def op_columns(self) -> tuple[str, ...]:
        return tuple(f"op_setting_{i}" for i in range(1, self.n_op_settings + 1))

    @property
    def sensor_columns(self) -> tuple[str, ...]:
        return tuple(f"sensor_{i}" for i in range(1, self.n_sensors + 1))

    @property
    def columns(self) -> tuple[str, ...]:
        return self.id_columns + self.op_columns + self.sensor_columns

    @property
    def n_columns(self) -> int:
        return len(self.columns)


# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterminismConfig:
    """Which AI4I failure modes count as recoverable from features.

    This is a MODELLING ASSUMPTION and it sets the headline recall ceiling, so
    it is configuration rather than a constant. From the generator description
    (Matzka 2020):

      HDF/PWF/OSF  exact threshold rules over observed features => deterministic
      TWF          tool retired at a wear threshold drawn from U[200, 240] min;
                   the draw is unobservable, so the boundary is only partly
                   recoverable => semi-deterministic
      RNF          independent 0.1% chance, unrelated to every feature
                   => irreducible

    Moving TWF between groups moves the ceiling from 84.66% to 97.35%. Change it
    deliberately and record it; never to make a result look better.
    """

    deterministic: tuple[str, ...] = ("HDF", "PWF", "OSF")
    semi_deterministic: tuple[str, ...] = ("TWF",)
    stochastic: tuple[str, ...] = ("RNF",)

    def validate_against(self, mode_flags: tuple[str, ...]) -> None:
        """Every mode must be classified exactly once.

        An unclassified mode would silently vanish from the ceiling arithmetic;
        a duplicated one would be double-counted.
        """
        assigned = self.deterministic + self.semi_deterministic + self.stochastic
        duplicates = {m for m in assigned if assigned.count(m) > 1}
        if duplicates:
            raise ValueError(f"failure modes classified more than once: {sorted(duplicates)}")
        missing = set(mode_flags) - set(assigned)
        if missing:
            raise ValueError(f"failure modes with no determinism classification: {sorted(missing)}")
        unknown = set(assigned) - set(mode_flags)
        if unknown:
            raise ValueError(f"determinism config references unknown modes: {sorted(unknown)}")

    @property
    def recoverable_strict(self) -> tuple[str, ...]:
        return self.deterministic

    @property
    def recoverable_extended(self) -> tuple[str, ...]:
        return self.deterministic + self.semi_deterministic


@dataclass(frozen=True)
class EDAConfig:
    """Settings for the exploratory analyses.

    `constant_detection` defaults to "nunique" for a measured reason: two FD001
    columns are constant only to floating-point tolerance (std 3e-18 and 5e-15),
    so an `std == 0` test misses them and they survive into a scaler as NaN.
    """

    constant_detection: Literal["nunique", "std_tolerance"] = "nunique"
    constant_std_tolerance: float = 1e-12
    variance_preview_rows: int = 12


@dataclass(frozen=True)
class WeibullMLEConfig:
    """Optimiser settings for the censored Weibull MLE (Layer 1).

    The likelihood is written by hand in `models/mle/censored_weibull.py`;
    these settings control how `scipy.optimize` searches it. They belong here
    rather than as literals in that module because a tight iteration cap can
    return an unconverged fit that still looks like a plausible answer --
    `CensoredWeibullMLE.fit` raises rather than accepting one, but the
    tolerance that decides "converged" is itself a setting, not a given.
    """

    optimizer_method: str = "L-BFGS-B"
    max_iterations: int = 500
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")


@dataclass(frozen=True)
class BayesianConfig:
    """Settings for Layer 2 (Gaussian Naive Bayes, Bayesian logistic regression).

    `use_class_weight_balanced` deliberately defaults to False -- unlike Layer
    3's tree models, which follow CLAUDE.md's locked `class_weight='balanced'`
    rule. Layer 2 exists to produce CALIBRATED probabilities (Brier score is a
    locked primary metric specifically because it catches miscalibration).
    Reweighting toward a 50/50 prior would push every probability away from
    the true ~3.4% base rate by the same mechanism CLAUDE.md rejects SMOTE
    for, just via the loss function instead of resampling. If this is ever
    flipped to True, note it in docs/DECISIONS.md -- it changes what a
    reported probability means.

    `logreg_C` is not just a regularisation knob here: for the Bayesian
    logistic regression's Laplace approximation, `C` *is* the prior variance
    on each weight (see `models/bayes/bayes_logreg.py`), so changing it changes
    the posterior, not only the MAP point.
    """

    use_class_weight_balanced: bool = False
    logreg_C: float = 1.0
    logreg_max_iter: int = 1000
    gnb_var_smoothing: float = 1e-9
    cnb_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.logreg_C <= 0:
            raise ValueError(f"logreg_C must be positive, got {self.logreg_C}")
        if self.logreg_max_iter < 1:
            raise ValueError(f"logreg_max_iter must be >= 1, got {self.logreg_max_iter}")
        if self.gnb_var_smoothing <= 0:
            raise ValueError(f"gnb_var_smoothing must be positive, got {self.gnb_var_smoothing}")
        if self.cnb_alpha <= 0:
            raise ValueError(f"cnb_alpha must be positive, got {self.cnb_alpha}")


@dataclass(frozen=True)
class CalibrationConfig:
    """Settings for reliability curves and the Brier decomposition.

    `strategy="quantile"` (equal-count bins) is the default rather than
    equal-width bins, and deliberately so: at AI4I's 3.39% base rate, most
    predicted probabilities cluster near zero, so equal-width bins leave
    nearly every point in the first one or two buckets and the rest empty --
    a reliability curve with almost no information in it. Equal-count bins
    keep every bucket populated at the cost of the bucket edges being
    data-dependent rather than round numbers.
    """

    n_bins: int = 10
    strategy: Literal["uniform", "quantile"] = "quantile"

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")


@dataclass(frozen=True)
class TreeConfig:
    """Layer 3: decision tree and Random Forest settings.

    `class_weight='balanced'` is applied directly in `models/trees/*.py`, not
    exposed as a toggle here -- unlike Layer 2's `BayesianConfig`, which had a
    real, stated reason to deviate from it (calibration). Trees are exactly
    the discriminative classifiers CLAUDE.md's locked imbalance rule targets;
    there is no carve-out for them.

    `depth_limited_max_depth` is the single most important number in this
    file. CLAUDE.md's falsification test is "does boosting beat this tree by
    more than the CV spread" -- this tree IS the baseline that test is run
    against. 4 is a deliberate, fixed choice, not a tuned one: tuning it
    would need nested CV (search inside each outer fold), which would defeat
    the point of "depth-limited" here -- a fixed, simple baseline that added
    complexity is measured against, not itself optimised. Chosen because each
    physics feature turns roughly one failure mode into one axis-aligned cut
    (HDF: 1 split on temp_diff; PWF: 2 splits bounding power_w; OSF: 2-3
    splits on wear_strain conditioned on type), so a small tree only needs
    enough depth to combine a handful of OR'd rules, not memorise rows.

    `forest_max_depth=None` is also a deliberate choice, not an oversight:
    Random Forest's variance control comes from bagging + averaging many
    trees, not from limiting any single tree's depth, so its per-tree depth
    is left unrestricted -- unlike the depth-limited baseline it is compared
    against.
    """

    depth_limited_max_depth: int = 4
    depth_limited_min_samples_leaf: int = 10
    random_state: int = 42

    forest_n_estimators: int = 300
    forest_max_depth: int | None = None
    forest_min_samples_leaf: int = 1

    def __post_init__(self) -> None:
        if self.depth_limited_max_depth < 1:
            raise ValueError(f"depth_limited_max_depth must be >= 1, got {self.depth_limited_max_depth}")
        if self.depth_limited_min_samples_leaf < 1:
            raise ValueError(
                f"depth_limited_min_samples_leaf must be >= 1, got {self.depth_limited_min_samples_leaf}"
            )
        if self.forest_n_estimators < 1:
            raise ValueError(f"forest_n_estimators must be >= 1, got {self.forest_n_estimators}")
        if self.forest_max_depth is not None and self.forest_max_depth < 1:
            raise ValueError(f"forest_max_depth must be >= 1 or None, got {self.forest_max_depth}")
        if self.forest_min_samples_leaf < 1:
            raise ValueError(f"forest_min_samples_leaf must be >= 1, got {self.forest_min_samples_leaf}")


@dataclass(frozen=True)
class BoostingConfig:
    """Layer 3: AdaBoost, Gradient Boosting, and XGBoost settings.

    Boosting fits weak learners sequentially, each one correcting the
    previous ensemble's mistakes. Unlike Random Forest's independently
    bagged trees, more estimators or a higher learning rate can genuinely
    overfit here rather than just plateau -- so `n_estimators`/
    `learning_rate` are fixed, stated choices, not left at whatever a
    library's default happens to be, and deliberately matched across GB and
    XGBoost so a result difference between them reflects the algorithms, not
    mismatched hyperparameters. Not tuned via nested CV, for the same reason
    the depth-limited tree's `max_depth` is not: this project measures
    complexity against a fixed baseline, it does not optimise every knob.

    Neither `AdaBoostClassifier` nor `GradientBoostingClassifier` accepts
    `class_weight` -- a real gap in sklearn's API, not an oversight here.
    `models/trees/boosting.py` applies CLAUDE.md's locked imbalance decision
    to each anyway, by the mechanism each algorithm actually supports:
    AdaBoost via `class_weight='balanced'` on its base stump, Gradient
    Boosting via explicit per-fit `sample_weight`, and XGBoost via
    `scale_pos_weight` -- which is literally what the locked decision names
    it for. All three are computed from each fit call's own `y`, not the
    whole dataset, so imbalance handling stays correctly scoped inside
    whatever fold is being fit.
    """

    adaboost_n_estimators: int = 200
    adaboost_learning_rate: float = 1.0
    adaboost_stump_max_depth: int = 1

    gb_n_estimators: int = 200
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 3

    xgboost_n_estimators: int = 200
    xgboost_learning_rate: float = 0.1
    xgboost_max_depth: int = 3

    random_state: int = 42

    def __post_init__(self) -> None:
        for name in ("adaboost_n_estimators", "gb_n_estimators", "xgboost_n_estimators"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("adaboost_learning_rate", "gb_learning_rate", "xgboost_learning_rate"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("adaboost_stump_max_depth", "gb_max_depth", "xgboost_max_depth"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CVConfig:
    """Cross-validation protocol.

    Repeated stratified k-fold, not a single split: one 80/20 split leaves ~68
    positives in test, where between-model differences are indistinguishable
    from noise.
    """

    n_splits: int = 5
    n_repeats: int = 5
    random_state: int = 42
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if self.n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1, got {self.n_repeats}")

    @property
    def n_fits(self) -> int:
        return self.n_splits * self.n_repeats


@dataclass(frozen=True)
class MetricConfig:
    """Metric settings.

    `beta = 2` weights recall 4x precision. That 4:1 is a stand-in for the real
    cost ratio in `CostConfig`; when F-beta and the cost model disagree about
    which model wins, the cost model is authoritative.

    `thresholds` is the sweep the cost curve is built over. The project's central
    claim is that this axis moves cost more than algorithm choice does, so the
    grid is configuration, not a magic number in a loop.
    """

    beta: float = 2.0
    thresholds: tuple[float, ...] = tuple(round(t, 3) for t in [i / 100 for i in range(1, 100)])
    report_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if not self.thresholds:
            raise ValueError("thresholds grid is empty")
        bad = [t for t in self.thresholds if not 0.0 <= t <= 1.0]
        if bad:
            raise ValueError(f"thresholds must lie in [0, 1]; offending values: {bad[:5]}")


@dataclass(frozen=True)
class CostConfig:
    """Layer 4 cost constants, in currency units.

    **Deliberately unset.** The ratio of missed-failure cost to false-alarm cost
    determines the optimal threshold, decides which recall ceiling is worth
    targeting, and is what the central hypothesis is measured against. Choosing
    it after seeing model results is indistinguishable from tuning toward the
    hypothesis.

    `validate()` raises rather than defaulting, so no cost number can be produced
    before the decision is made and recorded in docs/DECISIONS.md.
    """

    missed_failure: float | None = None
    false_alarm: float | None = None
    inspection: float | None = None
    horizon_hours: float = 1000.0

    @property
    def is_configured(self) -> bool:
        return None not in (self.missed_failure, self.false_alarm, self.inspection)

    @property
    def ratio(self) -> float:
        self.validate()
        assert self.missed_failure is not None and self.false_alarm is not None
        return self.missed_failure / self.false_alarm

    def validate(self) -> None:
        if not self.is_configured:
            unset = [n for n in ("missed_failure", "false_alarm", "inspection") if getattr(self, n) is None]
            raise ValueError(
                f"cost constants not set: {unset}. This is a pending project decision, "
                "not an oversight -- choose the values with a citable justification and "
                "record it in docs/DECISIONS.md before any cost figure is computed."
            )
        for name in ("missed_failure", "false_alarm", "inspection"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration. One object carries everything a run depends on."""

    name: str = "default"
    # Registry key for the estimator under test. A string rather than an object
    # so the choice is recorded verbatim in the results JSON.
    estimator: str = "dummy_constant_negative"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    paths: PathConfig = field(default_factory=PathConfig.default)
    ai4i: AI4ISchema = field(default_factory=AI4ISchema)
    cmapss: CMAPSSSchema = field(default_factory=CMAPSSSchema)
    determinism: DeterminismConfig = field(default_factory=DeterminismConfig)
    eda: EDAConfig = field(default_factory=EDAConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    cost: CostConfig = field(default_factory=CostConfig)

    def __post_init__(self) -> None:
        self.determinism.validate_against(self.ai4i.mode_flags)

    def with_(self, **overrides: Any) -> ExperimentConfig:
        """Return a copy with fields replaced. Configs are never mutated."""
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for the results record."""

        def encode(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: encode(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(v) for v in value]
            return value

        return encode(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=True, default_flow_style=False)

    # -- construction from files -------------------------------------------
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentConfig:
        """Build from a plain mapping, rejecting anything unrecognised.

        Unknown keys raise rather than being ignored. A typo in a config file
        that silently falls back to the default is precisely the failure this
        project cannot afford: the run would be recorded under a name implying
        settings it never used.
        """
        payload = dict(payload)
        sections = {
            "paths": PathConfig,
            "ai4i": AI4ISchema,
            "cmapss": CMAPSSSchema,
            "determinism": DeterminismConfig,
            "eda": EDAConfig,
            "cv": CVConfig,
            "metrics": MetricConfig,
            "cost": CostConfig,
        }
        scalars = {"name", "estimator", "seeds"}

        unknown = set(payload) - set(sections) - scalars
        if unknown:
            raise ValueError(
                f"unknown top-level settings {sorted(unknown)}; " f"valid: {sorted(set(sections) | scalars)}"
            )

        kwargs: dict[str, Any] = {}
        for key in ("name", "estimator"):
            if key in payload:
                kwargs[key] = payload[key]
        if "seeds" in payload:
            kwargs["seeds"] = tuple(payload["seeds"])

        for key, dc_cls in sections.items():
            if key in payload and payload[key] is not None:
                kwargs[key] = _section_from_mapping(dc_cls, payload[key])

        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(payload)


def _section_from_mapping(dc_cls: type, payload: Mapping[str, Any]) -> Any:
    """Build one config dataclass, coercing YAML types and rejecting typos.

    YAML has no tuple type, so sequences arrive as lists and are converted --
    the dataclasses use tuples so they stay immutable and hashable.
    """
    field_names = {f.name for f in fields(dc_cls)}
    unknown = set(payload) - field_names
    if unknown:
        raise ValueError(
            f"{dc_cls.__name__}: unknown settings {sorted(unknown)}; valid: {sorted(field_names)}"
        )

    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            value = tuple(tuple(v) if isinstance(v, list) else v for v in value)
        if dc_cls is PathConfig and value is not None:
            value = Path(value)
        kwargs[key] = value

    if dc_cls is PathConfig:
        return replace(PathConfig.default(), **kwargs)
    return dc_cls(**kwargs)


def default_config() -> ExperimentConfig:
    """The configuration that reproduces the committed EDA results."""
    return ExperimentConfig()
