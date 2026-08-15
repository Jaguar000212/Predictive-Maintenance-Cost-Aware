"""Dataset loaders.

Design rule: a loader validates its own output and raises rather than coercing.
A malformed load that raises costs ten minutes; a malformed load that returns a
plausible-looking frame costs a week of wrong results.

Structure: `DatasetLoader` fixes the contract (`path`, `load`, `_require`), and
each subclass owns exactly one file format. The base class prefixes every
validation failure with the offending filename, so no subclass has to remember
to. Adding a C-MAPSS subset is a `CMAPSSSchema.for_subset(...)` call; adding a
new dataset is one new subclass and no edits to existing ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AI4ISchema, CMAPSSSchema, PathConfig


class DataValidationError(RuntimeError):
    """A loaded file does not match its expected structure."""


class DatasetLoader(ABC):
    """Contract shared by every loader."""

    @property
    @abstractmethod
    def path(self) -> Path:
        """The file this loader reads."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return the validated frame."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def _require(self, condition: bool, message: str) -> None:
        if not condition:
            raise DataValidationError(f"{self.path.name}: {message}")

    def _require_exists(self) -> None:
        if not self.path.exists():
            raise DataValidationError(
                f"file not found at {self.path}. Raw data is gitignored -- restore it "
                "under data/raw/ before running."
            )


# ---------------------------------------------------------------------------
# AI4I 2020
# ---------------------------------------------------------------------------
class AI4ILoader(DatasetLoader):
    """Loader for the AI4I 2020 predictive maintenance CSV.

    Returns all 14 columns under normalised names. The five per-mode failure
    flags are kept so EDA can audit them; `AI4ISchema.excluded_from_features`
    is what bars them from a feature matrix.
    """

    def __init__(
        self,
        schema: AI4ISchema | None = None,
        paths: PathConfig | None = None,
        path: Path | str | None = None,
        validate: bool = True,
    ) -> None:
        self.schema = schema or AI4ISchema()
        self.paths = paths or PathConfig.default()
        self._path_override = Path(path) if path is not None else None
        self.validate = validate

    @property
    def path(self) -> Path:
        return self._path_override or self.paths.ai4i_csv

    def load(self) -> pd.DataFrame:
        self._require_exists()

        # The file carries a UTF-8 BOM. pandas strips it under any declared
        # encoding, so utf-8-sig is belt-and-braces here rather than required;
        # the column check below is what would actually catch a mangled header.
        df = pd.read_csv(self.path, encoding=self.schema.encoding)

        missing = set(self.schema.column_map) - set(df.columns)
        self._require(
            not missing,
            f"missing expected columns {sorted(missing)}. Found: {list(df.columns)}. "
            "A leading '\\ufeff' on the first name means the BOM was not stripped.",
        )

        df = df.rename(columns=self.schema.column_map)[list(self.schema.columns)]

        if self.validate:
            self._validate(df)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        s = self.schema
        if s.expected_rows is not None:
            self._require(len(df) == s.expected_rows, f"expected {s.expected_rows} rows, got {len(df)}")
        self._require(
            not df.isna().any().any(),
            f"contains NaN in {df.columns[df.isna().any()].tolist()} -- the published file has none",
        )
        self._require(df["udi"].is_unique, "'udi' is not unique; rows may have been duplicated")
        self._require(
            set(df["type"].unique()) <= {"L", "M", "H"},
            f"'type' has unexpected levels: {sorted(df['type'].unique())}",
        )
        for col in (s.target, *s.mode_flags):
            self._require(
                set(df[col].unique()) <= {0, 1},
                f"'{col}' is not binary; found {sorted(df[col].unique())}",
            )


# ---------------------------------------------------------------------------
# C-MAPSS
# ---------------------------------------------------------------------------
class CMAPSSLoader(DatasetLoader):
    """Loader for one C-MAPSS trajectory file, as a tidy cycle-level frame.

    The raw files are whitespace-delimited with trailing whitespace on every
    line and no header. Parsing with ``sep=" "`` yields 28 columns, two entirely
    NaN, which silently shifts positional indexing. The regex separator plus the
    column-count check below prevents that.

    Column order is fixed by the NASA spec and sensor numbering starts at 1 at
    *column index 5*. An off-by-one mislabels every sensor and raises nothing.
    """

    def __init__(
        self,
        split: str = "train",
        schema: CMAPSSSchema | None = None,
        paths: PathConfig | None = None,
        data_dir: Path | str | None = None,
        validate: bool = True,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.split = split
        self.schema = schema or CMAPSSSchema()
        self.paths = paths or PathConfig.default()
        self._data_dir = Path(data_dir) if data_dir is not None else None
        self.validate = validate

    @property
    def data_dir(self) -> Path:
        return self._data_dir or self.paths.cmapss_dir

    @property
    def path(self) -> Path:
        return self.data_dir / f"{self.split}_{self.schema.subset}.txt"

    def load(self) -> pd.DataFrame:
        self._require_exists()
        df = pd.read_csv(self.path, sep=r"\s+", header=None, engine="python")

        self._require(
            df.shape[1] == self.schema.n_columns,
            f"expected {self.schema.n_columns} columns, parsed {df.shape[1]}. "
            "Check the delimiter -- trailing whitespace can create phantom columns.",
        )
        df.columns = list(self.schema.columns)
        df["unit"] = df["unit"].astype(int)
        df["cycle"] = df["cycle"].astype(int)

        if self.validate:
            self._validate(df)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        self._require(not df.isna().any().any(), f"contains NaN in {df.columns[df.isna().any()].tolist()}")

        expected = self.schema.expected_units(self.split)
        if expected is not None:
            self._require(
                df["unit"].nunique() == expected,
                f"expected {expected} units, found {df['unit'].nunique()}",
            )

        # Units must form a contiguous 1..N block: the RUL file is matched to
        # units positionally, and that is only safe if this holds.
        units = np.sort(df["unit"].unique())
        self._require(
            np.array_equal(units, np.arange(1, len(units) + 1)),
            f"unit ids are not a contiguous 1..N block: min={units.min()}, "
            f"max={units.max()}, n={len(units)}",
        )

        for unit, grp in df.groupby("unit", sort=True):
            cycles = grp["cycle"].to_numpy()
            self._require(
                np.array_equal(cycles, np.arange(1, len(cycles) + 1)),
                f"unit {unit} has non-contiguous or unsorted cycles",
            )


class CMAPSSRULLoader(DatasetLoader):
    """Loader for the true remaining-useful-life vector of the *test* split.

    The file is a bare column of numbers with no unit identifier, matched to
    units **by position**: line i is test unit i+1. That is the convention NASA
    distributes it under, and `CMAPSSLoader` asserts test unit ids really are a
    contiguous 1..N block, which is what makes the positional match safe.
    """

    def __init__(
        self,
        schema: CMAPSSSchema | None = None,
        paths: PathConfig | None = None,
        data_dir: Path | str | None = None,
    ) -> None:
        self.schema = schema or CMAPSSSchema()
        self.paths = paths or PathConfig.default()
        self._data_dir = Path(data_dir) if data_dir is not None else None

    @property
    def data_dir(self) -> Path:
        return self._data_dir or self.paths.cmapss_dir

    @property
    def path(self) -> Path:
        return self.data_dir / f"RUL_{self.schema.subset}.txt"

    def load(self) -> pd.DataFrame:
        return self.load_series().to_frame()

    def load_series(self) -> pd.Series:
        self._require_exists()
        rul = pd.read_csv(self.path, sep=r"\s+", header=None, engine="python")
        self._require(rul.shape[1] == 1, f"expected 1 column, got {rul.shape[1]}")

        expected = self.schema.expected_test_units
        if expected is not None:
            self._require(len(rul) == expected, f"expected {expected} RUL values, got {len(rul)}")

        return pd.Series(
            rul[0].astype(int).to_numpy(),
            index=pd.Index(range(1, len(rul) + 1), name="unit"),
            name="rul_at_last_cycle",
        )


class CMAPSSLifetimeBuilder:
    """Builds the unit-level lifetime table in standard survival-analysis form.

    One row per engine with the two columns any censored-lifetime estimator
    needs: a ``duration`` and an ``event`` indicator.

    Censoring structure -- the part that silently corrupts results if misread:

    * **train** trajectories run to failure. ``duration`` is the last observed
      cycle and ``event = 1`` (uncensored).
    * **test** trajectories are truncated *before* failure. ``duration`` is the
      last observed cycle, which is a **censoring time, not a lifetime**, and
      ``event = 0``. Treating it as a lifetime biases every fitted distribution
      downward and nothing complains.
    * ``true_duration`` is ``last_cycle + RUL`` for test units. Because the RUL
      file is supplied, the test set is effectively **de-censored**: it is ground
      truth for evaluation, not an input. Fitting on it and reporting a
      "censored" model is circular. Keep it out of any fit.

    Composes three loaders rather than reading files itself, so a test can inject
    fixtures through `data_dir` without touching the real dataset.
    """

    def __init__(
        self,
        schema: CMAPSSSchema | None = None,
        paths: PathConfig | None = None,
        data_dir: Path | str | None = None,
    ) -> None:
        self.schema = schema or CMAPSSSchema()
        self.paths = paths or PathConfig.default()
        self.data_dir = data_dir
        self.train_loader = CMAPSSLoader("train", self.schema, self.paths, data_dir)
        self.test_loader = CMAPSSLoader("test", self.schema, self.paths, data_dir)
        self.rul_loader = CMAPSSRULLoader(self.schema, self.paths, data_dir)

    def build(self) -> pd.DataFrame:
        train_life = (
            self.train_loader.load()
            .groupby("unit")["cycle"]
            .max()
            .rename("duration")
            .reset_index()
            .assign(split="train", event=1, true_duration=lambda d: d["duration"])
        )

        test_life = (
            self.test_loader.load()
            .groupby("unit")["cycle"]
            .max()
            .rename("duration")
            .reset_index()
            .assign(split="test", event=0)
        )
        rul = self.rul_loader.load_series()
        if set(test_life["unit"]) != set(rul.index):
            raise DataValidationError("C-MAPSS test unit ids do not match the RUL file's positional index")
        test_life["true_duration"] = test_life["duration"] + test_life["unit"].map(rul)

        out = pd.concat([train_life, test_life], ignore_index=True)
        out = out[["split", "unit", "duration", "event", "true_duration"]]

        if not (out["duration"] > 0).all():
            raise DataValidationError("lifetime table contains a non-positive duration")
        if not (out["true_duration"] >= out["duration"]).all():
            raise DataValidationError("true_duration is below the observed duration for some unit")
        return out


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
# Thin delegations for the common case. Construct the classes directly when a
# non-default schema, path, or subset is needed.


def load_ai4i(path: Path | str | None = None, *, validate: bool = True, **kwargs) -> pd.DataFrame:
    return AI4ILoader(path=path, validate=validate, **kwargs).load()


def load_cmapss(split: str = "train", subset: str = "FD001", **kwargs) -> pd.DataFrame:
    schema = kwargs.pop("schema", None) or CMAPSSSchema.for_subset(subset)
    return CMAPSSLoader(split, schema, **kwargs).load()


def load_cmapss_rul(subset: str = "FD001", **kwargs) -> pd.Series:
    schema = kwargs.pop("schema", None) or CMAPSSSchema.for_subset(subset)
    return CMAPSSRULLoader(schema, **kwargs).load_series()


def cmapss_lifetimes(subset: str = "FD001", **kwargs) -> pd.DataFrame:
    schema = kwargs.pop("schema", None) or CMAPSSSchema.for_subset(subset)
    return CMAPSSLifetimeBuilder(schema, **kwargs).build()
