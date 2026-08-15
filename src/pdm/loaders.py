"""Dataset loaders for AI4I 2020 and NASA C-MAPSS.

Design rule: every loader validates its output and raises on anything
unexpected. A malformed load that raises costs ten minutes; a malformed load
that silently returns a plausible-looking frame costs a week of wrong results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


class DataValidationError(RuntimeError):
    """Raised when a loaded file does not match its expected structure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataValidationError(message)


# ---------------------------------------------------------------------------
# AI4I 2020
# ---------------------------------------------------------------------------
def load_ai4i(path: Path | str | None = None, *, validate: bool = True) -> pd.DataFrame:
    """Load the AI4I 2020 predictive maintenance dataset.

    Returns all 14 columns with normalised names. The five per-mode failure
    flags are included so EDA can audit them, but `config.AI4I_MODE_FLAGS`
    lists them explicitly as banned from any feature matrix -- they are
    deterministic functions of the label generator, i.e. pure leakage.

    The `encoding="utf-8-sig"` is load-bearing: the file carries a UTF-8 BOM,
    so without it the first column is named "\\ufeffUDI" and any lookup by the
    string "UDI" fails.
    """
    path = Path(path) if path is not None else C.AI4I_CSV
    _require(path.exists(), f"AI4I csv not found at {path}")

    df = pd.read_csv(path, encoding=C.AI4I_ENCODING)

    missing = set(C.AI4I_COLUMN_MAP) - set(df.columns)
    _require(
        not missing,
        f"AI4I csv missing expected columns {sorted(missing)}. "
        f"Found: {list(df.columns)}. "
        "A leading '\\ufeff' on the first name means the BOM was not stripped.",
    )

    df = df.rename(columns=C.AI4I_COLUMN_MAP)
    df = df[list(C.AI4I_COLUMN_MAP.values())]

    if validate:
        _require(
            len(df) == C.AI4I_N_ROWS,
            f"AI4I expected {C.AI4I_N_ROWS} rows, got {len(df)}",
        )
        _require(
            not df.isna().any().any(),
            f"AI4I contains NaN in columns "
            f"{df.columns[df.isna().any()].tolist()} -- the published file has none",
        )
        _require(
            df["udi"].is_unique,
            "AI4I 'udi' is not unique; rows may have been duplicated",
        )
        _require(
            set(df["type"].unique()) <= {"L", "M", "H"},
            f"AI4I 'type' has unexpected levels: {sorted(df['type'].unique())}",
        )
        for col in [C.AI4I_TARGET, *C.AI4I_MODE_FLAGS]:
            _require(
                set(df[col].unique()) <= {0, 1},
                f"AI4I '{col}' is not binary; found {sorted(df[col].unique())}",
            )

    return df


def ai4i_feature_columns() -> list[str]:
    """The only columns permitted in an AI4I feature matrix (pre-engineering)."""
    return C.AI4I_SENSOR_FEATURES + C.AI4I_CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# C-MAPSS
# ---------------------------------------------------------------------------
def load_cmapss(
    split: str = "train",
    subset: str = C.CMAPSS_SUBSET,
    *,
    data_dir: Path | str | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load one C-MAPSS trajectory file as a tidy cycle-level frame.

    Parameters
    ----------
    split : "train" or "test".
    subset : "FD001".."FD004". This project uses FD001 only.

    Notes
    -----
    The raw files are whitespace-delimited with *trailing* whitespace on every
    line and no header row. Parsing with ``sep=" "`` produces 28 columns, two
    of them entirely NaN, which silently shifts positional indexing. The regex
    separator plus the hard column-count assertion below prevents that.

    Column order is fixed by the NASA spec: unit, cycle, three operational
    settings, then sensors 1-21. Sensor numbering starts at 1 at *column index
    5*; an off-by-one here mislabels every sensor and produces no error.
    """
    _require(split in {"train", "test"}, f"split must be 'train' or 'test', got {split!r}")
    data_dir = Path(data_dir) if data_dir is not None else C.CMAPSS_DIR
    path = data_dir / f"{split}_{subset}.txt"
    _require(path.exists(), f"C-MAPSS file not found at {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    _require(
        df.shape[1] == C.CMAPSS_N_COLUMNS,
        f"{path.name}: expected {C.CMAPSS_N_COLUMNS} columns, parsed {df.shape[1]}. "
        "Check the delimiter -- trailing whitespace can create phantom columns.",
    )
    df.columns = C.CMAPSS_COLUMNS

    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    if validate:
        _require(
            not df.isna().any().any(),
            f"{path.name} contains NaN in {df.columns[df.isna().any()].tolist()}",
        )
        expected = C.CMAPSS_EXPECTED_UNITS.get(subset, {}).get(split)
        if expected is not None:
            _require(
                df["unit"].nunique() == expected,
                f"{path.name}: expected {expected} units, found {df['unit'].nunique()}",
            )
        # Units must be a contiguous 1..N block; the RUL file is matched to
        # units positionally, and that is only safe if this holds.
        units = np.sort(df["unit"].unique())
        _require(
            np.array_equal(units, np.arange(1, len(units) + 1)),
            f"{path.name}: unit ids are not a contiguous 1..N block: "
            f"min={units.min()}, max={units.max()}, n={len(units)}",
        )
        # Each trajectory must be a complete 1..T cycle sequence in order.
        for unit, grp in df.groupby("unit", sort=True):
            cyc = grp["cycle"].to_numpy()
            _require(
                np.array_equal(cyc, np.arange(1, len(cyc) + 1)),
                f"{path.name}: unit {unit} has non-contiguous or unsorted cycles",
            )

    return df


def load_cmapss_rul(
    subset: str = C.CMAPSS_SUBSET,
    *,
    data_dir: Path | str | None = None,
) -> pd.Series:
    """Load the true remaining-useful-life vector for the *test* trajectories.

    The file is a bare column of numbers with no unit identifier. It is matched
    to units **by position**: line i corresponds to test unit i+1. This is the
    convention NASA distributes it under, and `load_cmapss` asserts that test
    unit ids really are the contiguous block 1..N, which is what makes the
    positional match safe.

    Returns a Series indexed by unit id.
    """
    data_dir = Path(data_dir) if data_dir is not None else C.CMAPSS_DIR
    path = data_dir / f"RUL_{subset}.txt"
    _require(path.exists(), f"C-MAPSS RUL file not found at {path}")

    rul = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    _require(rul.shape[1] == 1, f"{path.name}: expected 1 column, got {rul.shape[1]}")

    expected = C.CMAPSS_EXPECTED_UNITS.get(subset, {}).get("test")
    if expected is not None:
        _require(
            len(rul) == expected,
            f"{path.name}: expected {expected} RUL values, got {len(rul)}",
        )

    out = pd.Series(
        rul[0].astype(int).to_numpy(),
        index=pd.Index(range(1, len(rul) + 1), name="unit"),
        name="rul_at_last_cycle",
    )
    return out


def cmapss_lifetimes(
    subset: str = C.CMAPSS_SUBSET,
    *,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Build the unit-level lifetime table in standard survival-analysis form.

    One row per engine, with the two columns any censored-lifetime estimator
    needs: a ``duration`` and an ``event`` indicator.

    Censoring structure -- this is the part that silently corrupts results if
    misread:

    * **train** trajectories run all the way to failure. ``duration`` is the
      last observed cycle and ``event = 1`` (failure observed, uncensored).

    * **test** trajectories are truncated at some point *before* failure.
      ``duration`` is the last observed cycle, which is a **censoring time,
      not a lifetime**, and ``event = 0``. Treating a test engine's last cycle
      as its lifetime biases every fitted lifetime downward, and nothing in the
      pipeline will complain.

    * ``true_duration`` is only populated for test units, as
      ``last_cycle + RUL``. Because the RUL file is supplied, the test set is
      effectively **de-censored**: it is ground truth for evaluation, not an
      input. Fitting on ``true_duration`` and then reporting a "censored"
      model would be circular. Keep it out of any fit.
    """
    frames = []

    train = load_cmapss("train", subset, data_dir=data_dir)
    train_life = (
        train.groupby("unit")["cycle"]
        .max()
        .rename("duration")
        .reset_index()
        .assign(split="train", event=1, true_duration=lambda d: d["duration"])
    )
    frames.append(train_life)

    test = load_cmapss("test", subset, data_dir=data_dir)
    rul = load_cmapss_rul(subset, data_dir=data_dir)
    test_life = (
        test.groupby("unit")["cycle"].max().rename("duration").reset_index().assign(split="test", event=0)
    )
    _require(
        set(test_life["unit"]) == set(rul.index),
        "C-MAPSS test unit ids do not match the RUL file's positional index",
    )
    test_life["true_duration"] = test_life["duration"] + test_life["unit"].map(rul)
    frames.append(test_life)

    out = pd.concat(frames, ignore_index=True)
    out = out[["split", "unit", "duration", "event", "true_duration"]]

    _require(
        (out["duration"] > 0).all(),
        "C-MAPSS lifetime table contains a non-positive duration",
    )
    _require(
        (out["true_duration"] >= out["duration"]).all(),
        "C-MAPSS true_duration is below the observed duration for some unit",
    )
    return out
