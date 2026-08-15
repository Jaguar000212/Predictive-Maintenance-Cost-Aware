"""Paths, column schemas, and dataset constants.

Single source of truth. Nothing else in the codebase should hardcode a path,
a column name, or a failure-mode grouping.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# config.py -> pdm -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = REPO_ROOT / "data" / "raw"
AI4I_CSV = DATA_RAW / "AI4I_2020" / "ai4i2020.csv"
CMAPSS_DIR = DATA_RAW / "CMAPSS_2008"

REPORTS_DIR = REPO_ROOT / "reports" / "eda"


# --- AI4I 2020 -------------------------------------------------------------
AI4I_N_ROWS = 10_000

# The CSV is written with a UTF-8 BOM, so the first header cell parses as
# "﻿UDI" unless the file is opened with encoding="utf-8-sig".
AI4I_ENCODING = "utf-8-sig"

# Raw header -> internal name. Raw names carry units in square brackets, which
# are awkward to reference and break some libraries' formula interfaces.
AI4I_COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    # Failure-mode flags keep their canonical uppercase acronyms.
    "TWF": "TWF",
    "HDF": "HDF",
    "PWF": "PWF",
    "OSF": "OSF",
    "RNF": "RNF",
}

AI4I_TARGET = "machine_failure"

# All five per-mode indicator flags.
AI4I_MODE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Locked decision: mode flags are dropped from the feature matrix. They are
# leakage -- each one is a deterministic function of the label's generator.
# They are retained in the loaded frame purely for EDA and ceiling analysis.
AI4I_SENSOR_FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
AI4I_CATEGORICAL_FEATURES = ["type"]

# Columns that must never enter a feature matrix.
AI4I_EXCLUDED_FROM_FEATURES = ["udi", "product_id", AI4I_TARGET] + AI4I_MODE_FLAGS


# --- Failure-mode determinism grouping -------------------------------------
# This grouping is a MODELLING ASSUMPTION, not a fact recorded in the file.
# It comes from the AI4I 2020 generator description (Matzka 2020):
#
#   HDF  heat dissipation failure -- fires when (process_temp - air_temp) is
#        below a fixed threshold AND rotational speed is below a fixed
#        threshold. Both operands are observed features => deterministic.
#   PWF  power failure -- fires when torque * angular_velocity falls outside a
#        fixed wattage band. Both operands are observed features
#        => deterministic.
#   OSF  overstrain failure -- fires when tool_wear * torque exceeds a limit
#        that depends on the product `type` (L/M/H). All operands are observed
#        features => deterministic.
#
#   TWF  tool wear failure -- the tool is retired at a wear threshold drawn
#        uniformly from [200, 240] minutes. The draw is NOT observable, so the
#        boundary is only partially recoverable: rows inside the overlap band
#        are genuinely ambiguous. Treated as semi-deterministic.
#
#   RNF  random failure -- an independent 0.1% chance per process, unrelated to
#        every feature. Irreducible by construction.
#
# A model can only be expected to recover the deterministic modes, so those
# define the recall ceiling. TWF is reported separately so the sensitivity of
# the ceiling to that judgement call is visible rather than buried.
AI4I_DETERMINISTIC_MODES = ["HDF", "PWF", "OSF"]
AI4I_SEMI_DETERMINISTIC_MODES = ["TWF"]
AI4I_STOCHASTIC_MODES = ["RNF"]


# --- C-MAPSS ---------------------------------------------------------------
CMAPSS_SUBSET = "FD001"

CMAPSS_N_OP_SETTINGS = 3
CMAPSS_N_SENSORS = 21

# The bundled readme.txt says the 26 columns are "sensor measurement 1 ... 26".
# That is a typo in the original NASA distribution. Columns 1-5 are unit,
# cycle, and three operational settings; the 21 sensors occupy columns 6-26.
CMAPSS_ID_COLUMNS = ["unit", "cycle"]
CMAPSS_OP_COLUMNS = [f"op_setting_{i}" for i in range(1, CMAPSS_N_OP_SETTINGS + 1)]
CMAPSS_SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, CMAPSS_N_SENSORS + 1)]
CMAPSS_COLUMNS = CMAPSS_ID_COLUMNS + CMAPSS_OP_COLUMNS + CMAPSS_SENSOR_COLUMNS
CMAPSS_N_COLUMNS = len(CMAPSS_COLUMNS)  # 26

CMAPSS_EXPECTED_UNITS = {
    "FD001": {"train": 100, "test": 100},
    "FD002": {"train": 260, "test": 259},
    "FD003": {"train": 100, "test": 100},
    "FD004": {"train": 248, "test": 249},
}
