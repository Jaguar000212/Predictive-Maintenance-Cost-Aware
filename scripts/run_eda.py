"""Entry point for the EDA report.

    python scripts/run_eda.py

Writes reports/eda/eda_report.txt plus one CSV per table, and echoes the
report to stdout. No models are trained here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm import config as C
from pdm import eda


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EDA for the predictive maintenance project.")
    parser.add_argument("--subset", default=C.CMAPSS_SUBSET, help="C-MAPSS subset (default: FD001)")
    parser.add_argument("--out", default=str(C.REPORTS_DIR), help="output directory")
    parser.add_argument("--no-write", action="store_true", help="print only, write nothing")
    args = parser.parse_args()

    text, tables = eda.run_all(args.subset)
    print(text)

    if args.no_write:
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eda_report.txt").write_text(text, encoding="utf-8")
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=True)

    print(f"\n[written] {out / 'eda_report.txt'}")
    print(f"[written] {len(tables)} CSV tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
