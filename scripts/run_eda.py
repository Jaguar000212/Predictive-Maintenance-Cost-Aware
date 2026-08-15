"""Composition root for the EDA report.

    python scripts/run_eda.py

This is the only place that knows both how to load data and which analyses to
run. The analyses receive frames through their constructors, which is what keeps
them testable against fixtures instead of the real dataset.

Writes reports/eda/eda_report.txt plus one CSV per table. No models are trained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Fallback for a clone where `pip install -e .` has not been run yet.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pdm.config import CMAPSSSchema, ExperimentConfig, default_config
from pdm.eda import (
    AI4IFailureCountAnalysis,
    AI4ILabelDisagreementAnalysis,
    AI4IRecallCeilingAnalysis,
    CMAPSSLifetimeAnalysis,
    EDAReport,
)
from pdm.loaders import AI4ILoader, CMAPSSLifetimeBuilder, CMAPSSLoader


def build_report(config: ExperimentConfig) -> EDAReport:
    """Wire loaders to analyses. Either side can be swapped without the other."""
    ai4i = AI4ILoader(config.ai4i, config.paths).load()
    lifetimes = CMAPSSLifetimeBuilder(config.cmapss, config.paths).build()
    train_cycles = CMAPSSLoader("train", config.cmapss, config.paths).load()

    return EDAReport(
        [
            CMAPSSLifetimeAnalysis(lifetimes, train_cycles, config.cmapss, config.eda),
            AI4IFailureCountAnalysis(ai4i, config.ai4i),
            AI4ILabelDisagreementAnalysis(ai4i, config.ai4i),
            AI4IRecallCeilingAnalysis(ai4i, config.ai4i, config.determinism),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EDA for the predictive maintenance project.")
    parser.add_argument("--subset", default=None, help="C-MAPSS subset (default: FD001)")
    parser.add_argument("--out", default=None, help="output directory (default: reports/eda)")
    parser.add_argument("--no-write", action="store_true", help="print only, write nothing")
    args = parser.parse_args(argv)

    config = default_config()
    if args.subset:
        config = config.with_(cmapss=CMAPSSSchema.for_subset(args.subset))

    report = build_report(config)
    text, tables = report.run()
    print(text)

    if args.no_write:
        return 0

    out = Path(args.out) if args.out else config.paths.eda_dir
    path = report.write(out)
    print(f"\n[written] {path}")
    print(f"[written] {len(tables)} CSV tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
