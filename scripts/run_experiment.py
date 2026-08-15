"""Composition root for a single experiment.

    python scripts/run_experiment.py configs/dummy.yaml

Loads a config, assembles the feature matrix, runs cross-validation, and writes
one JSON to results/ containing the config, seeds, git SHA, environment, and
metrics. Nothing else writes to results/.

This is the Week 1 gate: with configs/dummy.yaml it runs a constant-negative
baseline end to end. Expected output is recall 0 and PR-AUC = Brier = the
positive base rate (0.0339). Any other numbers mean the harness is broken, not
that the model is poor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Fallback for a clone where `pip install -e .` has not been run yet.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pdm.config import AI4ISchema, ExperimentConfig
from pdm.eval.cv import CrossValidator
from pdm.eval.metrics import MetricSuite
from pdm.eval.results import GitState, ResultsWriter, RunRecord
from pdm.loaders import AI4ILoader
from pdm.models import registry


class LeakageError(RuntimeError):
    """A banned column reached the feature matrix."""


def build_xy(df: pd.DataFrame, schema: AI4ISchema) -> tuple[pd.DataFrame, pd.Series]:
    """Split the loaded frame into features and target, and prove no leakage.

    The assertion is not decoration. Leaving a mode flag in the features is the
    project's most likely silent failure: TWF/HDF/PWF/OSF are deterministic
    functions of the label's generator, so a model trained on them scores ~0.99
    and raises nothing. Checking here means the guard runs on every experiment
    rather than depending on whoever assembles the matrix next.
    """
    X = df[list(schema.feature_columns)].copy()
    y = df[schema.target].copy()

    banned = set(schema.excluded_from_features) & set(X.columns)
    if banned:
        raise LeakageError(
            f"banned columns reached the feature matrix: {sorted(banned)}. "
            "Mode flags are deterministic functions of the label generator; "
            "training on them produces ~0.99 scores and no error."
        )
    return X, y


def run(config: ExperimentConfig, write: bool = True) -> tuple[RunRecord, Path | None]:
    """Execute one experiment and return its record."""
    df = AI4ILoader(config.ai4i, config.paths).load()
    X, y = build_xy(df, config.ai4i)

    factory = registry.build(config.estimator, config.ai4i, config.seeds[0])
    validator = CrossValidator(config.cv, MetricSuite(config.metrics))
    result = validator.run(factory, X, y, name=config.estimator)

    record = RunRecord(
        name=config.name,
        config=config.to_dict(),
        metrics=result.to_dict(),
        seeds=list(config.seeds),
        git=GitState.capture(config.paths.repo_root),
    )

    path = ResultsWriter(config.paths.results_dir).write(record) if write else None
    return record, path


def _format(record: RunRecord, config: ExperimentConfig) -> str:
    summary = pd.DataFrame(record.metrics["summary"]).T
    keep = [m for m in ("pr_auc", "recall", "precision", "f2", "brier") if m in summary.index]
    lines = [
        "=" * 78,
        f"EXPERIMENT: {config.name}   estimator={config.estimator}",
        "=" * 78,
        (
            f"folds        : {record.metrics['n_fits']} "
            f"({config.cv.n_splits} splits x {config.cv.n_repeats} repeats)"
        ),
        f"threshold    : {config.metrics.report_threshold}",
        (
            f"git          : {record.git.sha[:8] if record.git.sha else 'n/a'}"
            f"{' (DIRTY -- SHA does not identify this code)' if record.git.dirty else ''}"
        ),
        "",
        summary.loc[keep, ["mean", "std", "min", "max"]].to_string(),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one experiment from a YAML config.")
    parser.add_argument("config", help="path to a YAML config, e.g. configs/dummy.yaml")
    parser.add_argument("--no-write", action="store_true", help="print only, write no results JSON")
    args = parser.parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    record, path = run(config, write=not args.no_write)

    print(_format(record, config))
    if path:
        print(f"\n[written] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
