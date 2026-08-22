"""Composition root for Layer 4: the cost curve, honest threshold
optimisation, and the policy table -- the numbers `README.md`'s Layer 4
section and `docs/DECISIONS.md` D11/D12 report.

    python scripts/run_cost_analysis.py

`run_experiment.py` does not cover this layer: it only ever computes
threshold-0.5 classification metrics, never a cost curve or the oracle
policy comparison. Every Layer 4 number in this project was, until now,
produced by a throwaway scratch script and never written to `results/` --
exactly the "unrecorded run" this project's own rule prohibits citing. This
script is that recording, reusing the same `RunRecord`/`ResultsWriter`
`run_experiment.py` already uses rather than inventing a second schema.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

# Fallback for a clone where `pip install -e .` has not been run yet.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_experiment import build_xy

from pdm.config import AI4ISchema, CostConfig, CVConfig, DeterminismConfig, ExperimentConfig
from pdm.decision.cost_model import cross_validated_cost_curve
from pdm.decision.policy_sim import policy_table
from pdm.eval.results import GitState, ResultsWriter, RunRecord
from pdm.loaders import AI4ILoader
from pdm.models import registry

# The two models README/D10 actually cite cost numbers for. Matches the
# pre-registered falsification test (XGBoost, D10) and its baseline.
_ESTIMATORS = ("decision_tree", "xgboost")


def main() -> int:
    paths = ExperimentConfig().paths
    schema = AI4ISchema()
    cost = CostConfig()  # D11: 10 / 1 / 0.5
    cv_config = CVConfig(n_splits=5, n_repeats=5, random_state=42)
    determinism = DeterminismConfig()

    df = AI4ILoader(schema, paths).load()
    X, y = build_xy(df, schema)

    cv_results = {}
    for name in _ESTIMATORS:
        factory = registry.build(name, schema, seed=0)
        result = cross_validated_cost_curve(factory, X, y, cost=cost, cv_config=cv_config, name=name)
        optimal = result.optimal
        cv_results[name] = {
            "n_fits": result.n_fits,
            "optimal_threshold": float(optimal["threshold"]),
            "cost_per_row_mean": float(optimal["cost_per_row_mean"]),
            "cost_per_row_std": float(optimal["cost_per_row_std"]),
        }
        print(
            f"{name:15s} n_fits={result.n_fits:3d}  best_thr={optimal['threshold']:.2f}  "
            f"cost_mean={optimal['cost_per_row_mean']:.4f}  cost_std={optimal['cost_per_row_std']:.4f}"
        )

    policy = policy_table(df, cost=cost, schema=schema, determinism=determinism)
    print()
    print(policy.to_string(index=False))

    # D11's sensitivity check: is D12's strict-vs-extended verdict robust to
    # the exact ratio, or specific to 10:1? False_alarm/inspection held fixed.
    print()
    sensitivity = {}
    for missed_failure in (5.0, 10.0, 20.0):
        ratio_cost = CostConfig(missed_failure=missed_failure, false_alarm=1.0, inspection=0.5)
        ratio_table = policy_table(df, cost=ratio_cost, schema=schema, determinism=determinism).set_index(
            "policy"
        )
        strict = float(ratio_table.loc["strict_physics_ceiling", "cost_per_row"])
        extended = float(ratio_table.loc["strict_physics_plus_wear_band", "cost_per_row"])
        sensitivity[f"{missed_failure:.0f}:1:0.5"] = {
            "strict_cost_per_row": strict,
            "extended_cost_per_row": extended,
            "extended_cheaper": extended < strict,
        }
        print(
            f"ratio {missed_failure:.0f}:1:0.5  strict={strict:.4f}  extended={extended:.4f}  "
            f"-> {'extended cheaper' if extended < strict else 'strict cheaper'}"
        )

    record = RunRecord(
        name="layer4_cost_analysis",
        config={
            "cost": asdict(cost),
            "cv": asdict(cv_config),
            "determinism": asdict(determinism),
            "estimators": list(_ESTIMATORS),
        },
        metrics={
            "cross_validated_cost_curve": cv_results,
            "policy_table": policy.to_dict(orient="records"),
            "ratio_sensitivity": sensitivity,
        },
        seeds=[0],
        git=GitState.capture(paths.repo_root),
        notes=(
            "Layer 4: honest CV cost-optimal thresholds for decision_tree/xgboost "
            "(cost_model.cross_validated_cost_curve) and the oracle policy table "
            "(policy_sim.policy_table). See docs/DECISIONS.md D11/D12."
        ),
    )
    path = ResultsWriter(paths.results_dir).write(record)
    print(f"\n[written] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
