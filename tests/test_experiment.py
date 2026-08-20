"""End-to-end experiment tests, including the Week 1 gate as a regression test.

The gate numbers are asserted here rather than only observed once, so a future
change that quietly breaks the harness fails a test instead of producing a
plausible table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_experiment import LeakageError, build_xy, run

from pdm.config import AI4ISchema, CVConfig, ExperimentConfig
from pdm.models import registry

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_exposes_the_dummies():
    assert "dummy_constant_negative" in registry.available()
    assert "dummy_prior" in registry.available()


def test_build_returns_a_factory_not_an_instance():
    """CrossValidator needs a factory so every fold gets a fresh estimator."""
    factory = registry.build("dummy_constant_negative", AI4ISchema(), seed=0)
    assert callable(factory)
    assert factory() is not factory()


def test_unknown_estimator_is_rejected_with_the_valid_names():
    with pytest.raises(KeyError, match="unknown estimator"):
        registry.build("xgboost_9000", AI4ISchema(), seed=0)


@pytest.mark.parametrize(
    "name", ["dummy_stratified", "gnb", "bayes_logreg", "decision_tree", "random_forest"]
)
def test_every_factory_call_produces_a_fresh_unfitted_classifier(name):
    """Regression test for a structural risk fixed in registry.py: the
    classifier step (and, for the physics pipeline, every step) must be
    constructed fresh per call, not captured once and reused across folds.
    Today's estimators reset their own state on refit, so reuse would not
    currently produce a wrong number -- but the factory pattern exists
    precisely so that stops being something anyone has to get right by luck.
    """
    factory = registry.build(name, AI4ISchema(), seed=0)
    first = factory().named_steps["classifier"]
    second = factory().named_steps["classifier"]
    assert first is not second


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        registry.register("dummy_prior")(lambda schema, seed: None)


def test_preprocessor_keeps_type_and_drops_nothing_admissible():
    """OSF's threshold is tier-dependent; the ceiling needs `type` present."""
    schema = AI4ISchema()
    columns = [c for _, _, cols in registry.preprocessor(schema).transformers for c in cols]
    assert "type" in columns
    assert set(schema.numeric_features) <= set(columns)
    assert not set(schema.mode_flags) & set(columns)


# ---------------------------------------------------------------------------
# Leakage guard on the feature matrix
# ---------------------------------------------------------------------------
def test_feature_matrix_excludes_flags_target_and_identifiers():
    schema = AI4ISchema()
    df = pd.DataFrame({c: [0, 1] for c in schema.columns} | {"type": ["L", "M"]})
    X, y = build_xy(df, schema)

    assert not set(schema.mode_flags) & set(X.columns)
    assert schema.target not in X.columns
    assert "udi" not in X.columns and "product_id" not in X.columns
    assert list(y) == [0, 1]


def test_a_mode_flag_smuggled_into_features_is_caught():
    """The project's most likely silent failure: ~0.99 scores and no error."""
    leaky = AI4ISchema(numeric_features=("air_temp_k", "torque_nm", "HDF"))
    df = pd.DataFrame({c: [0, 1] for c in leaky.columns} | {"type": ["L", "M"]})

    with pytest.raises(LeakageError, match="HDF"):
        build_xy(df, leaky)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def test_dummy_config_file_loads():
    config = ExperimentConfig.from_yaml(CONFIGS / "dummy.yaml")
    assert config.estimator == "dummy_constant_negative"
    assert config.cv.n_splits == 5 and config.cv.n_repeats == 5
    assert not config.cost.is_configured  # the pending decision stays pending


def test_a_typo_in_a_config_is_rejected_not_ignored(tmp_path):
    """A silently-ignored key records a run under settings it never used."""
    path = tmp_path / "typo.yaml"
    path.write_text("name: x\ncv:\n  n_split: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown settings"):
        ExperimentConfig.from_yaml(path)


def test_an_unknown_top_level_section_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nmodel: xgboost\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top-level settings"):
        ExperimentConfig.from_yaml(path)


def test_yaml_round_trips_through_dict():
    original = ExperimentConfig.from_yaml(CONFIGS / "dummy.yaml")
    assert ExperimentConfig.from_dict(original.to_dict()).to_dict() == original.to_dict()


# ---------------------------------------------------------------------------
# The Week 1 gate
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gate_record():
    """One constant-negative run over the real AI4I data, 3x2 folds for speed."""
    config = ExperimentConfig.from_yaml(CONFIGS / "dummy.yaml").with_(
        cv=CVConfig(n_splits=3, n_repeats=2, random_state=42)
    )
    record, path = run(config, write=False)
    assert path is None
    return record


def test_gate_constant_negative_scores_the_base_rate(gate_record):
    """Harness validation: recall 0, PR-AUC = Brier = base rate (0.0339).

    Any other numbers mean the harness is broken, not that the model is poor.
    """
    summary = gate_record.metrics["summary"]

    assert summary["recall"]["mean"] == 0.0
    assert summary["f2"]["mean"] == 0.0
    assert summary["pr_auc"]["mean"] == pytest.approx(0.0339, abs=5e-4)
    assert summary["brier"]["mean"] == pytest.approx(0.0339, abs=5e-4)
    assert summary["base_rate"]["mean"] == pytest.approx(0.0339, abs=5e-4)


def test_gate_record_is_complete(gate_record):
    payload = gate_record.to_dict()
    assert payload["config"]["estimator"] == "dummy_constant_negative"
    assert payload["seeds"] == [0, 1, 2, 3, 4]
    assert payload["git"]["sha"] is not None
    assert payload["metrics"]["n_fits"] == 6


def test_gate_writes_a_results_json(tmp_path):
    config = ExperimentConfig.from_yaml(CONFIGS / "dummy.yaml").with_(
        cv=CVConfig(n_splits=3, n_repeats=1, random_state=42),
        paths=ExperimentConfig().paths.__class__(
            repo_root=ExperimentConfig().paths.repo_root,
            data_raw=ExperimentConfig().paths.data_raw,
            reports_dir=tmp_path / "reports",
            results_dir=tmp_path / "results",
        ),
    )
    _, path = run(config, write=True)
    assert path is not None and path.exists()
    assert path.parent == tmp_path / "results"


# ---------------------------------------------------------------------------
# Layer 3 on real data -- pins a finding documented in models/trees/tree.py
# ---------------------------------------------------------------------------
def test_balanced_tree_ranks_well_but_is_worse_calibrated_than_the_dummy():
    """class_weight='balanced' pushes probabilities away from the true base
    rate the same way SMOTE does (see tree.py's docstring) -- measured here as
    Brier WORSE than the trivial constant-negative baseline (0.0339), despite
    a PR-AUC far above it. Not a bug; a documented, load-bearing caveat about
    what this model's raw predict_proba() does and does not mean. If this
    regresses, the reasoning in tree.py needs revisiting, not just this test.
    """
    config = ExperimentConfig(estimator="decision_tree").with_(cv=CVConfig(n_splits=5, n_repeats=1))
    record, _ = run(config, write=False)
    summary = record.metrics["summary"]

    assert summary["pr_auc"]["mean"] > 0.6
    assert summary["brier"]["mean"] > 0.0339  # worse than the base-rate dummy


def test_random_forest_ranks_well_and_is_better_calibrated_than_the_tree():
    """Averaging many trees smooths leaf probabilities back toward something
    realistic -- an emergent property, not a design choice -- so Random
    Forest does not carry the single tree's calibration caveat.
    """
    config = ExperimentConfig(estimator="random_forest").with_(cv=CVConfig(n_splits=5, n_repeats=1))
    record, _ = run(config, write=False)
    summary = record.metrics["summary"]

    assert summary["pr_auc"]["mean"] > 0.6
    assert summary["brier"]["mean"] < 0.0339  # better than the base-rate dummy


def test_adaboost_is_not_broken_by_double_reweighting():
    """Regression test for a real bug: an earlier version of boosting.py put
    class_weight='balanced' on AdaBoost's base stump, which compounds with
    AdaBoost's own adaptive sample reweighting every round instead of just
    the first. Measured effect: PR-AUC 0.17 on real data (barely above the
    0.0339 base rate) versus ~0.78 once fixed to a one-time initial
    sample_weight. This pins the fixed, healthy range so a regression to the
    old behaviour fails loudly here rather than being noticed as "boosting
    doesn't help" months later.

    Also pins the (real, literature-documented) flip side: AdaBoost's SAMME
    probabilities are notoriously poorly calibrated -- Brier is expected to
    be the worst of any Layer 3 model, not a sign this is still broken.
    """
    config = ExperimentConfig(estimator="adaboost").with_(cv=CVConfig(n_splits=5, n_repeats=1))
    record, _ = run(config, write=False)
    summary = record.metrics["summary"]

    assert summary["pr_auc"]["mean"] > 0.6
    assert summary["brier"]["mean"] > 0.1
