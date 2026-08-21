"""Config tests.

The point of these is not that dataclasses work -- it is that the guards which
prevent silently-wrong results actually fire.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from pdm.config import (
    AI4ISchema,
    CMAPSSSchema,
    CostConfig,
    CVConfig,
    DeterminismConfig,
    ExperimentConfig,
    MetricConfig,
    default_config,
)


# ---------------------------------------------------------------------------
# Immutability and reproducibility
# ---------------------------------------------------------------------------
def test_config_is_frozen():
    """Mid-run mutation would produce irreproducible results and raise nothing."""
    config = default_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.cv.n_splits = 10


def test_with_returns_a_copy_and_leaves_the_original_alone():
    config = default_config()
    variant = config.with_(cv=CVConfig(n_splits=10))
    assert variant.cv.n_splits == 10
    assert config.cv.n_splits == 5


def test_serialises_to_json_for_the_results_record():
    """Every recorded metric must carry the config that produced it."""
    payload = json.loads(default_config().to_json())
    assert payload["cv"]["random_state"] == 42
    assert payload["determinism"]["deterministic"] == ["HDF", "PWF", "OSF"]
    # Paths must survive as strings, not repr() of Path objects.
    assert isinstance(payload["paths"]["repo_root"], str)


# ---------------------------------------------------------------------------
# The determinism grouping sets the headline ceiling, so it is guarded
# ---------------------------------------------------------------------------
def test_every_failure_mode_must_be_classified():
    """An unclassified mode would silently vanish from the ceiling arithmetic."""
    incomplete = DeterminismConfig(
        deterministic=("HDF", "PWF"), semi_deterministic=("TWF",), stochastic=("RNF",)
    )
    with pytest.raises(ValueError, match="no determinism classification"):
        incomplete.validate_against(AI4ISchema().mode_flags)


def test_a_mode_cannot_be_classified_twice():
    """Double classification would double-count that mode's failures."""
    duplicated = DeterminismConfig(
        deterministic=("HDF", "PWF", "OSF"), semi_deterministic=("TWF", "OSF"), stochastic=("RNF",)
    )
    with pytest.raises(ValueError, match="classified more than once"):
        duplicated.validate_against(AI4ISchema().mode_flags)


def test_unknown_modes_are_rejected():
    typo = DeterminismConfig(deterministic=("HDF", "PWF", "OSF", "XYZ"))
    with pytest.raises(ValueError, match="unknown modes"):
        typo.validate_against(AI4ISchema().mode_flags)


def test_experiment_config_validates_determinism_on_construction():
    with pytest.raises(ValueError, match="no determinism classification"):
        ExperimentConfig(
            determinism=DeterminismConfig(deterministic=(), semi_deterministic=(), stochastic=())
        )


def test_moving_twf_changes_which_modes_count_as_recoverable():
    """The 84.66% / 97.35% choice, expressed as configuration."""
    strict = DeterminismConfig()
    assert set(strict.recoverable_strict) == {"HDF", "PWF", "OSF"}
    assert set(strict.recoverable_extended) == {"HDF", "PWF", "OSF", "TWF"}


# ---------------------------------------------------------------------------
# Feature admissibility
# ---------------------------------------------------------------------------
def test_mode_flags_are_never_admissible_as_features():
    """Leaving them in is leakage that scores ~0.99 and raises nothing."""
    schema = AI4ISchema()
    assert not set(schema.mode_flags) & set(schema.feature_columns)
    assert set(schema.mode_flags) <= set(schema.excluded_from_features)


def test_target_and_identifiers_are_excluded():
    schema = AI4ISchema()
    for column in ("machine_failure", "udi", "product_id"):
        assert column in schema.excluded_from_features
        assert column not in schema.feature_columns


def test_type_stays_a_feature():
    """OSF's threshold is tier-dependent; without `type` its ceiling share is unreachable."""
    assert "type" in AI4ISchema().feature_columns


# ---------------------------------------------------------------------------
# C-MAPSS schema
# ---------------------------------------------------------------------------
def test_cmapss_has_21_sensors_not_26():
    """The NASA readme's 'sensor 1-26' is a typo; cols 1-5 are ids and op settings."""
    schema = CMAPSSSchema()
    assert schema.n_columns == 26
    assert len(schema.sensor_columns) == 21
    assert schema.columns[:5] == ("unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3")
    assert schema.columns[5] == "sensor_1"  # sensor numbering starts at column index 5


def test_subsets_carry_their_own_unit_counts():
    assert CMAPSSSchema.for_subset("FD002").expected_test_units == 259
    assert CMAPSSSchema.for_subset("FD001").expected_train_units == 100


def test_unknown_subset_is_rejected():
    with pytest.raises(ValueError, match="unknown C-MAPSS subset"):
        CMAPSSSchema.for_subset("FD005")


# ---------------------------------------------------------------------------
# Cost constants: decided in docs/DECISIONS.md D11, enforced structurally
# ---------------------------------------------------------------------------
def test_cost_constants_default_to_the_decided_ratio():
    """D11: missed_failure : false_alarm : inspection = 10 : 1 : 0.5."""
    cost = CostConfig()
    assert cost.is_configured
    cost.validate()  # must not raise -- the decision is made, not pending
    assert cost.missed_failure == 10.0
    assert cost.false_alarm == 1.0
    assert cost.inspection == 0.5
    assert cost.ratio == pytest.approx(10.0)


def test_computing_a_cost_before_an_explicit_unset_still_raises():
    """The guard that made 'no cost before the decision' enforceable while the
    ratio was pending still exists -- it now only trips if a field is
    explicitly unset (e.g. by an override or ablation), not by default.
    """
    with pytest.raises(ValueError, match="pending project decision"):
        CostConfig(missed_failure=None).validate()
    with pytest.raises(ValueError, match="pending project decision"):
        _ = CostConfig(false_alarm=None).ratio


def test_configured_costs_expose_the_ratio():
    cost = CostConfig(missed_failure=1000.0, false_alarm=50.0, inspection=10.0)
    assert cost.is_configured
    assert cost.ratio == pytest.approx(20.0)


def test_negative_costs_are_rejected():
    with pytest.raises(ValueError, match="must be non-negative"):
        CostConfig(missed_failure=-1.0, false_alarm=50.0, inspection=10.0).validate()


# ---------------------------------------------------------------------------
# CV and metric settings
# ---------------------------------------------------------------------------
def test_cv_rejects_a_single_split():
    with pytest.raises(ValueError, match="n_splits must be >= 2"):
        CVConfig(n_splits=1)


def test_cv_reports_total_fits():
    assert CVConfig(n_splits=5, n_repeats=5).n_fits == 25


def test_metric_config_rejects_bad_beta_and_thresholds():
    with pytest.raises(ValueError, match="beta must be positive"):
        MetricConfig(beta=0.0)
    with pytest.raises(ValueError, match=r"thresholds must lie in \[0, 1\]"):
        MetricConfig(thresholds=(0.5, 1.7))
    with pytest.raises(ValueError, match="thresholds grid is empty"):
        MetricConfig(thresholds=())
