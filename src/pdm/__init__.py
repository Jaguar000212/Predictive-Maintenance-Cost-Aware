"""Cost-aware predictive maintenance.

Layered so each piece can be replaced without touching the others:

    config    frozen, serialisable settings -- depends on nothing
    loaders   file -> validated DataFrame   -- depends on config
    eda       DataFrame -> analysis         -- depends on config only for schemas
    eval      arrays -> metrics             -- depends on config only for settings

Composition (which loader feeds which analysis) happens in `scripts/`, not here.
"""

from .config import (
    AI4ISchema,
    CMAPSSSchema,
    CostConfig,
    CVConfig,
    DeterminismConfig,
    EDAConfig,
    ExperimentConfig,
    MetricConfig,
    PathConfig,
    default_config,
)
from .loaders import (
    AI4ILoader,
    CMAPSSLifetimeBuilder,
    CMAPSSLoader,
    CMAPSSRULLoader,
    DatasetLoader,
    DataValidationError,
    cmapss_lifetimes,
    load_ai4i,
    load_cmapss,
    load_cmapss_rul,
)

__all__ = [
    "AI4ILoader",
    "AI4ISchema",
    "CMAPSSLifetimeBuilder",
    "CMAPSSLoader",
    "CMAPSSRULLoader",
    "CMAPSSSchema",
    "CVConfig",
    "CostConfig",
    "DataValidationError",
    "DatasetLoader",
    "DeterminismConfig",
    "EDAConfig",
    "ExperimentConfig",
    "MetricConfig",
    "PathConfig",
    "cmapss_lifetimes",
    "default_config",
    "load_ai4i",
    "load_cmapss",
    "load_cmapss_rul",
]
