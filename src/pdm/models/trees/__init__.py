"""Tree-based classifiers (Layer 3)."""

from __future__ import annotations

from .boosting import (
    BalancedAdaBoost,
    BalancedGradientBoosting,
    BalancedXGBClassifier,
    build_adaboost,
    build_gradient_boosting,
    build_xgboost,
)
from .forest import build_random_forest
from .tree import build_depth_limited_tree
from .voting import build_soft_voting

__all__ = [
    "BalancedAdaBoost",
    "BalancedGradientBoosting",
    "BalancedXGBClassifier",
    "build_adaboost",
    "build_depth_limited_tree",
    "build_gradient_boosting",
    "build_random_forest",
    "build_soft_voting",
    "build_xgboost",
]
