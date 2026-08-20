"""Soft-voting ensemble -- Layer 3's combination of its other five models.

"Soft" voting averages predicted PROBABILITIES across several independently
trained models, rather than averaging their hard 0/1 votes -- so a model
that is 90% confident counts for more than one that is 51% confident. The
five members here are the same decision tree, Random Forest, AdaBoost,
Gradient Boosting, and XGBoost built elsewhere in this layer, constructed
via the exact same `build_*` functions and configs used when each is
registered and run on its own. The point of this model is what averaging
their probabilities does, not a variant of any one of them.

`sklearn.ensemble.VotingClassifier` clones and fits each named estimator
independently inside its own `fit()`, so there is no additional cross-fold
state-sharing risk here beyond what each individual estimator already
carries -- and `registry.py`'s `clone()`-per-fold fix already covers that,
since `VotingClassifier` itself gets cloned fresh by `_physics_pipeline` like
every other classifier reaching it through the registry.
"""

from __future__ import annotations

from sklearn.ensemble import VotingClassifier

from ...config import BoostingConfig, TreeConfig
from .boosting import build_adaboost, build_gradient_boosting, build_xgboost
from .forest import build_random_forest
from .tree import build_depth_limited_tree


def build_soft_voting(
    tree_config: TreeConfig | None = None,
    boosting_config: BoostingConfig | None = None,
) -> VotingClassifier:
    tree_config = tree_config or TreeConfig()
    boosting_config = boosting_config or BoostingConfig()
    return VotingClassifier(
        estimators=[
            ("decision_tree", build_depth_limited_tree(tree_config)),
            ("random_forest", build_random_forest(tree_config)),
            ("adaboost", build_adaboost(boosting_config)),
            ("gradient_boosting", build_gradient_boosting(boosting_config)),
            ("xgboost", build_xgboost(boosting_config)),
        ],
        voting="soft",
    )
