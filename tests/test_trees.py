"""Tree model tests.

These are thin sklearn wrappers -- there is no hand-written math to pin
against a hand computation, the way there is for Weibull or the Laplace
covariance. What matters here is that the settings that make this THE
falsification baseline (max_depth, class_weight) actually reach the fitted
estimator, and that the registry wiring produces a working, fresh-per-fold
pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from pdm.config import AI4ISchema, TreeConfig
from pdm.models import registry
from pdm.models.trees.forest import build_random_forest
from pdm.models.trees.tree import build_depth_limited_tree


def _ai4i_shaped_frame(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "air_temp_k": rng.normal(300, 2, n),
            "process_temp_k": rng.normal(310, 2, n),
            "rot_speed_rpm": rng.normal(1500, 100, n),
            "torque_nm": rng.normal(40, 5, n),
            "tool_wear_min": rng.uniform(0, 250, n),
            "type": rng.choice(["L", "M", "H"], size=n),
        }
    )
    y = np.array([0] * (n - 8) + [1] * 8)
    return df, y


# ---------------------------------------------------------------------------
# TreeConfig guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"depth_limited_max_depth": 0},
        {"depth_limited_min_samples_leaf": 0},
        {"forest_n_estimators": 0},
        {"forest_max_depth": 0},
        {"forest_min_samples_leaf": 0},
    ],
)
def test_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        TreeConfig(**kwargs)


def test_forest_max_depth_none_is_allowed():
    """None is the deliberate default (see TreeConfig docstring) -- must not
    be rejected by the same guard that catches max_depth=0.
    """
    TreeConfig(forest_max_depth=None)


# ---------------------------------------------------------------------------
# Depth-limited tree
# ---------------------------------------------------------------------------
def test_depth_limited_tree_uses_the_configured_settings():
    config = TreeConfig(depth_limited_max_depth=3, depth_limited_min_samples_leaf=7, random_state=99)
    tree = build_depth_limited_tree(config)

    assert isinstance(tree, DecisionTreeClassifier)
    assert tree.max_depth == 3
    assert tree.min_samples_leaf == 7
    assert tree.class_weight == "balanced"
    assert tree.random_state == 99


def test_depth_limited_tree_actually_respects_the_depth_cap_after_fitting():
    """The setting must reach the fitted tree, not just the constructor."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    tree = build_depth_limited_tree(TreeConfig(depth_limited_max_depth=2)).fit(X, y)
    assert tree.get_depth() <= 2


def test_class_weight_balanced_changes_the_fit_on_imbalanced_data():
    rng = np.random.default_rng(2)
    n_neg, n_pos = 480, 20
    X = np.vstack([rng.normal(0, 1, (n_neg, 2)), rng.normal(1.5, 1, (n_pos, 2))])
    y = np.array([0] * n_neg + [1] * n_pos)

    balanced = build_depth_limited_tree(TreeConfig()).fit(X, y)
    unbalanced = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42).fit(X, y)

    # A balanced tree should flag meaningfully more positives as likely
    # failures than the same tree fit without reweighting the minority class.
    assert balanced.predict(X).sum() > unbalanced.predict(X).sum()


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
def test_random_forest_uses_the_configured_settings():
    config = TreeConfig(forest_n_estimators=50, forest_max_depth=6, forest_min_samples_leaf=3, random_state=7)
    forest = build_random_forest(config)

    assert isinstance(forest, RandomForestClassifier)
    assert forest.n_estimators == 50
    assert forest.max_depth == 6
    assert forest.min_samples_leaf == 3
    assert forest.class_weight == "balanced"
    assert forest.random_state == 7


def test_random_forest_default_max_depth_is_unrestricted():
    """See TreeConfig's docstring: bagging, not per-tree depth, is the
    variance control here -- unlike the depth-limited baseline.
    """
    forest = build_random_forest(TreeConfig())
    assert forest.max_depth is None


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["decision_tree", "random_forest"])
def test_registered_pipeline_runs_end_to_end_on_ai4i_shaped_data(name):
    df, y = _ai4i_shaped_frame()
    factory = registry.build(name, AI4ISchema(), seed=0)
    pipeline = factory()
    pipeline.fit(df, y)
    proba = pipeline.predict_proba(df)

    assert proba.shape == (len(df), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(pipeline.named_steps["classifier"].classes_) == [0, 1]


@pytest.mark.parametrize("name", ["decision_tree", "random_forest"])
def test_registered_pipeline_is_fresh_per_factory_call(name):
    factory = registry.build(name, AI4ISchema(), seed=0)
    first = factory().named_steps["classifier"]
    second = factory().named_steps["classifier"]
    assert first is not second
