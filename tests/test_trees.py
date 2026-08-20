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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

from pdm.config import AI4ISchema, BoostingConfig, TreeConfig
from pdm.models import registry
from pdm.models.trees.boosting import (
    BalancedAdaBoost,
    BalancedGradientBoosting,
    BalancedXGBClassifier,
    build_adaboost,
    build_gradient_boosting,
    build_xgboost,
)
from pdm.models.trees.forest import build_random_forest
from pdm.models.trees.tree import build_depth_limited_tree
from pdm.models.trees.voting import build_soft_voting

ALL_TREE_MODELS = [
    "decision_tree",
    "random_forest",
    "adaboost",
    "gradient_boosting",
    "xgboost",
    "soft_voting",
]


def _ai4i_shaped_frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature/label pairs with a genuine (not merely positional) signal.

    `y` is thresholded on the same torque*tool_wear product `wear_strain`
    reconstructs, so every classifier here -- including AdaBoost, which
    raises if its base learner is no better than chance -- has something
    real to find. A purely positional label (e.g. "the last few rows are
    positive") would make AdaBoost fail for a reason unrelated to anything
    this file is testing.
    """
    rng = np.random.default_rng(seed)
    tool_wear = rng.uniform(0, 250, n)
    torque = rng.normal(40, 5, n)
    risk = tool_wear * torque
    y = (risk > np.quantile(risk, 0.85)).astype(int)

    df = pd.DataFrame(
        {
            "air_temp_k": rng.normal(300, 2, n),
            "process_temp_k": rng.normal(310, 2, n),
            "rot_speed_rpm": rng.normal(1500, 100, n),
            "torque_nm": torque,
            "tool_wear_min": tool_wear,
            "type": rng.choice(["L", "M", "H"], size=n),
        }
    )
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
# BoostingConfig guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"adaboost_n_estimators": 0},
        {"adaboost_learning_rate": 0.0},
        {"adaboost_stump_max_depth": 0},
        {"gb_n_estimators": 0},
        {"gb_learning_rate": 0.0},
        {"gb_max_depth": 0},
        {"xgboost_n_estimators": 0},
        {"xgboost_learning_rate": 0.0},
        {"xgboost_max_depth": 0},
    ],
)
def test_boosting_config_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        BoostingConfig(**kwargs)


# ---------------------------------------------------------------------------
# AdaBoost
# ---------------------------------------------------------------------------
def test_adaboost_base_stump_has_no_class_weight():
    """The base stump must NOT carry class_weight='balanced'. Measured
    pathology if it does (see boosting.py's module docstring): it compounds
    with AdaBoost's own adaptive reweighting every round, driving weighted
    training error to exactly 1.0 by round 3 on real data (PR-AUC 0.17 vs
    0.76+ for every fix considered) -- silently wasting almost the whole
    ensemble.
    """
    config = BoostingConfig(adaboost_n_estimators=30, adaboost_learning_rate=0.5, adaboost_stump_max_depth=2)
    ada = build_adaboost(config)

    assert isinstance(ada, BalancedAdaBoost)
    assert ada.n_estimators == 30
    assert ada.learning_rate == 0.5
    assert ada.estimator.max_depth == 2
    assert ada.estimator.class_weight is None


def test_adaboost_applies_balanced_weight_only_as_the_initial_distribution():
    """The fix: balanced weighting happens once, as fit()'s sample_weight,
    not every boosting round. An auto-balanced fit and an explicitly-uniform
    fit must produce different ensembles (the balancing had an effect), and
    neither may show the degenerate all-error-1.0 collapse the bug caused.
    """
    df, y = _ai4i_shaped_frame()
    X = df[["torque_nm", "tool_wear_min"]]

    auto_balanced = build_adaboost(BoostingConfig()).fit(X, y)
    explicit_uniform = build_adaboost(BoostingConfig()).fit(X, y, sample_weight=np.ones(len(y)))

    assert not np.allclose(auto_balanced.estimator_errors_, explicit_uniform.estimator_errors_)
    assert np.all(auto_balanced.estimator_errors_ < 1.0)
    assert np.all(explicit_uniform.estimator_errors_ < 1.0)


def test_adaboost_fits_and_predicts_on_separable_data():
    df, y = _ai4i_shaped_frame()
    ada = build_adaboost(BoostingConfig()).fit(df[["torque_nm", "tool_wear_min"]], y)
    proba = ada.predict_proba(df[["torque_nm", "tool_wear_min"]])
    assert proba.shape == (len(df), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------
def test_gradient_boosting_uses_the_configured_settings():
    config = BoostingConfig(gb_n_estimators=40, gb_learning_rate=0.05, gb_max_depth=2, random_state=11)
    gb = build_gradient_boosting(config)

    assert isinstance(gb, BalancedGradientBoosting)
    assert gb.n_estimators == 40
    assert gb.learning_rate == 0.05
    assert gb.max_depth == 2
    assert gb.random_state == 11


def test_gradient_boosting_sample_weight_changes_the_fit_on_imbalanced_data():
    """Mirrors the depth-limited tree's class_weight test: since
    GradientBoostingClassifier has no class_weight, the equivalent check is
    that the balanced sample_weight this wrapper computes actually changes
    the fit relative to unweighted fitting.

    Measured on held-out data, not the training set: with heavily overlapping
    classes, 200 boosting rounds can still drive TRAINING error to ~zero
    either way, which would hide the reweighting effect this test exists to
    catch. A held-out set cannot be memorised, so the decision-boundary shift
    from reweighting toward recall shows up there instead.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(3)
    n_neg, n_pos = 480, 20
    X_all = np.vstack([rng.normal(0, 1.2, (n_neg, 2)), rng.normal(0.8, 1.2, (n_pos, 2))])
    y_all = np.array([0] * n_neg + [1] * n_pos)
    X_train, X_test, y_train, _ = train_test_split(
        X_all, y_all, test_size=0.3, stratify=y_all, random_state=0
    )

    balanced = build_gradient_boosting(BoostingConfig()).fit(X_train, y_train)
    unweighted = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
    ).fit(X_train, y_train)

    assert balanced.predict(X_test).sum() > unweighted.predict(X_test).sum()


def test_gradient_boosting_respects_an_explicitly_passed_sample_weight():
    """An explicit sample_weight must not be silently overridden by the
    automatic balanced computation -- checked by matching a plain
    GradientBoostingClassifier fit with that exact weight, prediction for
    prediction.
    """
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 2))
    y = (rng.random(100) < 0.3).astype(int)
    uniform_weight = np.ones_like(y, dtype=float)

    wrapped = build_gradient_boosting(BoostingConfig()).fit(X, y, sample_weight=uniform_weight)
    plain = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42).fit(
        X, y, sample_weight=uniform_weight
    )

    np.testing.assert_array_equal(wrapped.predict_proba(X), plain.predict_proba(X))


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def test_xgboost_uses_the_configured_settings():
    config = BoostingConfig(xgboost_n_estimators=40, xgboost_learning_rate=0.05, xgboost_max_depth=2)
    xgb = build_xgboost(config)

    assert isinstance(xgb, BalancedXGBClassifier)
    assert xgb.get_params()["n_estimators"] == 40
    assert xgb.get_params()["learning_rate"] == 0.05
    assert xgb.get_params()["max_depth"] == 2


def test_xgboost_computes_scale_pos_weight_from_the_fit_call_labels():
    rng = np.random.default_rng(5)
    n_neg, n_pos = 90, 10
    X = np.vstack([rng.normal(0, 1, (n_neg, 2)), rng.normal(2, 1, (n_pos, 2))])
    y = np.array([0] * n_neg + [1] * n_pos)

    xgb = build_xgboost(BoostingConfig()).fit(X, y)
    assert xgb.get_params()["scale_pos_weight"] == pytest.approx(n_neg / n_pos)


def test_xgboost_rejects_a_fit_call_with_no_positive_examples():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 0])
    with pytest.raises(ValueError, match="no positive-class examples"):
        build_xgboost(BoostingConfig()).fit(X, y)


# ---------------------------------------------------------------------------
# Soft voting
# ---------------------------------------------------------------------------
def test_soft_voting_combines_the_other_five_models():
    voting = build_soft_voting(TreeConfig(), BoostingConfig())

    assert isinstance(voting, VotingClassifier)
    assert voting.voting == "soft"
    names = [name for name, _ in voting.estimators]
    assert names == ["decision_tree", "random_forest", "adaboost", "gradient_boosting", "xgboost"]


def test_soft_voting_fits_and_predicts_on_ai4i_shaped_data():
    df, y = _ai4i_shaped_frame()
    voting = build_soft_voting(TreeConfig(), BoostingConfig()).fit(df[["torque_nm", "tool_wear_min"]], y)
    proba = voting.predict_proba(df[["torque_nm", "tool_wear_min"]])
    assert proba.shape == (len(df), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Registry integration -- all six Layer 3 models
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_TREE_MODELS)
def test_registered_pipeline_runs_end_to_end_on_ai4i_shaped_data(name):
    df, y = _ai4i_shaped_frame()
    factory = registry.build(name, AI4ISchema(), seed=0)
    pipeline = factory()
    pipeline.fit(df, y)
    proba = pipeline.predict_proba(df)

    assert proba.shape == (len(df), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(pipeline.named_steps["classifier"].classes_) == [0, 1]


@pytest.mark.parametrize("name", ALL_TREE_MODELS)
def test_registered_pipeline_is_fresh_per_factory_call(name):
    factory = registry.build(name, AI4ISchema(), seed=0)
    first = factory().named_steps["classifier"]
    second = factory().named_steps["classifier"]
    assert first is not second
