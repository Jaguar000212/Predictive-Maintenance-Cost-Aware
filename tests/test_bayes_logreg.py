"""Bayesian logistic regression tests.

Two things get pinned here beyond "does it run": the Laplace covariance
(hand-computed Hessian, since that formula is the genuinely new content), and
the limit behaviour of the uncertainty correction -- as posterior variance
goes to zero it must collapse back to the plain MAP-point sigmoid, and for
nonzero variance it must always pull probabilities toward 0.5, never away
from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from pdm.config import AI4ISchema, BayesianConfig
from pdm.models import registry
from pdm.models.bayes.bayes_logreg import BayesianLogisticRegression


def _separable_data(seed: int = 3, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    true_w = np.array([1.5, -2.0, 0.5])
    p = expit(X @ true_w)
    y = (rng.random(n) < p).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# The MAP point -- must match plain sklearn logistic regression exactly
# ---------------------------------------------------------------------------
def test_map_point_matches_plain_sklearn_logistic_regression():
    X, y = _separable_data()
    model = BayesianLogisticRegression().fit(X, y)

    oracle = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    oracle.fit(X, y)
    oracle_w = np.concatenate([oracle.intercept_, oracle.coef_.ravel()])

    np.testing.assert_allclose(model.w_map_, oracle_w, rtol=1e-6)


def test_balanced_weight_config_changes_the_map_fit():
    """The true-prior-by-default decision (see module docstring) should be
    visible in the fitted weights if flipped.
    """
    rng = np.random.default_rng(4)
    n_neg, n_pos = 180, 20
    X = np.vstack([rng.normal(0, 1, (n_neg, 2)), rng.normal(2, 1, (n_pos, 2))])
    y = np.array([0] * n_neg + [1] * n_pos)

    default = BayesianLogisticRegression(BayesianConfig(use_class_weight_balanced=False)).fit(X, y)
    balanced = BayesianLogisticRegression(BayesianConfig(use_class_weight_balanced=True)).fit(X, y)

    assert not np.allclose(default.w_map_, balanced.w_map_)


# ---------------------------------------------------------------------------
# The Laplace covariance, against a hand-computed Hessian
# ---------------------------------------------------------------------------
def test_covariance_matches_the_hand_computed_hessian():
    X = np.array([[1.0], [2.0], [-1.0], [-2.0], [0.5]])
    y = np.array([1, 1, 0, 0, 1])
    config = BayesianConfig(logreg_C=2.0)

    model = BayesianLogisticRegression(config).fit(X, y)

    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    p = expit(Xb @ model.w_map_)
    s = p * (1 - p)
    expected_hessian = (Xb * s[:, None]).T @ Xb
    expected_hessian += np.diag([0.0] + [1.0 / config.logreg_C] * d)

    np.testing.assert_allclose(np.linalg.inv(model.covariance_), expected_hessian, rtol=1e-8)


# ---------------------------------------------------------------------------
# The uncertainty correction: limit behaviour and direction
# ---------------------------------------------------------------------------
def test_predict_proba_reduces_to_map_sigmoid_when_variance_is_zero():
    """As the posterior collapses to a point, the probit correction must
    collapse back to the plain MAP-point prediction -- kappa -> 1.
    """
    X, y = _separable_data()
    model = BayesianLogisticRegression().fit(X, y)
    model.covariance_ = np.zeros_like(model.covariance_)  # force the confident-posterior limit

    mean_logit, var_logit = model._logit_moments(X)
    assert np.allclose(var_logit, 0.0)

    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba[:, 1], expit(mean_logit), rtol=1e-10)


def test_uncertainty_correction_pulls_probabilities_toward_half():
    X, y = _separable_data()
    model = BayesianLogisticRegression().fit(X, y)

    mean_logit, var_logit = model._logit_moments(X)
    assert np.any(var_logit > 0)  # the correction has something to do

    map_only = expit(mean_logit)
    corrected = model.predict_proba(X)[:, 1]

    above = map_only > 0.5
    below = map_only < 0.5
    assert np.all(corrected[above] <= map_only[above] + 1e-12)
    assert np.all(corrected[below] >= map_only[below] - 1e-12)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_rejects_non_binary_labels():
    X = np.array([[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="binary"):
        BayesianLogisticRegression().fit(X, np.array([0, 1, 2]))


def test_predict_proba_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        BayesianLogisticRegression().predict_proba(np.array([[1.0]]))


# ---------------------------------------------------------------------------
# Registry integration -- through PhysicsFeatures, scaling, and one-hot type
# ---------------------------------------------------------------------------
def test_registered_pipeline_runs_end_to_end_on_ai4i_shaped_data():
    rng = np.random.default_rng(2)
    n = 40
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
    y = np.array([0] * (n - 6) + [1] * 6)

    factory = registry.build("bayes_logreg", AI4ISchema(), seed=0)
    pipeline = factory()
    pipeline.fit(df, y)
    proba = pipeline.predict_proba(df)

    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(pipeline.named_steps["classifier"].classes_) == [0, 1]
