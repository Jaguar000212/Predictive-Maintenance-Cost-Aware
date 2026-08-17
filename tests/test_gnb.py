"""Mixed Naive Bayes tests.

The combination formula (Gaussian likelihood for continuous features,
categorical likelihood for `type`, joined by naive independence) is the
genuinely new content here, so it gets pinned against a hand-recomputation
from the same fitted sklearn parameters -- the same standard applied to the
Weibull likelihood and the physics formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from pdm.config import AI4ISchema
from pdm.models import registry
from pdm.models.bayes.gnb import MixedNaiveBayes


def _tiny_schema() -> AI4ISchema:
    """A one-continuous-feature schema, so the hand-recomputation stays small."""
    return AI4ISchema(numeric_features=("x",), engineered_features=())


def _tiny_frame() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.DataFrame(
        {
            "x": [1.0, 1.2, 1.1, 5.0, 5.5, 5.2],
            "type": ["L", "L", "M", "H", "H", "M"],
        }
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return df, y


# ---------------------------------------------------------------------------
# The combination formula, against a hand recomputation
# ---------------------------------------------------------------------------
def test_log_joint_likelihood_matches_manual_bayes_combination():
    df, y = _tiny_frame()
    model = MixedNaiveBayes(_tiny_schema()).fit(df, y)

    joint = model._log_joint_likelihood(df)

    theta, var = model._gnb.theta_, model._gnb.var_
    cat_encoded = model._encoder.transform(df[["type"]]).astype(int).ravel()
    expected = np.zeros((len(df), 2))
    for i in range(len(df)):
        for k in range(2):
            log_gauss = -0.5 * (
                np.log(2 * np.pi * var[k, 0]) + (df["x"].iloc[i] - theta[k, 0]) ** 2 / var[k, 0]
            )
            log_cat = model._cnb.feature_log_prob_[0][k, cat_encoded[i]]
            expected[i, k] = model.class_log_prior_[k] + log_gauss + log_cat

    np.testing.assert_allclose(joint, expected, rtol=1e-10)


def test_predict_proba_is_the_softmax_of_the_joint_log_likelihood():
    df, y = _tiny_frame()
    model = MixedNaiveBayes(_tiny_schema()).fit(df, y)

    proba = model.predict_proba(df)
    joint = model._log_joint_likelihood(df)
    expected = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))

    np.testing.assert_allclose(proba, expected, rtol=1e-10)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_matches_argmax_of_predict_proba():
    df, y = _tiny_frame()
    model = MixedNaiveBayes(_tiny_schema()).fit(df, y)

    preds = model.predict(df)
    proba = model.predict_proba(df)
    assert np.array_equal(preds, model.classes_[np.argmax(proba, axis=1)])


# ---------------------------------------------------------------------------
# The true-prior decision
# ---------------------------------------------------------------------------
def test_uses_true_empirical_prior_not_a_balanced_one():
    """Layer 2 exists for calibration -- see the module docstring. A 90/10
    split must show up as a 90/10 prior, not 50/50.
    """
    rng = np.random.default_rng(0)
    n_neg, n_pos = 90, 10
    df = pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, n_neg), rng.normal(3, 1, n_pos)]),
            "type": rng.choice(["L", "M", "H"], size=n_neg + n_pos),
        }
    )
    y = np.array([0] * n_neg + [1] * n_pos)

    model = MixedNaiveBayes(_tiny_schema()).fit(df, y)

    assert model.class_log_prior_[0] == pytest.approx(np.log(0.9), abs=1e-9)
    assert model.class_log_prior_[1] == pytest.approx(np.log(0.1), abs=1e-9)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_rejects_non_binary_labels():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "type": ["L", "M", "H"]})
    with pytest.raises(ValueError, match="binary"):
        MixedNaiveBayes(_tiny_schema()).fit(df, np.array([0, 1, 2]))


# ---------------------------------------------------------------------------
# Registry integration -- through PhysicsFeatures, on AI4I-shaped columns
# ---------------------------------------------------------------------------
def test_registered_pipeline_runs_end_to_end_on_ai4i_shaped_data():
    rng = np.random.default_rng(1)
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

    factory = registry.build("gnb", AI4ISchema(), seed=0)
    pipeline = factory()
    pipeline.fit(df, y)
    proba = pipeline.predict_proba(df)

    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(pipeline.named_steps["classifier"].classes_) == [0, 1]
