"""Gaussian Naive Bayes for AI4I -- Layer 2.

Naive Bayes' defining assumption is that features are conditionally
independent given the class: P(x | y) = prod_i P(x_i | y). That assumption is
what licenses modelling each feature with whichever distribution actually
fits it and simply multiplying (summing, in log space) the per-feature
likelihoods -- there is no requirement that every feature share one
distributional family.

AI4I's features are not all the same *kind* of variable: five are continuous
sensor/physics readings, and one -- `type` (L/M/H) -- is a three-level
category. One-hot-encoding `type` and handing it to plain `GaussianNB` fits a
bell curve to a column of 0s and 1s, a real if minor misspecification.
`MixedNaiveBayes` instead models the continuous block with `GaussianNB` and
`type` with `CategoricalNB`, and combines their log-likelihoods by hand --
which naive independence makes exact, not approximate.

**Why this uses the true class prior, not `class_weight='balanced'`.**
CLAUDE.md's locked imbalance decision targets Layer 3's discriminative
classifiers, whose training loss needs rebalancing so the 3.4% minority class
isn't ignored. This layer exists to produce CALIBRATED probabilities --
Brier score is a locked primary metric specifically because it catches
miscalibration -- and reweighting toward a 50/50 prior would bias every
probability away from the true base rate, the same failure mode CLAUDE.md
rejects SMOTE for, just reached through the prior instead of resampling. See
`BayesianConfig` for the toggle and full rationale.

**A predictable failure mode worth stating before running this.** `power_w`
is an exact function of `torque_nm` and `rot_speed_rpm`, both of which remain
in the feature matrix; `temp_diff` is similarly derived from `air_temp_k` and
`process_temp_k`. Naive independence treats every feature as separate
evidence, so including a feature alongside its own derivation triple- (and
double-) counts one physical signal, pushing predicted probabilities further
from 0.5 than the evidence actually supports. Expect this model to rank
reasonably (PR-AUC) but calibrate worse (Brier) than the logistic regression
beside it -- for exactly this reason, plus the independence assumption itself
not holding for physically correlated sensors.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import OrdinalEncoder

from ...config import AI4ISchema, BayesianConfig

# Verified data fact (see loaders.py's AI4ILoader._validate and CLAUDE.md):
# `type` has exactly these three levels. Fixed explicitly rather than
# inferred per fold, so a fold missing the rarest tier (H) still encodes
# consistently instead of silently shifting what category index 2 means.
_TYPE_LEVELS = ("L", "M", "H")


class MixedNaiveBayes(BaseEstimator, ClassifierMixin):
    """Naive Bayes with a Gaussian likelihood for continuous features and a
    categorical likelihood for `type`, combined by naive independence.

    Expects `X` as a DataFrame carrying the schema's raw numeric columns plus
    the physics-engineered ones (`temp_diff`, `power_w`, `wear_strain`) and
    `type` by name -- i.e. the output of `PhysicsFeatures`, not a one-hot- or
    column-transformed array. That is deliberate: `CategoricalNB` needs the
    category labels, not a one-hot encoding of them.
    """

    def __init__(self, schema: AI4ISchema | None = None, config: BayesianConfig | None = None) -> None:
        self.schema = schema or AI4ISchema()
        self.config = config or BayesianConfig()

    def fit(self, X, y) -> MixedNaiveBayes:
        s = self.schema
        self._continuous_cols = list(s.numeric_features) + list(s.engineered_features)
        self._cat_col = s.categorical_features[0]

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if list(self.classes_) != [0, 1]:
            raise ValueError(f"MixedNaiveBayes expects binary {{0, 1}} labels, got {list(self.classes_)}")

        # The true empirical prior -- see the module docstring for why this is
        # not class_weight='balanced'.
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        self.class_log_prior_ = np.log(counts / counts.sum())

        self._gnb = GaussianNB(var_smoothing=self.config.gnb_var_smoothing)
        self._gnb.fit(X[self._continuous_cols], y)

        self._encoder = OrdinalEncoder(categories=[list(_TYPE_LEVELS)])
        cat_encoded = self._encoder.fit_transform(X[[self._cat_col]]).astype(int).ravel()
        self._cnb = CategoricalNB(alpha=self.config.cnb_alpha)
        self._cnb.fit(cat_encoded.reshape(-1, 1), y)

        return self

    def _log_joint_likelihood(self, X) -> np.ndarray:
        """log P(y=k) + log P(x_continuous | y=k) + log P(type | y=k), per row per class."""
        theta, var = self._gnb.theta_, self._gnb.var_  # (n_classes, n_continuous_features)
        Xc = X[self._continuous_cols].to_numpy(dtype=float)
        log_gauss = np.stack(
            [
                -0.5 * np.sum(np.log(2 * np.pi * var[k]) + (Xc - theta[k]) ** 2 / var[k], axis=1)
                for k in range(len(self.classes_))
            ],
            axis=1,
        )

        cat_encoded = self._encoder.transform(X[[self._cat_col]]).astype(int).ravel()
        # feature_log_prob_[0]: shape (n_classes, n_categories) for the one categorical feature.
        log_cat = self._cnb.feature_log_prob_[0][:, cat_encoded].T

        return self.class_log_prior_[np.newaxis, :] + log_gauss + log_cat

    def predict_proba(self, X) -> np.ndarray:
        joint = self._log_joint_likelihood(X)
        log_proba = joint - logsumexp(joint, axis=1, keepdims=True)
        return np.exp(log_proba)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
