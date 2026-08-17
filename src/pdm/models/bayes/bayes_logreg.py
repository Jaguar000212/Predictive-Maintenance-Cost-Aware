"""Bayesian logistic regression via Laplace's approximation -- Layer 2.

Plain logistic regression returns one number per prediction: the probability
implied by a single "best" weight vector. That vector is itself an estimate
from noisy data -- a different sample of machines would fit slightly
different weights. Bayesian logistic regression accounts for that by keeping
a whole distribution over plausible weight vectors instead of one point, and
averaging the prediction over it. A wide posterior means the model is
genuinely unsure which weights are right; a narrow one means it's confident.
That spread is this layer's "+ uncertainty", which Layer 3's point-estimate
trees do not provide.

**No MCMC** (out of scope per CLAUDE.md) -- Laplace's approximation instead,
fully analytic given the MAP point:

1. Find the single most probable weight vector (the MAP estimate) by ordinary
   L2-regularised logistic regression. L2 regularisation with strength `1/C`
   *is* MAP estimation under a Gaussian prior N(0, C) on each weight:
   sklearn's minimised objective, `0.5*||w||^2 + C * sum(log_loss)`, divided
   through by `C`, is `(1/(2C))*||w||^2 + sum(log_loss)` -- a Gaussian-prior
   penalty plus the ordinary likelihood.
2. Approximate the posterior around that point as Gaussian, with covariance
   equal to the inverse Hessian of the negative log-posterior there. For
   logistic regression that Hessian has a closed form,
   `H = X^T diag(p*(1-p)) X + prior_precision`, so no numerical differentiation
   is needed.
3. When predicting, integrate the sigmoid over that Gaussian posterior instead
   of evaluating it only at the MAP point. That integral has no closed form,
   but MacKay's probit approximation is accurate and analytic:
   `sigmoid(mu / sqrt(1 + pi*sigma^2/8))`, where `mu` and `sigma^2` are the
   mean and variance of the logit (`x . w`) induced by the weight posterior.
   As `sigma^2 -> 0` this collapses back to the plain MAP-point sigmoid --
   the sanity check in `tests/test_bayes_logreg.py`.

**Why the true class prior, not `class_weight='balanced'`.** Same reasoning
as `gnb.py`: this model's job is calibration, and reweighting toward 50/50
would push every probability away from the true ~3.4% base rate the way
SMOTE does, just via the loss function instead of resampling.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from ...config import BayesianConfig


class BayesianLogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic regression with a Gaussian weight prior, fit by Laplace's
    approximation.

    Expects an already-numeric, already-scaled 2D input. Scaling matters here
    in a way it does not for trees: an L2 penalty and a shared prior variance
    `C` only mean the same thing across features that are on comparable
    scales -- `wear_strain` runs into the hundreds of thousands, `temp_diff`
    into the tens, and an unscaled fit would regularise them wildly unevenly.
    """

    def __init__(self, config: BayesianConfig | None = None) -> None:
        self.config = config or BayesianConfig()

    def fit(self, X, y) -> BayesianLogisticRegression:
        Xa = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if list(self.classes_) != [0, 1]:
            raise ValueError(
                f"BayesianLogisticRegression expects binary {{0, 1}} labels, got {list(self.classes_)}"
            )

        n, d = Xa.shape
        Xb = np.hstack([np.ones((n, 1)), Xa])  # intercept column, left unregularised below

        class_weight = "balanced" if self.config.use_class_weight_balanced else None
        # penalty is left at its default (l2) rather than passed explicitly --
        # sklearn 1.8+ deprecates the explicit form even when it names the
        # default, ahead of removing it in 1.10.
        map_model = LogisticRegression(
            C=self.config.logreg_C,
            max_iter=self.config.logreg_max_iter,
            class_weight=class_weight,
            solver="lbfgs",
        )
        map_model.fit(Xa, y)
        self.w_map_ = np.concatenate([map_model.intercept_, map_model.coef_.ravel()])

        p = expit(Xb @ self.w_map_)
        s = p * (1.0 - p)
        hessian = (Xb * s[:, None]).T @ Xb

        prior_precision = np.zeros(d + 1)
        prior_precision[1:] = 1.0 / self.config.logreg_C  # intercept unregularised, matching the MAP fit
        hessian = hessian + np.diag(prior_precision)

        self.covariance_ = np.linalg.inv(hessian)
        self._map_model = map_model
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "w_map_"):
            raise RuntimeError("BayesianLogisticRegression is not fitted yet -- call fit() first")

    def _logit_moments(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance of the logit `x . w` under the weight posterior."""
        self._check_is_fitted()
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        Xb = np.hstack([np.ones((n, 1)), Xa])

        mean_logit = Xb @ self.w_map_
        var_logit = np.einsum("ij,jk,ik->i", Xb, self.covariance_, Xb)
        return mean_logit, var_logit

    def predict_proba(self, X) -> np.ndarray:
        """P(y=1|x), integrated over the weight posterior via MacKay's probit
        approximation -- not evaluated only at the single MAP weight vector.
        """
        mean_logit, var_logit = self._logit_moments(X)
        kappa = 1.0 / np.sqrt(1.0 + np.pi * var_logit / 8.0)
        p1 = expit(kappa * mean_logit)
        return np.column_stack([1.0 - p1, p1])

    def predict_logit_stats(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance of the logit for each row -- the uncertainty this
        layer produces beyond a point probability. A wide variance means the
        posterior is genuinely unsure what probability to assign; that is
        different from a probability near 0.5, which means the model is
        confident the outcome itself is a toss-up.
        """
        return self._logit_moments(X)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
