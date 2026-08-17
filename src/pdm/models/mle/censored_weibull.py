"""Censored Weibull MLE -- Layer 1.

This is the project's only genuine maximum-likelihood-estimation content: the
log-likelihood below is written out explicitly rather than delegated to a
survival-analysis package, because writing it is the point of the exercise.

**Why a failure and a censored observation contribute differently.** A Weibull
distribution has a shape parameter beta (how sharply failure risk rises with
age) and a scale eta (the characteristic lifetime). For a unit observed to
fail at time t, the data says "it died near here" -- that is the density,
f(t). For a unit still running when observation stopped at time t, the data
says only "it survived at least this long" -- that is the survival function,
S(t) = 1 - CDF(t). Treating every observation as a failure (using f(t)
everywhere) biases every fitted lifetime short, and does so silently: the
optimiser still converges, it just converges to the wrong answer, because it
was handed the wrong question.

    f(t)     = (beta/eta) * (t/eta)^(beta-1) * exp(-(t/eta)^beta)
    S(t)     = exp(-(t/eta)^beta)
    log L    = sum_{failed}   log f(t_i)
             + sum_{censored} log S(t_i)

`CensoredWeibullMLE.fit` maximises this (minimises its negative) over
(beta, eta) with `scipy.optimize.minimize`. The search runs in log-space for
both parameters -- optimising (log_beta, log_eta) rather than (beta, eta)
directly -- because beta and eta must stay strictly positive, and an
unconstrained optimiser exploring negative values would hand the likelihood
nonsense (log of a negative number) rather than failing cleanly.

Validation against simulated censoring (does this recover known parameters
within tolerance?) lives in `tests/test_censored_weibull.py`, not here --
this module is the estimator, not its own proof.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ...config import WeibullMLEConfig


@dataclass(frozen=True)
class WeibullDistribution:
    """A fitted Weibull(beta, eta), with the queries Layer 1 needs.

    Immutable and separate from `CensoredWeibullMLE` so a fitted distribution
    can be passed around (into a policy simulation, say) without carrying the
    optimiser's fitting machinery along with it.
    """

    beta: float
    eta: float

    def survival(self, t: np.ndarray | float) -> np.ndarray:
        """P(lifetime > t)."""
        t = np.asarray(t, dtype=float)
        return np.exp(-((t / self.eta) ** self.beta))

    def cdf(self, t: np.ndarray | float) -> np.ndarray:
        return 1.0 - self.survival(t)

    def pdf(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return (self.beta / self.eta) * (t / self.eta) ** (self.beta - 1.0) * self.survival(t)

    def hazard(self, t: np.ndarray | float) -> np.ndarray:
        """Instantaneous failure rate given survival to t: f(t) / S(t)."""
        t = np.asarray(t, dtype=float)
        return (self.beta / self.eta) * (t / self.eta) ** (self.beta - 1.0)

    def quantile(self, p: float) -> float:
        """The age at which a fraction p of units are expected to have failed.

        This is the age-based maintenance interval Layer 1 produces: e.g.
        `quantile(0.1)` is the age by which 10% of units have failed, a
        common "replace before this age" policy input.
        """
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        return self.eta * (-math.log(1.0 - p)) ** (1.0 / self.beta)

    def mean(self) -> float:
        return self.eta * math.gamma(1.0 + 1.0 / self.beta)


def _neg_log_likelihood(log_params: np.ndarray, durations: np.ndarray, events: np.ndarray) -> float:
    """Negative log-likelihood of a Weibull(beta, eta) given censored data.

    `log_params` is `(log(beta), log(eta))` -- see the module docstring for
    why the search happens in log-space. `(t/eta)^beta` is computed as
    `exp(beta * (log(t) - log(eta)))` rather than a direct power, so a large
    beta or a t far from eta overflows `exp` (caught below) instead of
    silently overflowing a `**` into `inf` with no warning.
    """
    log_beta, log_eta = log_params
    beta = math.exp(log_beta)

    log_t_over_eta = np.log(durations) - log_eta
    with np.errstate(over="ignore"):
        pow_term = np.exp(beta * log_t_over_eta)  # (t / eta) ** beta

    log_pdf = math.log(beta) - log_eta + (beta - 1.0) * log_t_over_eta - pow_term
    log_survival = -pow_term
    log_lik = np.where(events == 1, log_pdf, log_survival)

    nll = -np.sum(log_lik)
    if not np.isfinite(nll):
        # A degenerate corner of the search space (e.g. beta driven huge by a
        # bad step) should push the optimiser back out, not crash it.
        return 1e12
    return float(nll)


def _initial_guess(durations: np.ndarray) -> tuple[float, float]:
    """A crude, censoring-naive starting point -- not an estimate in its own right.

    beta=1 is the exponential special case; eta0 is simply the median duration.
    Both are wrong in general (censored durations are treated as if they were
    failures), but the optimiser only needs a finite, right-order-of-magnitude
    place to start.
    """
    eta0 = max(float(np.median(durations)), 1e-3)
    return 1.0, eta0


class CensoredWeibullMLE:
    """Fits a Weibull(beta, eta) to right-censored lifetime data.

    Interface: `fit(durations, events)` then `predict_distribution()`, per the
    project's lifetime-model contract -- this does not expose `predict_proba`,
    that is the classifiers' interface (Layers 2-3).
    """

    def __init__(self, config: WeibullMLEConfig | None = None) -> None:
        self.config = config or WeibullMLEConfig()
        self.beta_: float | None = None
        self.eta_: float | None = None
        self.converged_: bool | None = None
        self.log_likelihood_: float | None = None
        self.n_observations_: int | None = None
        self.n_events_: int | None = None

    def _validate(self, durations: np.ndarray, events: np.ndarray) -> None:
        if durations.shape != events.shape:
            raise ValueError(
                f"durations and events must have the same shape, got {durations.shape} and {events.shape}"
            )
        if durations.size == 0:
            raise ValueError("no observations supplied")
        if not np.all(durations > 0):
            raise ValueError("all durations must be strictly positive (Weibull is undefined at t <= 0)")
        if not np.all(np.isin(events, (0, 1))):
            raise ValueError(f"events must be 0 or 1, got values {sorted(set(events.tolist()))}")
        if not np.any(events == 1):
            # With no failures, every term is a survival term -log S(t) =
            # (t/eta)^beta, which is maximised (driven toward zero) by sending
            # eta to infinity -- there is no finite maximiser, so the "fit"
            # would silently report whatever bound the optimiser happened to
            # stop at rather than a genuine MLE.
            raise ValueError(
                "no observed failures (all events are 0) -- the likelihood has no finite "
                "maximiser in that case, since eta can grow without bound"
            )

    def fit(self, durations: np.ndarray, events: np.ndarray) -> CensoredWeibullMLE:
        durations = np.asarray(durations, dtype=float)
        events = np.asarray(events, dtype=int)
        self._validate(durations, events)

        x0 = np.log(_initial_guess(durations))
        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(durations, events),
            method=self.config.optimizer_method,
            tol=self.config.tolerance,
            options={"maxiter": self.config.max_iterations},
        )

        if not result.success:
            raise RuntimeError(
                f"Weibull MLE did not converge ({result.message}). Accepting an unconverged "
                "fit would report parameters that are not the actual maximum-likelihood "
                "estimate; raise the iteration cap in WeibullMLEConfig or inspect the data."
            )

        self.beta_, self.eta_ = np.exp(result.x)
        self.converged_ = bool(result.success)
        self.log_likelihood_ = float(-result.fun)
        self.n_observations_ = int(durations.size)
        self.n_events_ = int(np.sum(events))
        return self

    def _check_is_fitted(self) -> None:
        if self.beta_ is None:
            raise RuntimeError("CensoredWeibullMLE is not fitted yet -- call fit() first")

    def predict_distribution(self) -> WeibullDistribution:
        """Return the fitted distribution as an immutable value object."""
        self._check_is_fitted()
        assert self.beta_ is not None and self.eta_ is not None
        return WeibullDistribution(beta=self.beta_, eta=self.eta_)
