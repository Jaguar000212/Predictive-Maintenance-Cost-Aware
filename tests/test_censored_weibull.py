"""Censored Weibull MLE tests.

The likelihood is the project's only hand-written MLE content, so the bar
here is higher than "does it run": these tests pin the formula against a
hand-computed value, check it against an independent oracle (`scipy.stats`)
on the uncensored special case, and verify it recovers known parameters under
simulated censoring -- a miniature version of the Week 2 validation protocol
in CLAUDE.md (simulate censoring at a known cutoff, check recovery).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from pdm.config import WeibullMLEConfig
from pdm.loaders import cmapss_lifetimes
from pdm.models.mle import censored_weibull as cw
from pdm.models.mle.censored_weibull import CensoredWeibullMLE, WeibullDistribution


# ---------------------------------------------------------------------------
# The likelihood itself, against a hand-computed value
# ---------------------------------------------------------------------------
def test_neg_log_likelihood_matches_hand_computation():
    """One failure at t=5, one censored at t=8, beta=2, eta=10.

    f(5) = (2/10) * (5/10)^1 * exp(-(5/10)^2) = 0.1 * exp(-0.25)
    S(8) = exp(-(8/10)^2) = exp(-0.64)
    -log L = -[log f(5) + log S(8)]
    """
    durations = np.array([5.0, 8.0])
    events = np.array([1, 0])
    log_params = np.log([2.0, 10.0])

    f_5 = 0.1 * math.exp(-0.25)
    log_s_8 = -0.64
    expected = -(math.log(f_5) + log_s_8)

    assert cw._neg_log_likelihood(log_params, durations, events) == pytest.approx(expected, rel=1e-10)


def test_neg_log_likelihood_all_failures_reduces_to_plain_density():
    """With no censoring, every term should be a plain log-density."""
    durations = np.array([3.0, 6.0, 9.0])
    events = np.array([1, 1, 1])
    beta, eta = 2.5, 12.0

    expected = -np.sum(
        np.log(beta) - np.log(eta) + (beta - 1) * np.log(durations / eta) - (durations / eta) ** beta
    )
    got = cw._neg_log_likelihood(np.log([beta, eta]), durations, events)
    assert got == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Parameter recovery under simulation
# ---------------------------------------------------------------------------
def test_recovers_known_parameters_from_uncensored_data():
    rng = np.random.default_rng(0)
    true_beta, true_eta = 3.0, 100.0
    durations = true_eta * rng.weibull(true_beta, size=2000)
    events = np.ones_like(durations, dtype=int)

    fit = CensoredWeibullMLE().fit(durations, events)

    assert fit.beta_ == pytest.approx(true_beta, rel=0.05)
    assert fit.eta_ == pytest.approx(true_eta, rel=0.05)
    assert fit.converged_ is True


def test_recovers_known_parameters_under_simulated_type_i_censoring():
    """Simulate censoring at a known cutoff and check the estimator recovers
    the true, uncensored-generating parameters -- not the biased-short
    estimate a naive fit (treating censored rows as failures) would produce.
    """
    rng = np.random.default_rng(1)
    true_beta, true_eta = 4.9, 224.0
    cutoff = 220.0  # chosen to censor a meaningful fraction, not all or none

    true_durations = true_eta * rng.weibull(true_beta, size=3000)
    events = (true_durations <= cutoff).astype(int)
    observed = np.minimum(true_durations, cutoff)

    censored_fraction = 1 - events.mean()
    assert 0.2 < censored_fraction < 0.8  # sanity: the scenario actually censors something

    fit = CensoredWeibullMLE().fit(observed, events)

    assert fit.beta_ == pytest.approx(true_beta, rel=0.1)
    assert fit.eta_ == pytest.approx(true_eta, rel=0.1)

    # The naive "treat every censored row as if it failed at the cutoff" fit
    # is what a broken implementation degenerates to -- it must NOT match.
    naive = CensoredWeibullMLE().fit(observed, np.ones_like(events))
    assert naive.eta_ < fit.eta_


def test_agrees_with_scipy_stats_on_the_uncensored_special_case():
    """Independent oracle: with no censoring, our MLE and scipy.stats' should
    agree, since both are maximising the same likelihood.
    """
    rng = np.random.default_rng(2)
    durations = 50.0 * rng.weibull(2.2, size=1500)
    events = np.ones_like(durations, dtype=int)

    ours = CensoredWeibullMLE().fit(durations, events)
    oracle_beta, _, oracle_eta = stats.weibull_min.fit(durations, floc=0)

    assert ours.beta_ == pytest.approx(oracle_beta, rel=0.02)
    assert ours.eta_ == pytest.approx(oracle_eta, rel=0.02)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_rejects_non_positive_durations():
    with pytest.raises(ValueError, match="strictly positive"):
        CensoredWeibullMLE().fit(np.array([5.0, 0.0]), np.array([1, 1]))


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        CensoredWeibullMLE().fit(np.array([5.0, 6.0]), np.array([1]))


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="no observations"):
        CensoredWeibullMLE().fit(np.array([]), np.array([]))


def test_rejects_non_binary_events():
    with pytest.raises(ValueError, match="0 or 1"):
        CensoredWeibullMLE().fit(np.array([5.0, 6.0]), np.array([1, 2]))


def test_rejects_all_censored_data():
    """No failures means eta has no finite maximiser -- must raise, not
    silently report whatever bound the optimiser happened to stop at.
    """
    with pytest.raises(ValueError, match="no observed failures"):
        CensoredWeibullMLE().fit(np.array([5.0, 6.0, 7.0]), np.array([0, 0, 0]))


def test_unconverged_fit_raises_rather_than_returning_a_silent_answer(monkeypatch):
    fake_result = SimpleNamespace(success=False, message="forced failure", x=np.array([0.0, 0.0]), fun=0.0)
    monkeypatch.setattr(cw, "minimize", lambda *args, **kwargs: fake_result)

    with pytest.raises(RuntimeError, match="did not converge"):
        CensoredWeibullMLE().fit(np.array([5.0, 6.0]), np.array([1, 1]))


def test_predict_distribution_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        CensoredWeibullMLE().predict_distribution()


def test_config_rejects_bad_settings():
    with pytest.raises(ValueError, match="max_iterations"):
        WeibullMLEConfig(max_iterations=0)
    with pytest.raises(ValueError, match="tolerance"):
        WeibullMLEConfig(tolerance=0.0)


# ---------------------------------------------------------------------------
# WeibullDistribution, against hand-computed values
# ---------------------------------------------------------------------------
def test_distribution_survival_and_quantile_are_inverses():
    dist = WeibullDistribution(beta=2.0, eta=10.0)
    p = 1 - math.exp(-1.0)  # survival(eta) = exp(-1) by construction
    assert dist.survival(10.0) == pytest.approx(math.exp(-1.0))
    assert dist.quantile(p) == pytest.approx(10.0, rel=1e-6)


def test_distribution_mean_matches_gamma_formula():
    dist = WeibullDistribution(beta=2.0, eta=10.0)
    assert dist.mean() == pytest.approx(10.0 * math.gamma(1.5))


def test_quantile_rejects_out_of_range_probability():
    dist = WeibullDistribution(beta=2.0, eta=10.0)
    with pytest.raises(ValueError, match="p must be"):
        dist.quantile(1.0)


# ---------------------------------------------------------------------------
# Real data -- the prediction stated before running
# ---------------------------------------------------------------------------
def test_fits_real_cmapss_train_lifetimes_near_the_predicted_band():
    """The predicted band here is the true MLE, confirmed against
    `scipy.stats.weibull_min.fit` as an independent oracle: beta=4.4087,
    eta=225.03 (see CLAUDE.md's "Verified Weibull fit" note). CLAUDE.md's
    earlier method-of-moments prediction (beta 4.9-5.0) was NOT this
    estimator being wrong -- moment-matching and maximum likelihood are
    different estimation principles that need not agree, and at n=100 they
    diverge by about 14% here. Train trajectories run to failure (event=1
    for all), so this is the uncensored case.
    """
    life = cmapss_lifetimes("FD001")
    train = life[life["split"] == "train"]
    durations = train["duration"].to_numpy(dtype=float)
    events = train["event"].to_numpy(dtype=int)

    fit = CensoredWeibullMLE().fit(durations, events)

    assert fit.converged_ is True
    assert fit.n_observations_ == fit.n_events_ == 100
    assert fit.beta_ == pytest.approx(4.4087, abs=0.01)
    assert fit.eta_ == pytest.approx(225.03, abs=0.5)


def test_fits_real_cmapss_full_table_including_censored_test_units():
    """Fitting the full lifetime table (train + right-censored test units) on
    duration/event alone -- true_duration is never touched -- must still
    converge to a sane, finite answer. This does not check a specific number:
    it exercises the mixed density/survival branch on real data rather than
    only on simulation.
    """
    life = cmapss_lifetimes("FD001")
    durations = life["duration"].to_numpy(dtype=float)
    events = life["event"].to_numpy(dtype=int)

    fit = CensoredWeibullMLE().fit(durations, events)

    assert fit.converged_ is True
    assert fit.beta_ > 0 and math.isfinite(fit.beta_)
    assert fit.eta_ > 0 and math.isfinite(fit.eta_)
