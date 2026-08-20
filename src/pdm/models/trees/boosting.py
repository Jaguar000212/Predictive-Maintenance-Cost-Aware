"""AdaBoost, Gradient Boosting, and XGBoost -- Layer 3's boosted ensembles.

Where Random Forest trains many trees independently and averages them,
boosting trains them one after another: each new tree is fit to correct
whatever the ensemble built so far still gets wrong, and the final
prediction combines all of them. That sequential correction is what makes
boosting more expressive than bagging -- and what CLAUDE.md's falsification
test is actually about: does that expressiveness beat a fixed, shallow
baseline tree by more than the cross-validation spread once physics
features have already done the easy work of exposing each failure mode as
a near axis-aligned rule?

**Imbalance handling is genuinely different per algorithm here**, because
neither `AdaBoostClassifier` nor `GradientBoostingClassifier` accepts
`class_weight` -- a real gap in sklearn's API, not an inconsistency in this
file:

- **AdaBoost** reweights samples internally as part of boosting itself --
  and that turns out to matter more than it sounds. An earlier version of
  this file put `class_weight='balanced'` on the base stump, by analogy with
  the depth-limited tree. Measured result: PR-AUC 0.17 on real data, versus
  0.83-0.90 for every other Layer 3 model. The base stump's `class_weight`
  reapplies the same fixed 28.5:1 correction on top of AdaBoost's own
  *evolving* sample weights on EVERY round, not just the first -- and by
  round 3 the compounded weights push the weighted training error to exactly
  1.0 (degenerate), silently wasting 197 of 200 rounds. `BalancedAdaBoost`
  instead injects the balanced weighting exactly once, as the ensemble's
  *initial* sample distribution, and lets AdaBoost's own per-round reweighting
  take over unmodified from there -- verified to bring PR-AUC back to 0.76,
  in line with a plain (no reweighting at all) stump. The base estimator
  itself is a plain, unweighted one-level stump, kept this shallow because
  AdaBoost's classic formulation is many weak learners, not `TreeConfig`'s
  own baseline depth.
- **Gradient Boosting** has no reweighting mechanism at all built in, so
  `BalancedGradientBoosting` computes `sample_weight` from `y` at fit time
  (`sklearn.utils.class_weight.compute_sample_weight("balanced", y)`) and
  passes it through explicitly -- computed from each fit call's own labels,
  not the whole dataset, so it stays correctly scoped to whatever fold is
  being fit.
- **XGBoost** has `scale_pos_weight`, which is what CLAUDE.md's locked
  decision names by name ("class_weight='balanced' / scale_pos_weight").
  Unlike sklearn's `class_weight='balanced'` string, it is a plain ratio
  (negative count / positive count) that XGBoost does not compute itself, so
  `BalancedXGBClassifier` computes it at fit time the same way.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ...config import BoostingConfig


class BalancedAdaBoost(AdaBoostClassifier):
    """`AdaBoostClassifier` with the locked imbalance decision applied as a
    ONE-TIME initial sample distribution, not a per-round reweighting.

    See the module docstring for why `class_weight='balanced'` on the base
    estimator is wrong here: it compounds with AdaBoost's own adaptive
    sample weights every round instead of just the first, and measurably
    breaks the ensemble. Injecting the balanced weighting only as the
    starting point -- exactly what `AdaBoostClassifier.fit`'s own
    `sample_weight` argument is for -- lets AdaBoost's normal per-round
    adaptation take over unmodified afterwards.

    No `__init__` override -- `clone()`/`get_params()` behave exactly like
    the parent class.
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


def build_adaboost(config: BoostingConfig | None = None) -> BalancedAdaBoost:
    config = config or BoostingConfig()
    stump = DecisionTreeClassifier(
        max_depth=config.adaboost_stump_max_depth,
        random_state=config.random_state,
    )
    return BalancedAdaBoost(
        estimator=stump,
        n_estimators=config.adaboost_n_estimators,
        learning_rate=config.adaboost_learning_rate,
        random_state=config.random_state,
    )


class BalancedGradientBoosting(GradientBoostingClassifier):
    """`GradientBoostingClassifier` with the locked imbalance decision applied
    via `sample_weight`, since it has no `class_weight` parameter to set.

    No `__init__` override -- `clone()`/`get_params()` behave exactly like
    the parent class, since nothing about construction changes, only `fit`.
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


def build_gradient_boosting(config: BoostingConfig | None = None) -> BalancedGradientBoosting:
    config = config or BoostingConfig()
    return BalancedGradientBoosting(
        n_estimators=config.gb_n_estimators,
        learning_rate=config.gb_learning_rate,
        max_depth=config.gb_max_depth,
        random_state=config.random_state,
    )


class BalancedXGBClassifier(XGBClassifier):
    """`XGBClassifier` with `scale_pos_weight` computed at fit time from that
    call's own labels, rather than left at XGBoost's default of 1 (no
    reweighting) or fixed to a dataset-level constant computed outside the
    fold.
    """

    def fit(self, X, y, **kwargs):
        y_arr = np.asarray(y).astype(int)
        neg, pos = np.bincount(y_arr, minlength=2)
        if pos == 0:
            raise ValueError("BalancedXGBClassifier: no positive-class examples in this fit call")
        self.set_params(scale_pos_weight=neg / pos)
        return super().fit(X, y, **kwargs)


def build_xgboost(config: BoostingConfig | None = None) -> BalancedXGBClassifier:
    config = config or BoostingConfig()
    return BalancedXGBClassifier(
        n_estimators=config.xgboost_n_estimators,
        learning_rate=config.xgboost_learning_rate,
        max_depth=config.xgboost_max_depth,
        random_state=config.random_state,
        eval_metric="logloss",
    )
