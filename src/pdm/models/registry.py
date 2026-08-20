"""Estimator registry.

A config names an estimator by string; this maps that string to a factory
producing a fresh, unfitted pipeline. Adding a model is one `@register` entry --
no edits to the runner, and no way for a config to reach an estimator the
registry does not know about.

Only dummies are registered so far. They exist to validate the harness: a
constant-negative baseline must score recall 0 and PR-AUC at the base rate, and
anything else means the harness is broken rather than the model being poor. Real
estimators arrive with Layers 2-3.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import AI4ISchema, BayesianConfig, BoostingConfig, TreeConfig
from ..features.physics import PhysicsFeatures
from .bayes.bayes_logreg import BayesianLogisticRegression
from .bayes.gnb import MixedNaiveBayes
from .trees.boosting import build_adaboost, build_gradient_boosting, build_xgboost
from .trees.forest import build_random_forest
from .trees.tree import build_depth_limited_tree
from .trees.voting import build_soft_voting

EstimatorFactory = Callable[[], Any]
Builder = Callable[[AI4ISchema, int], EstimatorFactory]

_REGISTRY: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    def decorator(builder: Builder) -> Builder:
        if name in _REGISTRY:
            raise ValueError(f"estimator {name!r} is already registered")
        _REGISTRY[name] = builder
        return builder

    return decorator


def available() -> list[str]:
    return sorted(_REGISTRY)


def build(name: str, schema: AI4ISchema, seed: int) -> EstimatorFactory:
    """Return a factory for the named estimator.

    The factory -- not an instance -- is what `CrossValidator.run` needs, so
    every fold gets a fresh unfitted object.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown estimator {name!r}; registered: {available()}")
    return _REGISTRY[name](schema, seed)


def preprocessor(schema: AI4ISchema) -> ColumnTransformer:
    """Minimal admissible-column preprocessing.

    One-hot encodes `type` and passes the numeric sensors through. `type` must
    survive because OSF's threshold is tier-dependent (L/M/H) and its share of
    the recall ceiling is unreachable without it.

    Deliberately NOT included: scaling (tree models do not need it, and which
    models get scaled is a Layer 2-3 decision) and the physics features
    (temp_diff, power_w, wear_strain), which belong to the features module.
    This is the pre-engineering baseline.
    """
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(schema.categorical_features)),
            ("numeric", "passthrough", list(schema.numeric_features)),
        ],
        remainder="drop",
    )


def engineered_preprocessor(schema: AI4ISchema) -> ColumnTransformer:
    """Preprocessing for a model consuming physics features.

    Same as `preprocessor()`, plus `schema.engineered_features` in the
    passthrough list. Placing `PhysicsFeatures` ahead of `preprocessor()`
    alone is not enough -- the plain `ColumnTransformer`'s `remainder="drop"`
    silently discards `temp_diff`/`power_w`/`wear_strain` because they are not
    in its column list. This is that list, corrected.
    """
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(schema.categorical_features)),
            ("numeric", "passthrough", list(schema.numeric_features) + list(schema.engineered_features)),
        ],
        remainder="drop",
    )


def _pipeline(schema: AI4ISchema, classifier: Any) -> EstimatorFactory:
    """`classifier` is cloned inside the returned closure, not reused.

    Every other piece here (`preprocessor(schema)`) was already built fresh
    per call; `classifier` was the one object captured once by the closure
    and handed to every fold's `Pipeline` unchanged. Nothing currently
    registered has warm-start or other cross-fit state, so this was not
    producing wrong numbers -- but it was relying on that being true rather
    than making it structurally impossible to get wrong, which is the whole
    point of the factory pattern documented in `eval/cv.py`. `clone()` is
    what `sklearn.model_selection` itself uses for exactly this reason.
    """
    return lambda: Pipeline([("preprocess", preprocessor(schema)), ("classifier", clone(classifier))])


def _physics_pipeline(schema: AI4ISchema, classifier: Any, scale: bool = False) -> EstimatorFactory:
    """Pipeline for a model that needs the physics features.

    `scale=True` adds a `StandardScaler` after preprocessing -- needed for
    regularised linear models (an L2 penalty only means the same thing across
    features on comparable scales) but deliberately not used for trees, which
    split on raw thresholds and do not need it.

    Every step, including `classifier`, is constructed fresh inside the
    returned closure -- see `_pipeline`'s docstring for why reusing a single
    captured instance across folds is a structural risk even when today's
    estimators happen to reset their own state cleanly on refit.
    """

    def factory() -> Pipeline:
        steps: list[tuple[str, Any]] = [
            ("physics", PhysicsFeatures(schema)),
            ("preprocess", engineered_preprocessor(schema)),
        ]
        if scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("classifier", clone(classifier)))
        return Pipeline(steps)

    return factory


@register("dummy_constant_negative")
def _constant_negative(schema: AI4ISchema, seed: int) -> EstimatorFactory:
    """Predicts "no failure" always. The harness-validation baseline.

    Expected: recall 0, F2 0, PR-AUC and Brier both at the positive base rate
    (0.0339 on AI4I). Would score 96.61% on accuracy, which is why accuracy is
    not implemented.
    """
    return _pipeline(schema, DummyClassifier(strategy="constant", constant=0))


@register("dummy_prior")
def _prior(schema: AI4ISchema, seed: int) -> EstimatorFactory:
    """Predicts the training base rate for every row.

    Ranks no better than chance, so PR-AUC sits at the base rate, but its Brier
    score is the best any non-discriminative model can achieve. Useful as the
    calibration floor.
    """
    return _pipeline(schema, DummyClassifier(strategy="prior"))


@register("dummy_stratified")
def _stratified(schema: AI4ISchema, seed: int) -> EstimatorFactory:
    """Samples predictions from the training class distribution.

    The only seeded dummy, and the one that shows fold-to-fold variance is being
    measured rather than being accidentally zero.
    """
    return _pipeline(schema, DummyClassifier(strategy="stratified", random_state=seed))


@register("gnb")
def _gaussian_nb(schema: AI4ISchema, seed: int) -> EstimatorFactory:
    """Layer 2: Gaussian + categorical Naive Bayes. See `models/bayes/gnb.py`.

    Bypasses `engineered_preprocessor` deliberately: `MixedNaiveBayes` needs
    `type` as a named category, not one-hot columns, and reads the numeric and
    physics columns by name from the DataFrame `PhysicsFeatures` produces.
    """
    return lambda: Pipeline(
        [
            ("physics", PhysicsFeatures(schema)),
            ("classifier", MixedNaiveBayes(schema)),
        ]
    )


@register("bayes_logreg")
def _bayes_logreg(schema: AI4ISchema, seed: int) -> EstimatorFactory:
    """Layer 2: Bayesian logistic regression via Laplace's approximation.

    See `models/bayes/bayes_logreg.py`. Uses `_physics_pipeline(scale=True)`:
    unlike the trees Layer 3 will add, an L2-regularised linear model needs
    its inputs on comparable scales for one shared prior variance (`C`) to
    mean the same thing across features.
    """
    return _physics_pipeline(schema, BayesianLogisticRegression(BayesianConfig()), scale=True)


@register("decision_tree")
def _decision_tree(schema: AI4ISchema, seed: int) -> EstimatorFactory:
    """Layer 3's baseline -- see `models/trees/tree.py` and `TreeConfig` for
    why `max_depth` is a fixed, stated choice rather than a tuned one. This
    is the model CLAUDE.md's falsification test (boosting vs. this tree,
    beyond the CV spread) is measured against.

    `scale=False`: trees split on raw thresholds and gain nothing from
    scaling.
    """
    config = TreeConfig(random_state=seed)
    return _physics_pipeline(schema, build_depth_limited_tree(config), scale=False)


@register("random_forest")
def _random_forest(schema: AI4ISchema, seed: int) -> EstimatorFactory:
    """Layer 3: bagged, unrestricted-depth trees. See `models/trees/forest.py`."""
    config = TreeConfig(random_state=seed)
    return _physics_pipeline(schema, build_random_forest(config), scale=False)


@register("adaboost")
def _adaboost(schema: AI4ISchema, seed: int) -> EstimatorFactory:
    """Layer 3: AdaBoost over `class_weight='balanced'` decision stumps.

    See `models/trees/boosting.py` for why the locked imbalance decision is
    applied to the base stump rather than to `AdaBoostClassifier` itself,
    which has no `class_weight` parameter.
    """
    config = BoostingConfig(random_state=seed)
    return _physics_pipeline(schema, build_adaboost(config), scale=False)


@register("gradient_boosting")
def _gradient_boosting(schema: AI4ISchema, seed: int) -> EstimatorFactory:
    """Layer 3: Gradient Boosting, reweighted via explicit `sample_weight`
    computed at fit time -- see `models/trees/boosting.py`. There is no
    `class_weight` parameter to set for this algorithm; this is the
    equivalent applied the only way sklearn's API allows.
    """
    config = BoostingConfig(random_state=seed)
    return _physics_pipeline(schema, build_gradient_boosting(config), scale=False)


@register("xgboost")
def _xgboost(schema: AI4ISchema, seed: int) -> EstimatorFactory:
    """Layer 3: XGBoost, reweighted via `scale_pos_weight` computed at fit
    time -- literally what CLAUDE.md's locked imbalance decision names by
    name. See `models/trees/boosting.py`.
    """
    config = BoostingConfig(random_state=seed)
    return _physics_pipeline(schema, build_xgboost(config), scale=False)


@register("soft_voting")
def _soft_voting(schema: AI4ISchema, seed: int) -> EstimatorFactory:
    """Layer 3: soft-voting ensemble of this layer's other five models.

    See `models/trees/voting.py`. Each member is built with the same configs
    used when it is registered and run on its own.
    """
    tree_config = TreeConfig(random_state=seed)
    boosting_config = BoostingConfig(random_state=seed)
    return _physics_pipeline(schema, build_soft_voting(tree_config, boosting_config), scale=False)
