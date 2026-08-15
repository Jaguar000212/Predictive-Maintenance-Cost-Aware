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

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..config import AI4ISchema

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


def _pipeline(schema: AI4ISchema, classifier: Any) -> EstimatorFactory:
    return lambda: Pipeline([("preprocess", preprocessor(schema)), ("classifier", classifier)])


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
