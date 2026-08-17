"""Physics-derived features for AI4I 2020.

Three failure modes fire on a threshold rule that lives on a *derived*
physical quantity, not on any single recorded sensor:

    temp_diff    = process_temp - air_temp        recovers the HDF boundary (8.6 K)
    power_w      = torque * (2*pi * rpm / 60)      recovers the PWF band (3500-9000 W)
    wear_strain  = tool_wear * torque              recovers the OSF limits (tier-dependent)

Without deriving these explicitly, a tree still learns *something* from the
raw sensors -- the rules are monotonic in them -- but it has to approximate a
diagonal or curved boundary with a staircase of splits instead of one. That
extra complexity is exactly the shape of thing that could make a deeper model
look like it "beats" a shallow one for the wrong reason: not because the
problem needs more capacity, but because the shallow tree was never given the
axis the rule actually lives on.

`power_w`'s `2*pi/60` term converts rpm to rad/s so that torque * angular
velocity comes out in watts. Drop it and the column is still monotonic in true
power -- nothing raises, PR-AUC can still look fine -- but it stops being
watts, so the 3500-9000 W band no longer lines up with an axis-aligned cut.
`tests/test_physics.py` pins the formula against a hand-computed value for
exactly that reason.

`type` (L/M/H) is untouched here. OSF's threshold is tier-dependent, and
`wear_strain` only makes that limit reachable when `type` also survives into
the feature matrix -- see `AI4ISchema.feature_columns`.

Integration note for Week 3: `registry.preprocessor()` is deliberately the
pre-engineering baseline and does not include these columns. Slotting this
transformer in ahead of it is not enough on its own -- the downstream
`ColumnTransformer` also needs `schema.engineered_features` added to its
passthrough list, or the new columns get silently dropped by
`remainder="drop"`. That wiring belongs to the model pipeline that consumes
it, not to this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..config import AI4ISchema


class PhysicsFeatures(BaseEstimator, TransformerMixin):
    """Adds `temp_diff`, `power_w`, and `wear_strain` to an AI4I feature frame.

    Stateless: `fit` only checks that the required raw columns are present, so
    the transform is identical whether it sees a fold's train or test rows.
    Every output value is a deterministic function of that row's own inputs --
    there is no cross-row statistic (no mean, no scaling) for a fold boundary
    to leak through. That is what makes this safe to place anywhere in a
    cross-validation pipeline, including before the split.
    """

    def __init__(self, schema: AI4ISchema | None = None) -> None:
        self.schema = schema or AI4ISchema()

    def _required_columns(self) -> tuple[str, ...]:
        s = self.schema
        return (s.air_temp_col, s.process_temp_col, s.rot_speed_col, s.torque_col, s.tool_wear_col)

    def _check_columns(self, X: pd.DataFrame) -> None:
        missing = set(self._required_columns()) - set(X.columns)
        if missing:
            raise ValueError(
                f"PhysicsFeatures: missing required column(s) {sorted(missing)}; " f"has {sorted(X.columns)}"
            )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> PhysicsFeatures:
        self._check_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(X)
        s = self.schema
        out = X.copy()
        out["temp_diff"] = out[s.process_temp_col] - out[s.air_temp_col]
        out["power_w"] = out[s.torque_col] * (2.0 * np.pi * out[s.rot_speed_col] / 60.0)
        out["wear_strain"] = out[s.tool_wear_col] * out[s.torque_col]
        return out

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        base = list(input_features) if input_features is not None else []
        return np.asarray(base + list(self.schema.engineered_features))
