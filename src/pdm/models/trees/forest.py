"""Random Forest -- Layer 3's first ensemble.

A single decision tree trained on all the data picks one specific sequence of
splits; a slightly different training sample could easily produce a
different tree. Random Forest tames that instability by training many trees,
each on a bootstrap resample of the rows and a random subset of features at
each split, then averaging their predicted probabilities. Individual trees
can overfit their own resample; the average across many largely cancels that
out.

That averaging is *why* `forest_max_depth=None` (see `TreeConfig`): unlike
the single depth-limited baseline tree, Random Forest's variance control
comes from bagging across many trees, not from restricting any one tree's
depth. Constraining depth here would be solving a problem this model doesn't
have via a mechanism meant for a model that does.

`class_weight='balanced'` is CLAUDE.md's locked imbalance decision, applied
directly -- Random Forest is a discriminative classifier like the tree above,
with no Layer 2 style calibration carve-out.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from ...config import TreeConfig


def build_random_forest(config: TreeConfig | None = None) -> RandomForestClassifier:
    config = config or TreeConfig()
    return RandomForestClassifier(
        n_estimators=config.forest_n_estimators,
        max_depth=config.forest_max_depth,
        min_samples_leaf=config.forest_min_samples_leaf,
        class_weight="balanced",
        random_state=config.random_state,
        n_jobs=-1,  # parallelism only -- does not affect the fitted result
    )
