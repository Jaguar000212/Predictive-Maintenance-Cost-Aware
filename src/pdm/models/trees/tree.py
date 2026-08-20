"""Depth-limited decision tree -- Layer 3's baseline.

A decision tree predicts by asking a sequence of yes/no questions about the
features ("is temp_diff > 8.4?") and following whichever branch matches,
until it reaches a leaf that reports a probability -- the fraction of
training rows that landed there and failed. Unlike Layer 2's models, it makes
no assumption about how features relate to each other; at each step it just
picks whichever single feature and threshold best separates failures from
non-failures among the rows that reached that point.

**This is the fixed point the project's hypothesis is measured against.**
CLAUDE.md's falsification test is "does boosting beat a depth-limited tree by
more than the cross-validation spread" -- everything else in Layer 3 is
compared to what this one produces. See `TreeConfig` in `config.py` for why
`max_depth` is a stated, fixed choice (4) rather than a tuned one: tuning it
would need nested cross-validation, which would turn "depth-limited baseline"
into just another optimised model, defeating the point of having one.

`class_weight='balanced'` is CLAUDE.md's locked imbalance decision, applied
directly here -- unlike Layer 2, there is no calibration-based reason to
deviate from it for a discriminative classifier.

**Measured consequence, worth stating rather than discovering by surprise
later.** Reweighting toward a balanced prior does exactly what it does to any
model: it pushes leaf probabilities away from the true ~3.4% base rate, the
same mechanism CLAUDE.md rejects SMOTE for. On the real data this tree scores
PR-AUC 0.84 (it ranks well) but Brier 0.0352 -- WORSE than the trivial
constant-negative baseline's 0.0339. That is not a bug: `predict_proba()`
here answers "how does this compare to other rows" reliably, but is not a
trustworthy probability at the dataset's real prevalence. Threshold-swept
metrics (`MetricSuite.sweep`) are unaffected, since they count confusion
outcomes empirically at each cutoff rather than trusting the probability
value -- but this Brier score is not comparable to Layer 2's on its own
terms, and any later use of this model's raw probability in a cost formula
would need recalibrating first.
"""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from ...config import TreeConfig


def build_depth_limited_tree(config: TreeConfig | None = None) -> DecisionTreeClassifier:
    config = config or TreeConfig()
    return DecisionTreeClassifier(
        max_depth=config.depth_limited_max_depth,
        min_samples_leaf=config.depth_limited_min_samples_leaf,
        class_weight="balanced",
        random_state=config.random_state,
    )
