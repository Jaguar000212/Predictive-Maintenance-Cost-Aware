# Predictive-Maintenance-Cost-Aware

Cost-aware evaluation of probabilistic and ensemble models for industrial
predictive maintenance — do complex models actually save money?

## Question

On near-deterministic failure boundaries, does algorithmic complexity buy
anything once models are evaluated on expected cost rather than accuracy?

## Hypothesis

Once physics-derived features are engineered, tree-based models will converge to
near-identical performance, and the choice of alarm threshold will affect total
cost more than the choice of algorithm.

**Falsified if** boosting beats a depth-limited decision tree by more than the
cross-validation spread; or any model exceeds the computed ceiling without an
identifiable leakage path; or algorithm choice moves cost more than threshold
choice does.

A negative result is a valid finding. Models are not tuned toward the hypothesis.

## Data

| Dataset | Role | Size |
|---|---|---|
| AI4I 2020 (UCI 601) | Binary classification | 10,000 rows, 339 failures (3.39%) |
| NASA C-MAPSS FD001 | Censored lifetime estimation | 100 train + 100 test trajectories |

Raw data is not committed. Restore it under `data/raw/AI4I_2020/` and
`data/raw/CMAPSS_2008/` before running anything.

## Findings so far — EDA

**AI4I has two recall ceilings, not one.**

| Definition | Recoverable | Ceiling |
|---|---|---|
| Strict — HDF, PWF, OSF (exact threshold rules) | 287 / 339 | **84.66%** |
| Extended — plus TWF | 330 / 339 | **97.35%** |

The 15.34% the strict ceiling leaves behind is *not* mostly randomness, which was
the initial assumption. It decomposes as:

- **12.68%** TWF-only — the tool-wear threshold is drawn from U[200, 240] min, so
  high wear is a visible risk signal even though which tool fails is not
- **0.00%** RNF-only — the random failure mode explains **no** failure uniquely
- **2.65%** unexplained — 9 rows labelled positive with no mode flag, genuinely
  irreducible

So the achievable ceiling is a **range**, and which end to target is a cost
decision, not a modelling one. Benchmarking against a single fixed 84.66% would
make a correct model look anomalous.

Also measured: 18 rows carry a mode flag but a negative label — all of them
RNF-only, and since RNF is independent of every feature, they are
indistinguishable from ordinary negatives and do not cap precision. 24 rows trip
two or more modes, which is why the ceiling is computed over distinct rows rather
than summed flags.

**C-MAPSS FD001 lifetimes** — mean 206.31 cycles, SD 46.34, range 128–362 over the
100 complete training trajectories. Test trajectories are truncated at 37.2% of
life on average (worst case 79.3%), so censoring handling is not a refinement
here. Seven columns are constant, two of them only to floating-point tolerance.

Full report: `reports/eda/eda_report.txt` (regenerate with the command below).

## Architecture

| Layer | Technique | Produces |
|---|---|---|
| 1 | Censored Weibull MLE (explicit likelihood) | Age-based maintenance interval |
| 2 | Gaussian NB + Bayesian logistic regression | Calibrated probabilities |
| 3 | Decision tree, Random Forest, AdaBoost, GB, XGBoost, Voting | Discriminative predictions |
| 4 | Cost model + threshold optimisation | Cost per 1000h per policy |

Layer 1 uses C-MAPSS; Layers 2–3 use AI4I. They converge only at Layer 4. No model
or parameter transfers between datasets.

## Setup

Requires the conda env `AI` (Python 3.11).

```bash
conda install -n AI -c conda-forge --file requirements.txt
```

```bash
conda run -n AI python -m pip install -e .
```

## Usage

```bash
conda run -n AI python scripts/run_eda.py
```

```bash
conda run -n AI python -m pytest tests/ -q
```

Modules under `src/pdm/` use relative imports and cannot be executed directly —
run the entry points in `scripts/` instead.

## Design

Layered, one-way dependencies, composed at the edge:

```
config    frozen, serialisable settings   depends on nothing
loaders   file -> validated DataFrame     depends on config
eda       DataFrame -> analysis           depends on config (schemas only)
eval      arrays -> metrics               depends on config (settings only)
```

Analyses never load files — frames arrive through constructors, and `scripts/`
is the only place that knows both halves. Adding a dataset means one
`DatasetLoader` subclass; adding a report section means one `Analysis` subclass
appended to a list. No existing code changes either way.

**Every setting that can change a result is a field on a frozen dataclass in
`config.py`.** Frozen so mid-run mutation raises rather than silently producing
irreproducible numbers, and serialisable so each results JSON records the exact
configuration behind it. Two carry particular weight:

- `DeterminismConfig` — which failure modes count as recoverable. Moving TWF
  between groups moves the headline ceiling by 12.7 points, so it is a recorded
  setting rather than a constant.
- `CostConfig` — ships deliberately **unset**, and raises if a cost is computed
  before the ratio is chosen and justified. Picking it after seeing model results
  would be indistinguishable from tuning toward the hypothesis.

## Metrics

Accuracy is **not implemented**, deliberately: a constant-negative model scores
96.61% on this data. Evaluation uses PR-AUC, recall, F2, Brier score, and cost per
1000h. PR-AUC is average precision, not trapezoidal area — the two differ and only
one is intended. The decision threshold has no default; it is always explicit.

## Cross-validation

Repeated stratified k-fold (5×5 by default). A single 80/20 split would leave ~68
positives in test, where differences between models are indistinguishable from
resampling noise.

Estimators are supplied as a **factory** — a callable returning a fresh, unfitted
object — rather than an instance. Passing an instance invites reuse across folds,
where a fitted scaler or warm-started ensemble carries test-fold information
forward. Leakage raises nothing and produces ~0.99 scores that look like success,
so the harness prevents it structurally and is tested against a known-answer case:
an unlimited-depth tree on random labels must score at the base rate, and does.

`compare()` implements the falsification rule — challenger beats baseline by more
than the CV spread. One caveat that belongs in the write-up: repeated k-fold folds
reuse rows, so the standard deviation across fits is a descriptive spread, not a
standard error. Treat the rule as a decision procedure, not a significance test.

## Running an experiment

Experiments are YAML files in `configs/`, one per experiment:

```bash
conda run -n AI python scripts/run_experiment.py configs/dummy.yaml
```

Each run writes one JSON to `results/` carrying the full config, seed list, git
SHA, library versions, and metrics. Those files are committed — they are the
audit trail behind the rule that no figure may come from an unrecorded run. A
run recorded from a dirty working tree is flagged `reproducible: false` and
warns, because the SHA then names a commit that does not contain the code that
ran.

Unknown keys in a config are rejected rather than ignored. A silently-dropped
typo would record a run under settings it never used.

## Features

`src/pdm/features/physics.py` derives three columns the raw sensors don't carry
on their own, each one turning a failure-mode rule into an axis-aligned cut:

| Feature | Formula | Recovers |
|---|---|---|
| `temp_diff` | `process_temp − air_temp` | HDF boundary (8.6 K) |
| `power_w` | `torque × (2π × rpm / 60)` | PWF band (3500–9000 W) |
| `wear_strain` | `tool_wear × torque` | OSF limits (tier-dependent on `type`) |

The transformer is stateless — every output is a per-row function of that
row's own inputs, so there is no train-derived statistic for a CV fold
boundary to leak through. The `2π/60` conversion is load-bearing, not
cosmetic: skip it and `power_w` is still monotonic in true power, so nothing
raises and PR-AUC can still look fine — it just stops being watts, and the
PWF band stops lining up with a single split. `tests/test_physics.py` pins
the formula against a hand-computed value for that reason.

## Layer 1 — censored Weibull MLE

`src/pdm/models/mle/censored_weibull.py` fits a Weibull(β, η) to C-MAPSS engine
lifetimes by maximum likelihood, with the log-likelihood written out by hand
(no survival-analysis package) — a failure contributes the density, a censored
engine contributes the survival function, and conflating the two silently
biases every fitted lifetime short.

`CensoredWeibullMLE.fit(durations, events)` raises rather than returning an
unconverged result; `predict_distribution()` returns an immutable
`WeibullDistribution` exposing `survival`, `hazard`, and `quantile` — the last
one is the age-based maintenance interval this layer produces (the age by
which a given fraction of units are expected to have failed).

**Verified against an independent oracle.** On the 100 real train lifetimes
(uncensored — every train engine runs to failure), this estimator gives
β = 4.4087, η = 225.03, matching `scipy.stats.weibull_min.fit` to 4 decimal
places. An earlier method-of-moments prediction (β ≈ 4.9–5.0) turned out to be
the wrong number to check against, not evidence of a bug: moment-matching and
maximum likelihood are different estimation principles, and at n=100 they can
diverge by double digits in percentage terms even on the same data. Detail in
`CLAUDE.md`.

Parameter recovery under simulated censoring is tested directly: fit against
data censored at a known cutoff, and check the estimator recovers the
*generating* parameters — not the shorter lifetime a naive fit (treating every
censored row as if it failed at the cutoff) would produce.

## Layer 2 — Gaussian Naive Bayes and Bayesian logistic regression

Both live under `src/pdm/models/bayes/` and both deliberately use AI4I's
**true** class prior (~3.4%), not `class_weight='balanced'`. That rule is
locked for Layer 3's discriminative trees, where the training loss needs
rebalancing so the minority class isn't ignored; Layer 2 exists to produce
*calibrated* probabilities (Brier score is a locked metric precisely because
it catches miscalibration), and reweighting toward 50/50 would bias every
probability away from the true base rate the way SMOTE does, just via the
loss function instead of resampling.

**`gnb.py` — `MixedNaiveBayes`.** Naive Bayes' independence assumption
licenses modelling each feature with whatever distribution actually fits it:
the five continuous/physics features as Gaussian (`GaussianNB`), and `type`
(L/M/H) as a proper category (`CategoricalNB`) rather than one-hot columns
fed into a Gaussian. The two log-likelihoods are combined by hand — exact
under naive independence, not an approximation.

Predictable and stated before running: `power_w` is an exact function of
`torque_nm` and `rot_speed_rpm`, both of which stay in the feature matrix.
Naive independence treats every feature as separate evidence, so a feature
and its own derivation get double-counted, pushing probabilities further
from 0.5 than the evidence supports. Measured on the real data: GNB's Brier
score (0.0320) is worse than the logistic regression's (0.0231) at otherwise
comparable settings — the calibration cost of that violated assumption,
exactly as predicted before running either model.

**`bayes_logreg.py` — `BayesianLogisticRegression`.** No MCMC (out of scope
per CLAUDE.md) — Laplace's approximation instead: find the MAP weight vector
by ordinary L2-regularised logistic regression (L2 strength `1/C` *is*
Gaussian-prior MAP estimation), then approximate the posterior around it as
Gaussian using the closed-form Hessian of a logistic log-likelihood. A new
prediction integrates the sigmoid over that whole posterior via MacKay's
probit approximation rather than evaluating it only at the MAP point — the
"+ uncertainty" this layer exists to add. As posterior variance goes to zero
this collapses back to a plain point-estimate prediction; tested directly.

Needs scaled inputs (unlike trees): an L2 penalty and one shared prior
variance `C` only mean the same thing across features that are on comparable
scales, and `wear_strain` (~10⁵) and `temp_diff` (~10) are not.

## Calibration diagnostics

`src/pdm/eval/calibration.py` provides reliability curves and a Brier-score
decomposition (Murphy 1973): `reliability` (calibration error), `resolution`
(ability to separate cases at all), `uncertainty` (fixed by the base rate).

One identity is exact and one is not, and the module is explicit about which
is which: `reliability - resolution + uncertainty` reconstructs the score you
get by replacing every prediction with its bucket's mean (`binned_brier`) —
verified against an independent row-level recomputation in
`tests/test_calibration.py` — not the raw per-row score `eval.metrics.brier`
reports. The two converge as bucket count grows but are not the same number
at any finite bucket count, and both are reported rather than conflated.

Bucketing defaults to equal-*count* bins (`strategy="quantile"`), not
equal-width: at AI4I's 3.39% base rate, equal-width bins leave almost the
whole `[0, 1]` range empty and cram nearly every prediction into the first
bucket or two.

**Isolates exactly where GNB's calibration cost comes from.** Decomposing
both Layer 2 models on a held-out split: resolution is nearly identical
(GNB 0.00516 vs BLR 0.00547 — both separate failures from non-failures about
equally well), but reliability differs by two orders of magnitude (GNB
0.00600 vs BLR 0.00003). GNB's worse Brier score is not a ranking problem —
it is calibration error, isolated precisely to the mechanism predicted before
this was run: correlated features counted as independent evidence.

## Status

**Week 1 gate passed.** The constant-negative baseline runs end to end and writes
a results JSON. Measured over 25 folds: recall 0, PR-AUC 0.0339, Brier 0.0339 —
all at the positive base rate, exactly as predicted before running. That number
is now a permanent regression test; if it moves, the harness is broken rather
than the model being poor.

EDA complete. Evaluation harness complete: metrics, cross-validation, the
model-comparison rule, calibration diagnostics, and run recording. Physics
features, the Layer 1 Weibull MLE, and Layer 2 (Gaussian NB, Bayesian
logistic regression) are built. Layer 3 (tree models) is next — the last
piece before Week 2's gate (Weibull validated, reliability diagrams exist)
closes.

164 tests cover the config guards, the loader corruption paths (against synthetic
fixtures, not the real data), every metric against hand-computed values, the
cross-validation leakage guarantees, the physics formulas, the Weibull MLE
(including recovery under simulated censoring and agreement with an
independent oracle), the Layer 2 likelihood/posterior formulas (against hand
recomputations, including the Laplace covariance and its zero-variance limit),
the Brier decomposition (against an independent row-level recomputation),
and the gate itself.

Working agreement, locked decisions, and verified data facts are in `CLAUDE.md`.
Rationale for each decision is in `docs/DECISIONS.md`.
