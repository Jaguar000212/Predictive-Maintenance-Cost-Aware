# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Semester ML project. Cost-aware evaluation of probabilistic and ensemble models
for industrial predictive maintenance.

## Working agreement

**Both developers have no prior ML background.** This changes how you work here:

- Explain each component *before* writing it, in plain language. Define every ML
  term and algorithm name on first use.
- Neither developer commits code they can't explain in one sentence — not what it
  does mechanically, but *why it's there*.
- Before running any model, state the number you expect. Flag it loudly when the
  result differs materially: in ML, broken code runs fine and prints confident
  numbers. Leakage does not raise exceptions.
- When asked for a component, build that component. Don't build ahead.
- Proactively answer: "what would make this silently produce wrong results?"

## The question

> On near-deterministic industrial failure boundaries, does algorithmic complexity
> improve outcomes once models are evaluated on expected cost rather than accuracy?

**Hypothesis.** Once physics features are engineered, tree models converge to
statistically indistinguishable performance at a computable ceiling, and total
expected cost varies more with the alarm threshold than with the algorithm.

**Falsified if** boosting — XGBoost, the pre-registered test (`docs/DECISIONS.md`
D10) — beats a depth-limited tree by more than the CV standard deviation; or any
model exceeds the ceiling without an identifiable leakage path; or algorithm
choice moves cost more than threshold choice does.

Gradient Boosting and AdaBoost stay in the Layer 3 comparison table as
descriptive context — informative about how boosting implementations differ,
but not alternate falsification tests. Only XGBoost's result decides this.

A negative result is a valid finding. Do not tune toward confirming the hypothesis.

## Locked decisions — do not revisit without explicit instruction

| Decision | Rule |
|---|---|
| Imbalance | `class_weight='balanced'` / `scale_pos_weight`. **Not** SMOTE. |
| SMOTE | Ablation only, judged on Brier score and reliability curves. |
| Accuracy | **Banned as a metric.** Constant-negative scores 96.6%. |
| Primary metrics | PR-AUC, recall, F2, Brier score, cost per 1000h |
| Leakage | Drop TWF/HDF/PWF/OSF/RNF flags from the feature matrix |
| CV | All scaling, resampling, calibration, tuning **inside** each fold |
| Gaussian Processes | Rejected. Do not add. |
| Deep learning, MCMC | Out of scope |
| C-MAPSS subsets | FD001 only |

Rationale for each is in `docs/DECISIONS.md`.

## Verified data facts

Measured from the actual files. Do not re-derive or substitute published figures.

### AI4I 2020 (UCI 601) — classification

- 10,000 rows; 339 failures (3.39%); class ratio 28.5:1
- Mode counts **as observed**: HDF 115, PWF 95, OSF 98, TWF 46, RNF 19
  - Published paper says TWF 51, RNF 5. The deterministic three match exactly;
    the stochastic two do not. Use the measured values.
- 24 rows trip ≥2 modes; 18 rows flagged but labelled negative (all RNF-only,
  feature-indistinguishable, therefore inert); 9 rows labelled positive with no
  flag (irreducible, caps recall)

**Two recall ceilings:**

- **84.66%** strict — 287/339 failures trip HDF, PWF, or OSF (exact threshold rules)
- **97.35%** extended — 330/339 including TWF

TWF is *semi*-deterministic: the wear threshold is drawn from U[200,240] min, so
high wear is a visible risk signal, but which tool in that band fails is random.
The 12.7-point gap is buyable only by flagging the whole high-wear band, at a
fixed precision cost. **Which ceiling to target is a Layer 4 cost decision, not a
modelling decision.** Report both; let the cost curve choose.

Anything above 97.35% is a leakage path. Investigate, don't celebrate.

### C-MAPSS FD001 — lifetime estimation

- 100 train trajectories (run to failure), 100 test (truncated)
- 26 columns: unit, cycle, 3 op settings, **21 sensors** (cols 6–26).
  The NASA readme says "sensor 1–26" — that is a typo in the readme.
- Whitespace-delimited, no header. `sep=r'\s+'`. Some copies have two trailing
  empty columns.
- Train lifetimes: mean 206.31, SD 46.34, range 128–362
- Test truncation: mean 37.2% of life unobserved
- **7 constant columns**: sensor_1, 5, 10, 16, 18, 19, op_setting_3. Two are
  float-noise constant (3e-18, 5e-15) — use a tolerance, not `std == 0`.

**Verified Weibull fit** (uncensored MLE on the 100 train lifetimes): shape
β = 4.4087, scale η = 225.03. Confirmed against `scipy.stats.weibull_min.fit`
as an independent oracle to 4 decimal places — this is the number any correct
implementation should reproduce.

An earlier method-of-moments prediction here said β ≈ 4.9–5.0. That number was
**not wrong because of a bug** — it's wrong because moment-matching (solve for
β from the sample coefficient of variation alone) and maximum likelihood are
different estimation principles that need not agree. MoM matches the sample
CV (0.2246) by construction; the true MLE fit implies a different CV (0.2569)
because MLE is shaped by the full likelihood, not two summary statistics. At
n=100 they diverged by ~14%. Lesson for next time: a moments-based number is a
sanity check on convergence, not a ground truth to fit toward.

**Censoring — critical.** `RUL_FD001.txt` supplies true remaining life, so the
test set is *de-censored*. Fitting to true durations and calling it a censored fit
is circular. Protocol:

1. Simulate censoring on the training set at a known cutoff; verify the estimator
   recovers the uncensored parameters within CIs.
2. Fit the test partition using **only** truncation times + censoring indicator,
   with `RUL_FD001.txt` withheld.
3. *Then* use `RUL_FD001.txt` as held-out ground truth to check the fit.

Step 3 is unusually strong validation — censored estimation is normally
unverifiable. Do not shortcut it.

## Architecture

| Layer | Technique | Produces |
|---|---|---|
| 1 | Censored Weibull MLE (hand-written likelihood, `scipy.optimize`) | Age-based maintenance interval |
| 2 | Gaussian NB + Bayesian logistic regression | Calibrated probabilities + uncertainty |
| 3 | Decision tree, Random Forest, AdaBoost, GB, XGBoost, soft Voting | Discriminative predictions |
| 4 | Cost model + threshold optimisation | Cost per 1000h per policy |

Layer 1 uses C-MAPSS. Layers 2–3 use AI4I. They converge only at Layer 4.
No model or parameter transfers between datasets.

**Do not use a survival-analysis package for Layer 1.** The censored
log-likelihood is written explicitly — complete observations contribute the
density, censored observations contribute the survival term. That code is the
project's only genuine MLE content.

## Features (AI4I)

| Feature | Formula | Recovers |
|---|---|---|
| `temp_diff` | process_temp − air_temp | HDF boundary (8.6 K) |
| `power_w` | torque × (2π × rpm / 60) | PWF band (3500–9000 W) |
| `wear_strain` | tool_wear × torque | OSF limits (11000/12000/13000) |

`type` (L/M/H) **must** stay in the feature matrix — OSF's threshold is
tier-dependent and the ceiling is unreachable without it.

The 2π/60 conversion is not optional; without it the PWF boundary stops being an
axis-aligned cut.

## Commands

Python lives in the **conda env `AI`** (Python 3.11). There is no venv in the repo.
Either activate it or prefix every command with `conda run -n AI`.

```bash
conda run -n AI python scripts/run_eda.py
```

| Task | Command |
|---|---|
| One-time setup | `conda run -n AI python -m pip install -e .` |
| EDA report | `conda run -n AI python scripts/run_eda.py` |
| Run an experiment | `conda run -n AI python scripts/run_experiment.py configs/dummy.yaml` |
| Experiment, no results file | `... run_experiment.py configs/dummy.yaml --no-write` |
| Tests | `conda run -n AI python -m pytest tests/ -q` |
| One test | `conda run -n AI python -m pytest tests/test_cv.py::test_folds_are_stratified -q` |
| Format (line length 110) | `conda run -n AI python -m black src scripts tests` |
| Lint | `conda run -n AI python -m ruff check src scripts tests` |

The editable install is **required**, not a convenience: without it `from pdm import ...`
resolves only from `scripts/`, and notebooks or ad-hoc scripts fail.

**Commit before any run whose output reaches the report.** `ResultsWriter` warns when
a run is recorded from a dirty tree, because the recorded SHA then names a commit that
does not contain the code that ran.

### Environment traps

- **Never run a module inside `src/pdm/` directly** — `python src/pdm/eda.py` raises
  `ImportError: attempted relative import with no known parent package`, because the
  package modules use relative imports. Run `scripts/run_eda.py`, or `python -m pdm.<mod>`.
  PyCharm's run-current-file button hits this every time; point run configs at `scripts/`.
- **`conda run` cannot execute a multi-line `python -c`** — it fails with
  `NotImplementedError: Support for scripts where arguments contain newlines`.
  Use a one-liner or write a temp file.
- The import package is `pdm`; the distribution is `predictive-maintenance-cost-aware`.
  Do not `pip install pdm` (the unrelated build tool) into this env — it shadows the import.
- **`/data/raw/` and `/reports/` are gitignored.** A fresh clone has no data, and every
  loader raises `DataValidationError` until the AI4I CSV and C-MAPSS text files are
  restored under `data/raw/AI4I_2020/` and `data/raw/CMAPSS_2008/`.

## Repo layout

Actual, as of the EDA milestone:

```
src/pdm/config.py      Frozen, serialisable settings. Depends on nothing.
src/pdm/loaders.py     DatasetLoader ABC + AI4ILoader, CMAPSSLoader,
                       CMAPSSRULLoader, CMAPSSLifetimeBuilder
src/pdm/eda.py         Analysis ABC + four analyses, composed by EDAReport
src/pdm/eval/metrics.py  Pure metric functions + MetricSuite
scripts/run_eda.py     Composition root: wires loaders to analyses
tests/                 test_config, test_loaders, test_metrics
docs/DECISIONS.md      Rationale for every locked decision
```

Dependency direction is one-way: `config` → `loaders` / `eda` / `eval`. Analyses
never load files; frames arrive through their constructors, and `scripts/` is the
only place that knows both halves. That is what lets any analysis run against a
fixture.

### Configuration

**Every setting that can change a result is a field on a frozen dataclass in
`config.py`.** No result-affecting literal belongs anywhere else.

- Frozen, so mid-run mutation raises instead of silently producing
  irreproducible numbers
- `ExperimentConfig.to_dict()` feeds the results JSON, which is what makes the
  "no figure from an unrecorded run" rule enforceable
- `config.with_(...)` returns a copy — never edit defaults to run a variant, or
  the recorded history stops matching the code

`DeterminismConfig` is the one to watch: moving TWF between groups moves the
headline ceiling from 84.66% to 97.35%. It validates that every mode is
classified exactly once, so a mode cannot silently vanish from the arithmetic.

`CostConfig` ships **deliberately unset** and `validate()` raises. No cost figure
can be produced before the ratio is chosen, justified, and recorded in
`docs/DECISIONS.md`.

Loaders **assert** their output shape and raise `DataValidationError` rather than
coercing. That is deliberate: a malformed load that raises costs minutes, one that
returns a plausible frame costs a week. The base class prefixes every failure with
the offending filename.

Correction to a common assumption: `encoding="utf-8-sig"` on the AI4I read is
**defensive, not required**. Measured on pandas 2.3, both the C and python engines
strip the UTF-8 BOM whatever encoding is declared. It still matters outside pandas
— `open(path, encoding="utf-8")` on that file yields a column named `﻿UDI`.

`cmapss_lifetimes()` returns the canonical survival form — one row per engine with
`duration` / `event` / `true_duration` — which is what Layer 1 consumes. `duration` for
test units is a **censoring time, not a lifetime**; `true_duration` is ground truth for
step 3 of the censoring protocol only, and must never enter a fit.

### Final structure — locked

This is the target at project end. Create each directory **only when the component
that lives in it is built** — do not scaffold ahead. Anything not on this list needs
an explicit decision before it is added.

```
data/raw/                       gitignored; AI4I_2020/, CMAPSS_2008/
src/pdm/
  config.py                     paths, schemas, determinism grouping, cost constants
  data/       ai4i.py, cmapss.py
  features/   physics.py (temp_diff, power_w, wear_strain), pipeline.py
  models/
    registry.py                 name -> estimator factory; configs address models by string
    mle/      censored_weibull.py            Layer 1
    bayes/    gnb.py, bayes_logreg.py        Layer 2
    trees/    tree.py, forest.py, boosting.py, voting.py   Layer 3
  eval/       metrics.py, cv.py, results.py, calibration.py
  decision/   cost_model.py, policy_sim.py   Layer 4
  eda.py
scripts/      run_eda.py, run_experiment.py
tests/
configs/      one YAML per experiment          committed
results/      one JSON per run                 committed — this is the provenance record
reports/      generated figures and tables     gitignored
notebooks/    read from results/ only; never compute a final number
docs/         DECISIONS.md, report drafts
```

**Everything Python nests under the single `pdm` package.** Do not create top-level
`src/data/`, `src/models/`, or `src/eval/` — src-layout would turn each into a separate
top-level import name, and `data`/`models`/`eval` are generic enough to collide with
other packages in the shared `AI` env. One top-level name is what keeps `pip install -e .`
clean. Imports read `from pdm.models.mle import censored_weibull`.

`configs/` (per-experiment YAML) and `src/pdm/config.py` (project-wide constants) are
different things and both exist. `results/` is committed and `reports/` is not: results
JSON is the audit trail behind the "no unrecorded run" rule, while reports are
regenerable output.

Migration when Layer 1 starts: `src/pdm/loaders.py` splits into `src/pdm/data/ai4i.py`
and `src/pdm/data/cmapss.py`. Not before — at ~230 lines it does not yet warrant the split.

**Interface contract.** Classifiers expose `fit(X, y)` and `predict_proba(X)`.
The lifetime model exposes `fit` and `predict_distribution`. Every run writes a
JSON to `results/` containing config + seed list + git SHA + metrics. Notebooks
read from `results/` and never compute a final number.

No figure or table in the report may come from an unrecorded run.

## Gates

| Gate | Criterion |
|---|---|
| **Week 1** | ✅ **PASSED.** `run_experiment.py configs/dummy.yaml` runs the constant-negative baseline end to end and writes a results JSON. Measured: recall 0, PR-AUC 0.0339, Brier 0.0339 — all at the base rate, as predicted. Asserted permanently in `tests/test_experiment.py`. |
| Week 2 | Weibull validated against simulated censoring; reliability diagrams exist |
| Week 3 | Model comparison table populated with CV mean ± SD |
| Week 4 | Cost curves and policy table complete. **Code freeze day 24.** |

Harness validation: constant-negative dummy gives recall 0, PR-AUC ≈ 0.0339
(the base rate). Anything else means the harness is broken, not the model.

Cut order if behind: SMOTE ablation → SHAP → multi-mode analysis.
**Never cut Layer 4.** It is the contribution.

## Known traps

- Splitting or resampling before the CV fold → silent leakage, ~0.99 scores
- Leaving mode flags in features → same
- SMOTE shifts the prior 3.4% → ~50%, miscalibrating probabilities by ~30×.
  Layer 4 does arithmetic on those probabilities. The cost analysis becomes
  fiction with no visible error.
- SMOTE also interpolates torque/rpm/power independently, producing rows where
  power ≠ τ·2πN/60 — physically impossible machines
- Single train/test split → ~68 positives in test, differences are noise.
  Use repeated stratified k-fold.
- `std == 0` misses the two float-noise-constant C-MAPSS columns
- Reading NASA's readme literally gives 26 sensor names for 21 sensors