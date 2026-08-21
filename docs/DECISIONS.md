# Decisions log

Rationale behind the locked decisions in `CLAUDE.md`. Read when a decision is
questioned or when writing the report's methodology section. Each entry records
what was chosen, what was rejected, and why — so the reasoning survives even if
the people don't remember it.

---

## D1 — Class weights, not SMOTE

**Chosen:** `class_weight='balanced'` / `scale_pos_weight`.
**Rejected:** SMOTE, SMOTE-ENN as the default pipeline.

Three reasons, in order of severity:

1. **It invalidates Layer 4 silently.** Resampling shifts the class prior from
   3.4% to ~50%. Output probabilities become miscalibrated by roughly 30× against
   the real base rate. The decision layer computes expected costs directly from
   those probabilities, so every currency figure would be wrong — with no error,
   no warning, and plausible-looking output.
2. **It injects label noise into a noiseless problem.** HDF, PWF, and OSF are
   exact threshold rules. Interpolating between two minority points draws a line
   that crosses the true boundary, manufacturing mislabelled rows.
3. **It breaks physical consistency.** Interpolating torque, rpm, and power
   independently yields rows where power ≠ τ·2πN/60. Those machines cannot exist.

ENN compounds this by deleting real minority points near boundaries — the only
points carrying information about where the cliff is.

**Retained as ablation.** Run it once, judge on Brier score and reliability
curves. Expected finding: ranking performance roughly unchanged, calibration
materially degraded. That negative result is more valuable than the default
"we applied SMOTE because the literature does."

---

## D2 — Gaussian Processes rejected

**Chosen:** Gaussian Naive Bayes + Bayesian logistic regression.
**Rejected:** Gaussian Process Classification.

- sklearn's `GaussianProcessClassifier` **does not expose predictive variance**.
  Uncertainty quantification is the entire reason to reach for GPC, and the
  standard implementation cannot deliver it. Would require GPflow/GPyTorch with
  inducing points — a week that doesn't exist.
- O(n³) on 10,000 rows. The workaround is subsampling, which discards the data.
- A stationary RBF kernel encodes a smoothness prior. The failure boundaries are
  hard axis-aligned cliffs. Wrong inductive bias.
- Neither developer could defend it in a viva. That is a real constraint.

The advice document that proposed GPC (see D8) treated it as the "highest echelon"
of Bayesian modelling. Sophistication is not the objective; a defensible,
calibrated probability is.

---

## D3 — Two datasets

**Chosen:** AI4I 2020 for classification, C-MAPSS FD001 for lifetime estimation.
**Rejected:** AI4I alone; C-MAPSS alone; SECOM; MetroPT-3; Backblaze.

AI4I has no unit identity and no time axis. Nothing possesses a lifetime, so
lifetime estimation is structurally impossible — not hard, impossible. Restricted
to AI4I, the MLE requirement collapses to "logistic regression is fitted by
maximum likelihood," which is a definition shared with most parametric models,
not a contribution.

C-MAPSS records gradual degradation with no discrete labelled failure events.
Forcing it into classification requires an invented labelling rule, making every
downstream result contingent on an arbitrary choice.

Alternatives considered:
- **SECOM** — real data, but 590 anonymised features. No physics engineering, no
  computable ceiling. Kills the project's main analysis.
- **MetroPT-3** — real, unlabelled. Failures arrive as external timestamps needing
  manual alignment. Days of work before any model trains.
- **Backblaze** — arguably the best lifetime data available, but tens of GB.
  Infrastructure cost exceeds the schedule.

**Domain mismatch is acknowledged, not hidden.** Milling machines and turbofans
are different. The datasets are never merged and no model transfers. Layer 4
compares *policy classes* — schedule-based vs sensor-based — each demonstrated on
data that can support it. This appears in the report's limitations section.

---

## D4 — FD001 only

FD002 and FD004 have six operating conditions, so engine lifetimes are confounded
by workload intensity. Controlling for that is a separate piece of work with no
additional insight for this project. FD003 adds a second fault mode, same issue.

FD001: one condition, one fault mode, 100 clean trajectories. Sufficient.

---

## D5 — Accuracy banned

A constant-negative prediction achieves 96.6% on AI4I. Any table containing
accuracy invites a reader to notice this and discount everything around it.

The metric set is chosen for the failure asymmetry: missing a failure costs
roughly an order of magnitude more than a false alarm. Hence F2 over F1, PR-AUC
over ROC-AUC (the latter is inflated by the majority class), and Brier score as a
first-class metric because Layer 4 depends on calibration rather than ranking.

---

## D6 — Two ceilings, and why it matters

Discovered during EDA, not planned. Initially the design assumed a single ~85%
ceiling with TWF and RNF both unlearnable.

RNF is genuinely unlearnable — independent of every feature by construction.
TWF is not. Its threshold is drawn from U[200,240] min, so elevated tool wear
*is* a visible risk signal; what's hidden is which tool within that band fails
rather than being replaced.

So there are two ceilings — 84.66% strict, 97.35% extended — and the 12.7 points
between them are purchasable only by flagging the entire high-wear band, at a
precision cost fixed by the generator's replace/fail ratio.

**Whether to buy them depends on C_cm/C_pm.** That is Layer 4 answering a question
Layer 3 cannot, which is precisely the project's thesis. Report both ceilings and
both policies ("strict physics" vs "physics + high-wear band") and let the cost
curve decide.

This is the strongest finding available in the data. Do not collapse it back to a
single number.

---

## D7 — Censoring protocol

The obvious approach — fit the Weibull to C-MAPSS test units as censored
observations — is circular, because `RUL_FD001.txt` supplies their true lifetimes.
The test set is de-censored.

Two-stage protocol instead:

1. Simulate censoring on the *training* set at a known cutoff. A correct
   implementation recovers the uncensored parameters within their CIs. This
   validates the code before it is trusted.
2. Fit the test partition using only truncation times and the censoring
   indicator, with the RUL file withheld. Then use the RUL file as held-out
   ground truth.

Stage 2 is rare and worth emphasising in the report: censored estimation is
normally unverifiable, since the estimand is by definition unobserved. Here 100
censored observations exist alongside their true values, at a realistic 37% mean
unobserved fraction.

---

## D8 — On the origin of the plan

An earlier LLM-generated research document proposed: AI4I only, SMOTE-ENN as
default, GPC for the Bayesian layer, single train/test split, accuracy reported
alongside other metrics. All five were rejected — see D1, D2, D3, D5 above and
the CV rule in `CLAUDE.md`.

It also contained a Weibull survival section that was unimplementable on the
dataset it recommended, and citations that could not all be verified.

**Standing rule:** every citation is opened and verified before it enters the
report. Every published figure is checked against the file (this caught TWF 46≠51
and RNF 19≠5). Plausible prose is not evidence.

---

## D9 — Report before code

The evaluation protocol, cost model, and falsification criteria were fixed before
any model was trained. This is deliberate: metrics chosen after results are
observed tend to be chosen to flatter them.

The report's results section exists as a scaffold with `[TBD]` values and fixed
table structures. Week 4 fills blanks; it does not decide what to report.

---

## D10 — XGBoost is *the* falsification test's boosting algorithm

**Chosen:** XGBoost is the pre-registered "boosting" algorithm CLAUDE.md's
falsification test runs against. Gradient Boosting and AdaBoost stay in the
Layer 3 comparison table as descriptive context -- informative about how
boosting implementations differ from each other -- but they are not
alternate falsification tests. Only XGBoost's result decides this.

**Why this came up at all.** CLAUDE.md's hypothesis says "falsified if
boosting beats a depth-limited tree by more than the CV standard deviation,"
but Layer 3 has three boosting algorithms (AdaBoost, Gradient Boosting,
XGBoost), and nobody had noticed the hypothesis never said which one counts
until all three existed to compare against the baseline
(`configs/decision_tree.yaml`).

**Why XGBoost, specifically.** It is the one CLAUDE.md's own locked
imbalance decision already names by parameter: "class_weight='balanced' /
scale_pos_weight" -- `scale_pos_weight` is XGBoost's own API, so XGBoost is
the boosting implementation this project's documentation already treated as
canonical, independent of anything to do with which algorithm currently
wins. That independence from the result is the entire point, per D9 --
choosing after seeing the table which algorithm to elevate would be exactly
the post-hoc metric selection D9 exists to prevent.

**The measured numbers this decision was made in front of** (5×5 CV, real
data, PR-AUC vs. the depth-limited tree) -- recorded here so the choice
above is checkable against them, not just asserted:

| Algorithm | Δ PR-AUC | Beats baseline beyond the CV SD? |
|---|---|---|
| Gradient Boosting | +0.065 | Yes |
| **XGBoost (the test)** | **+0.026** | **No** |
| AdaBoost | −0.057 | Yes, but *worse*, not better |

Matched hyperparameters between GB and XGBoost (`BoostingConfig`) rule out
"different settings" as the explanation for the split verdict between them.

**Resulting verdict, on this metric: the hypothesis is NOT falsified.**
XGBoost's improvement over the depth-limited tree does not exceed the CV
spread. Gradient Boosting's larger, SD-exceeding improvement is no longer
the test -- it is now a data point showing boosting implementations can
disagree with each other, which is itself consistent with the hypothesis's
broader claim that algorithm choice matters less than it might appear.

**Still open:** this is a PR-AUC proxy. The hypothesis is stated against
expected cost, which needs Layer 4 (not yet built) before this verdict can
be called authoritative rather than preliminary.

---

## D11 — Cost ratio: missed failure : false alarm : inspection = 10 : 1 : 0.5

**Chosen:** `missed_failure = 10.0`, `false_alarm = 1.0`, `inspection = 0.5`
(abstract currency units — AI4I carries no real monetary figures, so this is
a ratio decision, not a dollar estimate).

**Why now, and why this shape.** `CostConfig` was deliberately left unset
from the start of the project (see `CLAUDE.md`, "Configuration") because the
missed-failure : false-alarm ratio decides the optimal alarm threshold,
decides which recall ceiling (84.66% strict vs. 97.35% extended, D6) is
worth buying, and is the quantity the central hypothesis is measured
against. Choosing it after seeing *cost* results would be tuning toward the
hypothesis. It is chosen here in front of Layer 3's PR-AUC results only —
no cost figure has been computed yet, so nothing about this ratio could
have been reverse-engineered from a cost table.

**Why 10:1, not something else.** Industrial predictive-maintenance
literature commonly places unplanned-failure cost at roughly 5–10× planned
maintenance cost — unplanned downtime carries emergency labor, lost
production, and risk of collateral damage that a scheduled stoppage does
not. 10:1 is the upper end of that commonly cited range, chosen because
AI4I's failure modes (tool overstrain, power-band failure, heat dissipation
failure) plausibly cause the kind of abrupt, damage-risking stoppage that
justifies the higher end rather than the low end.

**Why inspection = 0.5, not 0.** A correct alarm still costs something — the
machine is stopped and someone checks it — but it is the cheapest outcome in
the table because it is planned and non-destructive. Setting it below
`false_alarm` (1.0) encodes that a *correct* stoppage is preferable to an
*incorrect* one, even though both involve stopping the machine.

**What this is not.** This is an asserted ratio with a literature-based
justification, not a measured cost from this project's own data — AI4I does
not carry currency figures. The report's limitations section should state
this plainly. A sensitivity analysis (does the optimal threshold move a lot
or a little as this ratio is varied, e.g. 5:1 and 20:1 alongside 10:1) is
cheap to run once Layer 4 exists and materially strengthens a ratio that is
asserted rather than measured — recommended before this is treated as final
in the report, not required to unblock building Layer 4 itself.

**Structural consequence.** `CostConfig`'s dataclass defaults now carry
these values directly (`src/pdm/config.py`), so every experiment config uses
this ratio unless a config explicitly overrides it. `validate()` still
raises on an explicitly unset or negative value — that guard stays, it just
no longer trips on the default case now that the default is a real decision
instead of a placeholder.