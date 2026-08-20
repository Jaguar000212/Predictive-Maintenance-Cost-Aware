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

## D10 — Pending: which boosting algorithm is *the* falsification test

**Not yet decided.** CLAUDE.md's hypothesis says "falsified if boosting beats a
depth-limited tree by more than the CV standard deviation," but Layer 3 has
three boosting algorithms (AdaBoost, Gradient Boosting, XGBoost), and this gap
was never noticed until all three existed to compare against the baseline
(`configs/decision_tree.yaml`).

Measured on identical 5×5 CV folds, PR-AUC vs. the depth-limited tree:

| Algorithm | Δ PR-AUC | Beats baseline beyond the CV SD? |
|---|---|---|
| Gradient Boosting | +0.065 | Yes — falsifies the hypothesis |
| XGBoost | +0.026 | No |
| AdaBoost | −0.057 | Yes, but *worse*, not better |

Matched hyperparameters between GB and XGBoost (`BoostingConfig`) rule out
"different settings" as the explanation for the split verdict.

Choosing a single algorithm *now*, after seeing this table, would be exactly
the kind of post-hoc metric selection D9 exists to prevent. Two honest ways
to resolve it, neither taken yet:

1. Pick one algorithm as the pre-registered test **before** looking at Layer
   4's cost-based results, on grounds independent of which one currently
   wins (e.g. "XGBoost, because it is the one CLAUDE.md's locked imbalance
   decision names by parameter name").
2. Require all three to trip the criterion before calling the hypothesis
   falsified, treating a split result as "inconclusive on this metric."

Also unresolved: this whole comparison is on PR-AUC, a proxy. The hypothesis
is stated against expected cost, which needs Layer 4 (not yet built). Revisit
this decision when Layer 4 lands, not before -- and record whichever option
is chosen here, with the reasoning, before Layer 4 results are reviewed.