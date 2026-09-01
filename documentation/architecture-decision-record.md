<p align="center">
<img src="https://raw.githubusercontent.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/main/figures/SSOS_longer-aspect-ratio.png" alt="SSOS" width="600" />
</p>

<p align="center">
• <a href="https://mayzune.com/"><strong>May Zune</strong></a> •
<a href="https://github.com/hellomayzune"><strong>GitHub</strong></a> •
<a href="https://orcid.org/0000-0003-0282-2633"><strong>ORCID</strong></a> •
<a href="https://scholar.google.com/citations?user=LmP8B_4AAAAJ&hl=en"><strong>Google Scholar</strong></a> •
<a href="https://www.researchgate.net/profile/May-Zune"><strong>ResearchGate</strong></a> •
<a href="https://www.linkedin.com/in/mayzune/"><strong>Linkedin</strong></a> •
</p>

<p align="center">
  <a href="https://www.imperial.ac.uk/">
    <img src="https://img.shields.io/badge/Imperial%20College%20London-000025?style=flat" alt="Imperial College London">
  </a>
  <a href="https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white" alt="Python Version">
  </a>
  <a href="https://scikit-learn.org/stable/">
    <img src="https://img.shields.io/badge/scikit--learn-v1.4.0-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  </a>
  <img src="https://img.shields.io/badge/PyTorch-v2.2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
</p>

# 🧩 Architecture Decision Record | SSOS

👩‍🔬 Author: May Zune | Imperial College Business School 2026

🆚 Version: v1 (31 August 2026) 

©️ Licence: MIT License

#### 🧭 Content

This **architecture decision record** document contains the following contents.
- [Purpose and scope](#-purpose-and-scope)
- [How to read an entry](#-how-to-read-an-entry)
- [Cross-cutting decisions](#-cross-cutting-decisions)
    - [ADR-000: Adoption of a controlled-comparison ADR format](#adr-000-adoption-of-a-controlled-comparison-adr-format)
    - [ADR-001: Global shift from exploration to pure exploitation](#adr-001-global-shift-from-exploration-to-pure-exploitation)
    - [ADR-014: Protocol adherence, by function](#adr-014-protocol-adherence-by-function)
- [F1 — Radiation-source detection](#-f1--radiation-source-detection)
- [F2 — Noisy log-likelihood](#-f2--noisy-log-likelihood)
- [F3 — Drug discovery](#-f3--drug-discovery)
- [F4 — Warehouse placement](#-f4--warehouse-placement)
- [F5 — Chemical process yield](#-f5--chemical-process-yield)
- [F6 — Recipe optimisation](#-f6--recipe-optimisation)
- [F7 — ML hyperparameter tuning](#-f7--ml-hyperparameter-tuning)
- [F8 — High-dimensional hyperparameter tuning](#-f8--high-dimensional-hyperparameter-tuning)
- [Cross-function observations](#-cross-function-observations)
- [References](#-references)

> 📌 This document is one of four. [Methodology](methodology.md) defines the general comparison protocol. This **ADR** applies it (or, where it wasn't applied, says so) to each specific, one-off decision. [Week summary](week-summary.md) is the chronological log of what was implemented. [Datasheet](datasheet.md) / [model card](model-card.md) describe the resulting data and models.

## 🎯 Purpose and scope

| Field | Answers |
|---|---|
| Context | What was observed, in one or two factual sentences |
| Options compared | Named alternatives, incumbent first |
| Protocol used | Which part of [methodology.md](methodology.md) applied, and whether the entry actually complied with it |
| Evidence | The metric values that separated the options, where they exist |
| Decision | What was chosen |
| Consequences / risk accepted | What could go wrong given this choice |
| Revisited | Whether and when this decision was later changed |

It covers 14 function/project-level decisions (ADR-001 through ADR-014) plus this ADR-000 meta-decision, broken down as:

* **One project-wide decision** (ADR-001)
* **One cross-cutting reference table** (ADR-014, protocol adherence)
* **Twelve function-specific decisions** across F1–F8

[Back To The Top](#-content)

## 📖 How to read an entry

The **Protocol used** field carries one of three labels, taken directly from [methodology.md's adherence table](methodology.md#-protocol-adherence-across-functions):

- ✅ **Compliant** — a defined comparison (typically LOOCV, per methodology §Model comparison protocol) was run before the decision, with a preserved metric value.
- ⚠️ **Partial** — a quantitative diagnostic informed the decision, but it wasn't the primary accuracy comparison the protocol specifies (e.g. a calibration check, an overfitting gap, a kernel-hyperparameter likelihood).
- ❌ **Not compliant** — no quantitative comparison was recorded; the decision was a heuristic judgement call.

None of these labels imply the decision was wrong. Several ❌ and ⚠️ entries turned out fine by the week-13 fidelity checks. The label only states what kind of evidence backed the decision at the time it was made.

[Back To The Top](#-content)

## 🔀 Cross-cutting decisions

### 💡 ADR-000: Adoption of a controlled-comparison ADR format

**Status:** Accepted (week 14, applied retroactively to all 14 entries below).

| Field | Content |
|---|---|
| Context | The original ADR (v1) recorded each decision as a retrospective narrative — what was tried, then a reflective justification — which conflated the general evaluation protocol with the specific per-function decision, and made it hard to audit which decisions had quantitative backing. |
| Options compared | (a) Keep the narrative format; (b) split a standalone methodology document from the decision log and reformat every entry as a controlled comparison. |
| Protocol used | N/A — this is a documentation-process decision, not a modelling decision. |
| Evidence | Cross-referencing the 14 v1 entries against methodology.md's protocol showed only 4 of 12 function-specific decisions cited a preserved quantitative metric (see [ADR-014](#adr-014-protocol-adherence-by-function)); the rest were narrated as reasonable-sounding choices without a stated comparison. |
| Decision | Adopt (b). [methodology.md](methodology.md) now holds the general protocol; every entry below states, explicitly, whether it followed that protocol. |
| Consequences / risk accepted | This is an honest downgrade of some entries' apparent rigor (they now read as "no comparison run" rather than being wrapped in justifying prose) rather than an upgrade — no new experiments were re-run to backfill missing comparisons for F1, F2, or F5. |
| Revisited | N/A — current format. |

### 💡 ADR-001: Global shift from exploration to pure exploitation

**Status:** Accepted (W12–W13, applied independently across all 8 functions).

| Field | Content |
|---|---|
| Context | By W12, the weekly query budget was nearly exhausted for every function. Rasmussen & Williams (GP theory), Frazier (BO tutorial), Srinivas et al. (GP-UCB regret bounds), and Shahriari et al. (BO survey) all motivate reducing exploration as budget runs out. |
| Options compared | (a) Continue exploration-weighted acquisition (EI/UCB) through W13; (b) gradually decay κ/ξ; (c) hard switch to pure exploitation (argmax μ). |
| Protocol used | ❌ Not compliant with the model comparison protocol — this is an acquisition-policy decision, which methodology.md explicitly notes has no formal comparison protocol defined (see [Acquisition function protocol](methodology.md#-acquisition-function-protocol)). |
| Evidence | Remaining-budget count only (1–2 queries left per function by W12); no quantitative comparison between (a)/(b)/(c) was run. |
| Decision | (c), applied fully from W13; (b) was used briefly in W12 for F1 (κ=0.2) and F2 (ξ=0.005) as an intermediate step. |
| Consequences / risk accepted | If the true global optimum sat outside the local basin, pure exploitation from W12 onward would miss it. F1 and F4 had previously shown premature-exploitation problems (see [ADR-008](#-f4--warehouse-placement)), which flagged this risk without stopping the project-wide shift. |
| Revisited | No — W13 was the final submission week. |

---

### ✅ ADR-014: Protocol adherence, by function

**Status:** Reference table (restates [methodology.md's adherence table](methodology.md#-protocol-adherence-across-functions) at the ADR level, with the specific decision each row maps to).

| Function | ADR entry | Comparison protocol used | Metric | Compliance |
|---|---|---|---|---|
| F4 | ADR-007 | LOOCV R² (GP vs RF) | 0.91 vs 0.70 | ✅ Compliant |
| F3 | ADR-006 | LOOCV comparison of target transforms | Metric value not preserved | ✅ Compliant in form |
| F7 | ADR-012 | LOOCV comparison (SVR ensemble vs RF vs GP-residual) | Metric value not preserved | ✅ Compliant in form |
| F2 | ADR-005 | Qualitative anisotropy diagnostic (RF partial dependence) | — | ⚠️ Partial |
| F6 | ADR-010 | Train-vs-LOO MAE gap | Gap magnitude not preserved | ⚠️ Partial |
| F8 | ADR-013 | Top-k LOO calibration check | Calibration ratio | ⚠️ Partial |
| F1 | ADR-002, ADR-003 | None recorded | — | ❌ Not compliant |
| F5 | ADR-009 | None recorded | — | ❌ Not compliant |

| Field | Content |
|---|---|
| Decision | Standardise on LOOCV as the default going forward (already stated in methodology.md); no retroactive backfilling of F1/F2/F5's missing comparisons was performed for this submission. |
| Consequences / risk accepted | Results for F1, F2, and F5's surrogate choices cannot be defended on the same evidentiary basis as F4 or F7; they are documented as heuristic choices, not as comparisons that happened to go undocumented. |
| Revisited | N/A — standing reference, updated whenever a new decision is logged. |

[Back To The Top](#-content)

## 📡 F1 — Radiation-source detection

*Radiation field, 2D. Locate contamination where only proximity gives a non-zero reading.*

### ⚖️ ADR-002: Output transform for extreme dynamic range

**Status:** Superseded twice (W4 → W6 → W9; the W9 version remained in place through W13).

| Field | Content |
|---|---|
| Context | F1's outputs span roughly 180 orders of magnitude (~1e-188 to ~1e-3 by W7), with occasional negative values (see ADR-003). A GP fit directly on raw outputs cannot resolve structure across that range. |
| Options compared | (a) W4: `log10(\|y\| + 1e-300)`; (b) W6: "signal masking" — drop points where `\|y\| < 1e-30`, fit only the remaining ~4 points; (c) W9: signed monotonic transform — add a large constant to `log10(\|y\|)` to keep the result positive, then reapply the sign, ranking all positive outputs above all negative ones. |
| Protocol used | ❌ Not compliant — no LOOCV or held-out comparison between (a), (b), (c); each replaced the last after visibly failing to separate signal from near-zero noise. |
| Evidence | No preserved metric; (b) is noted to have left only ~4 signal-bearing points, an observed data-loss count rather than an accuracy score. |
| Decision | (c), retained unchanged through W13. |
| Consequences / risk accepted | (c) introduces a hyperparameter (`OFFSET`) that must stay large enough for the data's dynamic range; an assertion was added in W11 to guard this after it surfaced as an actual failure mode. (b)'s data-discarding approach was defensible under a near-zero field but unsafe if the "narrow signal region" assumption had been wrong. |
| Revisited | No further changes after W9. |

### ⚖️ ADR-003: Response to two negative-valued query outputs

**Status:** Accepted (root cause addressed at W9, not immediately).

| Field | Content |
|---|---|
| Context | Two queries returned negative outputs: W4 `[0.513943, 0.454749]` → −3.89×10⁻⁸; W7 `[0.658199, 0.549042]` → −7.45×10⁻⁶. Both occurred under high UCB exploration bonuses (κ=5.0 at W4, κ=2.0 at W7) pushing search into unstable, unsampled regions. |
| Options compared | (a) Discard negative outputs; (b) set negative values to zero/a noise floor; (c) log and retain, resolve via the output transform (ADR-002). |
| Protocol used | ❌ Not compliant — no comparison run; the interim response (lowering κ to 0.5 by W8) was reactive, not evaluated against alternatives. |
| Evidence | None preserved; resolution validated only via an explicit ranking check in W13 (post-hoc), not at implementation time. |
| Decision | (c) — retain and log as standard observations; permanently resolved via ADR-002's W9 signed transform. |
| Consequences / risk accepted | Between W4 and W9, the surrogate had no principled way to rank small positive outputs against negative ones on a log scale. |
| Revisited | No further changes after W9. |

[Back To The Top](#-content)

## 🎲 F2 — Noisy log-likelihood

*2D, multiple local optima; tests exploration vs. exploitation.*

### ⚖️ ADR-004: Search domain extended beyond assignment bounds, then reverted

**Status:** Reverted (introduced W10, reverted W11).

| Field | Content |
|---|---|
| Context | The assignment specifies `[0,1]` per dimension. At W10, candidate generation set `DOMAIN_HI = [1.2, 1.2]`, extending 20% beyond the valid domain across a 100,000-point candidate pool before GP posterior-mean maximisation. |
| Options compared | Not a designed comparison — this was an implementation defect, likely intended to give the optimiser room off a boundary-seeking solution, not a deliberate strategy choice. |
| Protocol used | ❌ Not compliant / not applicable — treated and fixed as a bug, not evaluated as a design option. |
| Evidence | None. Available records do not confirm whether the W10 *submitted* query itself fell outside `[0,1]²` or only the candidate pool did. |
| Decision | Revert candidate generation to strict `[0,1]²` from W11 onward. |
| Consequences / risk accepted | Any candidate outside `[0,1]²` risked an invalid submission — a software defect, not a modelling trade-off. |
| Revisited | Reverted at W11; constrained through project end. |

### ⚖️ ADR-005: Surrogate family selection — GP → Random Forest → PDP-informed GP

**Status:** Accepted from W9 onward (anisotropic/ARD GP with RF-informed kernel bounds).

| Field | Content |
|---|---|
| Context | F2 began on F1's W3 GP+EI+distance pipeline, then cycled: W4 GP(Matérn+White)+Thompson → W5 RF greedy argmax → W6 RF(500 trees, depth 3)+per-tree Thompson → W7 RF(depth 3)+full-ensemble UCB → W8 anisotropic GP. The W8 diagnosis: under isotropic fits, the x₂ length-scale continually hit its upper bound (10–100), signalling unresolved anisotropy. |
| Options compared | (a) Continue RF+Thompson/UCB; (b) isotropic GP without PDP-derived bounds; (c) anisotropic GP with length-scale bounds taken from RF partial-dependence plots. |
| Protocol used | ⚠️ Partial — the anisotropy diagnosis (length-scale repeatedly saturating its bound) is a quantitative signal, but no LOOCV comparison was run between (a), (b), (c) as model families. |
| Evidence | Qualitative: length-scale saturation observed under (b); RF retained purely as a diagnostic tool, not compared head-to-head against the GP on a held-out metric. |
| Decision | (c). |
| Consequences / risk accepted | F2 shows the most surrogate-family churn in the project (GP→GP→RF→RF→RF→GP across W3–W8) before stabilising — reflecting exploratory adjustment rather than a planned comparison. |
| Revisited | Refined further in W12 (anisotropic Matérn ν=1.5 + WhiteKernel, MLE-fit, 100 restarts) — same family, improved fitting procedure. |

[Back To The Top](#-content)

## 💊 F3 — Drug discovery

*3D, minimise adverse reactions across three compounds (negated for maximisation).*

### ⚖️ ADR-006: Response to query stagnation — identical W5/W6 queries

**Status:** Accepted (addressed at W12 by switching to a local model).

| Field | Content |
|---|---|
| Context | F3 produced identical queries at W5 and W6 (`[0.378956, 0.302768, 0.459346]`) — the global ExtraTrees ensemble had converged to a fixed argmax that additional data failed to perturb. |
| Options compared | (a) Continue tuning ExtraTrees (polynomial degree, `max_features`, weighting) through W11; (b) add an exclusion filter (LCB with min-distance) to prevent literal duplicates without changing the model; (c) replace the global ensemble with a bootstrap ensemble of local Ridge models fit only inside a basin around the current best. |
| Protocol used | ✅ Compliant, in part — a LOOCV comparison across none/sqrt/log target transforms (W10–W11) selected the log transform, satisfying the LOOCV protocol for that sub-decision, though the metric value was not preserved. The subsequent family switch to (c) at W12 itself was not re-validated by LOOCV before deployment. |
| Evidence | LOOCV comparison favoured the log transform (W10–W11); the W5/W6 duplicate itself is the evidence that (a) and (b) had not resolved the underlying stagnation. |
| Decision | (c), at W12: bootstrap ensemble (n=500) of local Ridge models, 0.15-radius basin around the best point, UCB over local perturbations. |
| Consequences / risk accepted | The stagnation went unresolved as a structural issue for roughly six weeks (W6–W12) while (a) and (b) were tried. The local model trades global coverage for local responsiveness — acceptable this late under the project-wide exploitation policy (ADR-001), but would have limited exploration if adopted earlier. |
| Revisited | No further changes after W12; carried through to W13 unmodified. |

[Back To The Top](#-content)

## 🏭 F4 — Warehouse placement

*4D, ML surrogate approximates an expensive biweekly calculation over four hyperparameters.*

### ⚖️ ADR-007: Surrogate model selection via leave-one-out R² (GP vs RF)

**Status:** Accepted.

| Field | Content |
|---|---|
| Context | By W8, F4 had passed through GP+Thompson (W4), RF (W5), and an SVM-gate+GP hybrid (W6–W7). |
| Options compared | Gaussian Process (Matérn kernel) vs Random Forest. |
| Protocol used | ✅ Compliant — direct LOOCV R² comparison, the reference case for this project's [model comparison protocol](methodology.md#-model-comparison-protocol). |
| Evidence | LOO R²: **GP 0.91 vs RF 0.70** — a gap large enough, at this sample size, to be treated as decisive under methodology §Step 4. |
| Decision | GP, applied with UCB (κ=0.5) over mixed local/global candidate pools. |
| Consequences / risk accepted | LOO R² was estimated from a small sample (30 initial points + a few weekly additions by W8); no adjustment was made for this variability, a limitation shared with F3's transform comparison and flagged generally in [methodology.md](methodology.md#-methodological-limitations). |
| Revisited | No family-level change after W8; only kernel refinements through W13 (fixed vs learned length-scale, log-marginal-likelihood-based selection at W12). |

### ⚖️ ADR-008: Response to identical queries and premature pure exploitation

**Status:** Accepted (addressed at W11).

| Field | Content |
|---|---|
| Context | F4 showed two convergence symptoms ahead of the general late-project exploitation shift (ADR-001): identical W4=W5 queries, and a W10 switch to pure exploitation (argmax μ), flagged in its own changelog as intentional but later characterised as risky without an uncertainty term. |
| Options compared | (a) Retain W10's pure-exploitation argmax; (b) revert to EI (ξ=0.01) with a redesigned candidate pool (local perturbations σ=0.08 around W7/W8 queries + global random samples). |
| Protocol used | ❌ Not compliant — the reversal was a risk judgement, not a metric-driven comparison. |
| Evidence | None quantitative; evidence is the qualitative recognition that no uncertainty-aware safety net existed under (a). |
| Decision | (b), at W11. |
| Consequences / risk accepted | The W4=W5 duplicate signals an early convergence issue parallel to F3's ExtraTrees stagnation (ADR-006), resolved here at the acquisition layer rather than by replacing the surrogate family. |
| Revisited | Superseded by the project-wide exploitation shift at W12–W13 (ADR-001), by then treated as intentional and budget-justified rather than premature. |

[Back To The Top](#-content)

## ⚗️ F5 — Chemical process yield

*4D, typically unimodal, single optimum.*

### ⚖️ ADR-009: Boundary-seeking optimum left unconstrained

**Status:** Accepted (risk carried through to W13 rather than resolved).

| Field | Content |
|---|---|
| Context | Queries drifted consistently toward the domain corner `(1,1,1,1)` from early weeks; one query (W2) fell outside `[0,1]⁴` entirely. By W10, out-of-bounds historical points were excluded from training data; W13 applied a strict `[0,1]⁴` filter. |
| Options compared | (a) Treat boundary-seeking as a modelling artifact and constrain candidate generation away from the corner; (b) treat it as a genuine monotonic trend and allow candidates to target the boundary while keeping training data strictly in-bounds. |
| Protocol used | ❌ Not compliant — no LOOCV or other comparison distinguished "genuine trend" from "extrapolation artifact"; the call was a judgement based on repeated observation of the same drift. |
| Evidence | Repeated observation only: consistent drift toward `(1,1,1,1)` across W2–W12, no stable interior cluster. |
| Decision | (b), formalised at W12 ("no exploration term — justified by strong monotonic trend toward domain corner"). |
| Consequences / risk accepted | If `(1,1,1,1)` is not the true optimum and the surrogate is extrapolating near the boundary, this strategy cannot detect it — it keeps re-confirming the same corner rather than testing alternatives. This is the direct opposite of F4's ADR-008, where the same kind of boundary-convergent behaviour was treated as a risk and reversed. |
| Revisited | No — carried through to W13 (SVR + GP pure exploitation, in-bounds training data). |

[Back To The Top](#-content)

## 🍰 F6 — Recipe optimisation

*5D, negative score reframed as maximisation.*

### ⚖️ ADR-010: SVR demoted from acquisition role to interpretability-only

**Status:** Superseded (role changed at W11, SVR reintroduced W12 as part of a bootstrap ensemble).

| Field | Content |
|---|---|
| Context | F6 evaluated a hybrid GP+RF blend (W8), then a two-stage LHS design scored by GP+UCB alongside an SVR (W10). At W11, a train-vs-LOO MAE gap showed the SVR was overfitting. |
| Options compared | (a) Discard the SVR entirely; (b) retune it via regularisation; (c) keep it for SHAP-based interpretability only, remove it from the acquisition path. |
| Protocol used | ⚠️ Partial — the train-vs-LOO MAE gap is a quantitative overfitting diagnostic (methodology §Step 3), but it diagnoses one model rather than comparing named alternatives head-to-head. |
| Evidence | Large train-vs-LOO MAE gap for the SVR (magnitude not preserved in the record). |
| Decision | (c), at W11; GP (ARD Matérn + White) became the sole model driving UCB acquisition and two-stage LHS refinement. |
| Consequences / risk accepted | The overfit SVR continued to influence search indirectly via SHAP diagnostics, risking bias if its interpretations were accepted uncritically. |
| Revisited | At W12, SVR was reintroduced as a bootstrap ensemble (n=300, GridSearchCV-tuned), superseding the W11 demotion by fixing the overfitting directly rather than sidelining the model. |

[Back To The Top](#-content)

## 🔧 F7 — ML hyperparameter tuning

*6D, e.g. learning rate, regularisation.*

### ⚖️ ADR-011: Two-stage RF filtering audited and found to underperform

**Status:** Reverted (W10 approach abandoned at W11 following an explicit audit).

| Field | Content |
|---|---|
| Context | F7 used a progressively narrowing candidate-filtering pipeline from W7: RF feature importance defined a target subspace, growing by W10 into a three-stage funnel (100,000 → 10,000 → 1,000 candidates) feeding a GP on `sqrt(Y)` with custom EI. At W11, an audit against observed historical data showed the pipeline's predicted top candidate fell **below** the best already-known observed value. |
| Options compared | (a) Adjust the RF funnel's parameters (pool sizes, importance thresholds); (b) revert to the simpler W3 single-stage RF+SHAP subspace; (c) replace discrete filtering with continuous trust-region optimisation (GP with Matérn+DotProduct on `sqrt(Y)`, EI, multi-start L-BFGS-B, adaptive trust region around the top-8 historical points + 15% margin). |
| Protocol used | ✅ Compliant with the *spirit* of the protocol, though the audit was a direct ground-truth check rather than LOOCV: the candidate-set predicted maximum was compared against the best already-observed value and found to be worse — a stronger, more direct falsification than a held-out accuracy score. |
| Evidence | Predicted top candidate under the funnel < best observed value — a structural failure, not a marginal metric gap. |
| Decision | (c). |
| Consequences / risk accepted | The multi-stage funnel was built up over four weeks (W7–W10) before being abandoned — incremental refinement without ground-truth validation cannot substitute for checking predictions against reality directly. |
| Revisited | Superseded at W12 by a bootstrap SVR ensemble, chosen via a formal LOOCV comparison (ADR-012) — not because the W11 fix failed, but because the subsequent comparison step selected a different family. |

### ⚖️ ADR-012: Final surrogate family chosen via LOOCV comparison

**Status:** Accepted (implemented W12, carried through W13).

| Field | Content |
|---|---|
| Context | Following the W11 trust-region GP fix (ADR-011), W12 ran a direct three-way comparison. |
| Options compared | Bootstrap SVR ensemble vs Random Forest vs "GP-residual" pipeline. |
| Protocol used | ✅ Compliant — LOOCV comparison across three named candidates, per methodology §Step 1–2. |
| Evidence | RF and GP-residual were both evaluated via LOOCV and rejected; specific R²/error values were not preserved for this comparison (unlike F4's ADR-007). |
| Decision | Bootstrap ensemble (n=200) of SVR models, manually set C=10, γ=0.5, ε=0.01. Acquisition used EI with ξ=0.15 — roughly 15× the standard default — since the model's own LOO-CV error (~0.24) made a near-zero exploration margin meaningless. |
| Consequences / risk accepted | SVR hyperparameters were set manually rather than via grid search, an inconsistency against F6's W12 GridSearchCV-tuned SVR in the same week. |
| Revisited | W13 retained this family after re-confirming it via LOOCV against RF and near-interpolating GP baselines on the full 41-point dataset. |

[Back To The Top](#-content)

## 🧠 F8 — High-dimensional hyperparameter tuning

*8D — the highest-dimensional function in the project; global optimum is hard to find, strong local maxima accepted.*

### ⚖️ ADR-013: MC-Dropout capacity and calibration tuning

**Status:** Accepted (iteratively refined W5–W12, then replaced at W13).

| Field | Content |
|---|---|
| Context | F8 moved to an MC-Dropout neural network surrogate at W5, replacing the W4 GP+MLP hybrid, to get uncertainty estimates via stochastic forward passes in a data-scarce setting. |
| Options compared | Not a single comparison — a sequence of capacity/calibration adjustments made in response to observed calibration ratios: W5 (2 hidden layers, 50 passes, κ=3.0) → W6 (8→24→24→1 architecture) → W7 (dropout 0.2, hidden 24, κ=1.7, SHAP-scaled noise) → W8 (adaptive κ = κ_base × calibration ratio) → W10 (expanded capacity, weighted loss, 50k LHS candidates, LOO-derived adaptive κ) → W11 (dropout 0.2→0.25, hidden 24→48) → W12 (asymmetric sample weighting, 5×/10× for high-performing points, after top-k LOO evaluation showed high-scoring regions were under-predicted). |
| Protocol used | ⚠️ Partial — each step responded to a calibration-ratio measurement (methodology §Step 3), which is a genuine quantitative diagnostic, but no two architectures were ever compared head-to-head under LOOCV; capacity was tuned sequentially rather than selected by comparison. |
| Evidence | Calibration ratio at dropout=0.20: 1.10 (overconfident); improved to 1.029 at dropout=0.25 (W8/W11). Top-k LOO evaluation (W12) showed systematic under-prediction in high-scoring regions, motivating the 5×/10× loss reweighting. |
| Decision | Iteratively adjust capacity, dropout, and κ based on calibration measurements rather than fixing settings upfront. |
| Consequences / risk accepted | The W12 under-prediction finding implies earlier UCB acquisition choices under-explored promising regions near the eventual optimum. The calibration scaling became progressively automated, but the underlying κ_base (1.7 at W8/W11, reduced to 1.0 at W12) was still set manually, not by a formalised rule. |
| Revisited | At W13, the surrogate was replaced entirely with a GP (ARD Matérn kernel) fit across all 51 historical observations — a final-week family change made without a documented LOOCV comparison against the MC-Dropout model it replaced, itself a ❌ Not-compliant sub-decision within an otherwise ⚠️ Partial entry. |

[Back To The Top](#-content)

## 🔍 Cross-function observations

* **Audit-driven correction is the exception, not the rule.** F7's ADR-011 is the sole case where a candidate's prediction was directly benchmarked against an already-observed value and failed. Every other correction in this record was triggered by an internal metric shift (LOOCV, calibration ratio) or a qualitative pattern (duplicate queries, boundary drift), not by checking a prediction against ground truth before it was too late.
* **Two functions solved query stagnation at different layers.** F3 (W5=W6) and F4 (W4=W5) both produced duplicate consecutive queries from a stalled surrogate. F3 was fixed by replacing the model family (global ExtraTrees → local Ridge ensemble, ADR-006); F4 was fixed at the acquisition layer (restoring EI's exploration margin, ADR-008). Neither is compliant with the model comparison protocol as a *selection* step, though F3's transform choice within that fix was.
* **The same boundary-seeking pattern was read two opposite ways.** F4's ADR-008 treated sustained edge-convergence as premature stagnation and reintroduced exploration. F5's ADR-009 treated the identical pattern as a genuine signal and removed exploration entirely. Both were ❌ Not-compliant judgement calls; the record does not claim one was more rigorous than the other, only that they reached opposite conclusions from the same kind of evidence.
* **Compliance is concentrated, not evenly spread.** Of the twelve function-specific entries, only F4 (ADR-007) and F7 (ADR-012) ran a full LOOCV comparison with a preserved metric. F1 and F5 never ran a quantitative comparison at all. See [ADR-014](#adr-014-protocol-adherence-by-function) for the complete breakdown.

[Back To The Top](#-content)

## 📄 References

- Gebru, T., Morgenstern, J., Vecchione, B., Wortman Vaughan, J., Wallach, H., Daumé, H. and Crawford, K., 2021. Datasheets for datasets. Communications of the ACM, 64(12), pp.86–92.
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I.D. and Gebru, T., 2019. Model cards for model reporting. In Proceedings of the conference on fairness, accountability, and transparency (pp. 220-229).
- Frazier, P.I., 2018. A tutorial on Bayesian optimization. arXiv preprint arXiv:1807.02811.
- Rasmussen, C.E. and Williams, C.K.I., 2006. Gaussian processes for machine learning. Cambridge, MA: MIT Press.
- Shahriari, B., Swersky, K., Wang, Z., Adams, R.P. and de Freitas, N., 2016. Taking the human out of the loop: a review of Bayesian optimization. Proceedings of the IEEE, 104(1), pp.148–175.
- Srinivas, N., Krause, A., Kakade, S.M. and Seeger, M., 2010. Gaussian process optimization in the bandit setting: no regret and experimental design. In: Proceedings of the 27th International Conference on Machine Learning (ICML 2010). Haifa, Israel, 21–24 June 2010.
- See also [methodology.md](methodology.md) for the standardised comparison protocol these entries are checked against.

[Back To The Top](#-content)
