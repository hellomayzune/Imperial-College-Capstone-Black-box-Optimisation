<p align="center">
<img src="https://raw.githubusercontent.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/main/figures/SSOS_longer-aspect-ratio.png" alt="SSOS" width="600" />
</p>

<p align="center">
• <a href="https://mayzune.com/"><strong>May Zune</strong></a> •
<a href="https://github.com/hellomayzune"><strong>GitHub</strong></a> •
<a href="https://orcid.org/0000-0003-0282-2633"><strong>ORCID</strong></a> •
<a href="https://scholar.google.com/citations?user=LmP8B_4AAAAJ&hl=en"><strong>Google Scholar</strong></a> •
<a href="https://www.researchgate.net/profile/May-Zune"><strong>ResearchGate</strong></a> •
<a href="https://www.linkedin.com/in/mayzune//"><strong>Linkedin</strong></a> •
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

# 🔎 Evaluation | SSOS

👩‍🔬 Author: May Zune | Imperial College Business School 2026

🆚 Version: v1 (31 August 2026)

©️ Licence: MIT License

#### 🧭 Content

This **evaluation** contains the following contents.
- [Purpose and scope](#-purpose-and-scope)
- [Progress: how the BBO evolved week to week](#-progress-how-the-bbo-evolved-week-to-week)
    - [Methodological maturity curve](#methodological-maturity-curve)
    - [Best-observed-value trajectory by function](#best-observed-value-trajectory-by-function)
- [Real-world application fit](#-real-world-application-fit)
    - [Function-by-function assessment](#function-by-function-assessment)
    - [Cross-cutting take on real-world fit](#cross-cutting-take-on-real-world-fit)
- [Cross-audience reflections](#-cross-audience-reflections)
    - [For Students: Applied Engineering Principles](#for-students-applied-engineering-principles)
    - [For Lecturers: Pedagogical Frameworks](#for-lecturers-pedagogical-frameworks)
    - [For Machine Learning Beginners: Practical Insights](#for-machine-learning-beginners-practical-insights)
- [Successful optimisation strategies by function](#-successful-optimisation-strategies-by-function)
- [Most and least successful weeks by function](#-most-and-least-successful-weeks-by-function)
    - [Most successful week](#most-successful-week)
    - [Least successful week](#least-successful-week)
- [Strategies in professional decision-making contexts](#-strategies-in-professional-decision-making-contexts)
- [Overall outcome](#-overall-outcome)

> 📌 This **evaluation** is intended to complement the accompanying [datasheet](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation), [model-card](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation), [architecture-decision-record](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation) and [week-summary](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation). Together, these documents describe the project's development, key design decisions, model evolution, and evaluation.

## 🎯 Purpose and scope

This document evaluates the Capstone black-box optimisation project across its full 12-week timeline from Week 2 through Week 13. It examines the maturation of the underlying methodology, analyses the behaviour of all eight benchmark functions, and measures how effectively this BBO reflected real-world optimisation challenges. By isolating the specific strategies that yielded genuine performance gains, the analysis provides context for their practical applications beyond an academic setting. While the complementary **Model Card** details architectural specifications and fidelity metrics, and the **Datasheet** outlines data composition and collection protocols, this evaluation concentrates on critical outcomes—tracking weekly progress, identifying successful and unsuccessful approaches, and determining the broader significance of the results.

[Back To The Top](#-content)

## 📈 Progress: how the BBO evolved week to week

### Methodological maturity curve

| Phase | Weeks | Characteristic behaviour |
| --- | --- | --- |
| **Establishing baselines** | W3–W4 | Eight independent, hand-designed pipelines (GP+EI, GP+UCB, RF, SVR pre-filter, RF+SHAP subspace, etc.). The Week 3 strategy table logs "Key Limitations" per function from the outset — self-critique is present from the first submission. |
| **Model comparison / churn** | W5–W7 | Several functions swap surrogate families (F2 GP→RF→GP; F3 RF→ExtraTrees; F6 GP→RF→hybrid). Week 6 ties methodology explicitly to course content and reviews model-development choices critically. Week 7 adds a structured "tuning impact / main risk" table, the first place risk is tracked function-by-function rather than only by strategy. |
| **Rigour and diagnostics** | W8–W9 | Leave-one-out cross-validation (LOO-CV) is introduced for surrogate selection (F4: GP chosen over RF, LOO $R^2$ 0.91 vs 0.70), calibration checks appear for MC-Dropout (F8), and per-function "Reflection: next time..." notes begin — the first systematic retrospective layer. Fidelity metrics (surrogate mean/$\sigma$ vs. true output) start appearing from W9 onward. |
| **Correction and validity control** | W10–W12 | A genuine methodological error surfaces and is fixed: F5's Week 10 query extrapolated candidates beyond the $[0,1]^4$ domain (up to 2.0), producing an apparent record of 14,653 that could not actually be submitted, since the portal formatter clips inputs to $[0, 0.999999]$. Week 11 explicitly separates "best" from "best VALID (in-bounds)"; Week 12 formally redefines the record as `CURRENT_BEST_VALID`. Bootstrap-ensemble uncertainty (F3, F6, F7) and trust-region local optimisation (F7) also appear as the pipelines become more statistically careful. |
| **Consolidation / pure exploitation** | W13 | A deliberate strategy change: every function drops its exploration bonus (no UCB/EI margin) in favour of local-gradient / local-GP exploitation around the current record, following an explicit eight-step process — find best, define local neighbourhood, estimate ascent direction, check curvature, fit local GP, take a conservative step, report predicted improvement, rank/sanity-check results. |

[Back To The Top](#-content)

### Best-observed-value trajectory by function

The table constructs the running best for each function, using the maximum best values reported in the weekly notebook outputs to establish the benchmarks evaluated by subsequent surrogates. Instances where a function's running best remains unchanged across two or more consecutive weeks indicate that intervening queries failed to improve upon the established record.

| Fn | W2 baseline | Trajectory of the record | W13 final query result | Net change | Longest stagnation streak |
|---|---|---|---|---|---|
| **F1** Radiation field (2D) | 1.33e-22 | 7.71e-16 (W3) → flat W4–W9 → 6.88e-15 (W9→W10) → 3.17e-11 (W10→W11) → flat W11–W12 | **9.92e-09** (beat record ~313×) | ~7.5×10¹³ | 6 weeks (W4–W9) |
| **F2** Noisy log-likelihood (2D) | −0.0407 | 0.611 (W3) → flat W4 → 0.647 (W5) → flat W6–W7 → 0.683 (W8) → flat W9–W12 | 0.674 (did **not** beat 0.683) | +0.72 | 6 weeks (W9–W13 pre-query) |
| **F3** Drug discovery (3D) | −0.0506 | −0.0195 (W3) → flat W4–W6 → −0.0151 (W6→W7) → flat W7–W13 | −0.0335 (did **not** beat record) | +0.036 | 7 weeks (W7–W13) |
| **F4** Warehouse business (4D) | −7.13 | −4.03 (post-W2) → 0.204 (W3) → flat W4–W6 → 0.657 (W7→W8) → 0.664 (W8→W9) → flat W9–W13 | 0.511 (did **not** beat record) | +7.8 | 5 weeks (W9–W13) |
| **F5** Chemical process (4D) | 1,829.8 | flat W3 → 2,699 (W5) → 3,179 (W6) → flat W7 → 4,461 (W8) → 4,826 (W9) → 5,023 (W10) → [invalid spike to 14,653, corrected] → 6,470 valid (W12) → 8,233 valid (W13 pre) | **8,662** (beat record, 4.7×) | +6,833 | 2–3 weeks, plus one data-validity incident |
| **F6** Cake recipe (5D) | −1.242 | −0.353 (W3) → flat W4–W9 → −0.338 (W9→W10) → flat W10–W13 | −0.394 (did **not** beat record) | +0.85 | 6 weeks (W4–W9) |
| **F7** ML hyperparameters (6D) | 0.593 | 2.595 (W3) → flat W4–W9 → 2.749 (W9→W10) → flat W10–W12 | **3.014** (beat record) | +2.42 | 6 weeks (W4–W9) |
| **F8** Unspecified (8D) | 8.265 | 9.598 (W1, unbeaten by W2/W3) → flat W4 → 9.760 (W4→W5) → flat W5–W10 | **9.992** (beat record marginally; already 9.908 after W10→W11) | +1.73 | 6 weeks (W5–W10) |

[Back To The Top](#-content)

## 🌍 Real-world application fit

Mapping each benchmark function to an industry equivalent fundamentally alters the evaluation. Beyond simply tracking numerical improvements, a rigorous assessment must determine whether a week-long, single-query optimisation loop reflects real-world engineering workflows and whether the framework adequately accounts for the real-world cost of high-risk, uninformative queries.

### Function-by-function assessment

| Fn | Real-world domain | What a query costs in reality | Synchronized with real-world constraints | What the notebooks got right | Real-world gap |
| --- | --- | --- | --- | --- | --- |
| **F1** | Radiation field | Detector positioning for a radiation source | Physical repositioning, possible exposure risk, rare-event physics | Plausible — surveys are slow and safety-gated | Log/signed-score transform matches the physics of a mostly-null, occasionally-hot signal; conservative late-game exploitation matches "don't waste dose on a bad guess" |
| **F2** | Noisy log-likelihood | Statistical model calibration / online experiment metric | Usually cheap computation or a live A/B test | **Poor match** — real evaluation is cheap and parallelisable; nothing forces a weekly cadence | GP+UCB/Thompson is textbook-appropriate for a noisy objective |
| **F3** | Drug discovery | Candidate compound synthesis + assay | Weeks of lab time, real cost per compound, ethical/safety review | Very realistic — the canonical BBO use case | Sample-efficient, uncertainty-aware querying is exactly right for expensive points |
| **F4** | Warehouse business | Operating-policy tuning (staffing, inventory, layout) | Live business risk, opportunity cost, customer impact | Realistic — most firms can only trial one policy change per review cycle | SVM feasibility gate mirrors real operational/regulatory constraints |
| **F5** | Chemical process | Production yield optimisation | Raw materials, energy, safety envelope of a real reactor | Realistic cadence, but see gap → | Log-space GP for yield is appropriate |
| **F6** | Cake recipe | Food product development / sensory panel testing | Ingredient cost, panel scheduling | Realistic — low-stakes, good teaching case | Appropriately treated as low-risk: heavier exploration, SHAP used freely |
| **F7** | ML hyperparameter tuning | ML model tuning | Compute/training time per trial | **Poor match** — real HPO tools (Optuna, Ray Tune, ASHA) run hundreds–thousands of trials in parallel with early stopping | RF+SHAP subspace narrowing, later audited and corrected, is genuinely good BBO practice transferable to real HPO |
| **F8** | Unspecified (8D) | Generic high-dimensional engineering/formulation problem | Unstated, but non-trivial per query | Ambiguous, hard to judge directly | MC-Dropout with calibration correctly acknowledges GP kernel-hyperparameter estimation gets unreliable in 8D from ~40 points |


### Cross-cutting take on real-world fit

Black-box optimisation (BBO) proves most valuable when evaluations are expensive, slow, or high-risk. Benchmark functions F1, F3, F4, and F5 represent the strongest real-world alignments, as sample-efficient search significantly outperforms grid search or manual intuition when every query incurs substantial financial, temporal, or safety costs.

Conversely, BBO is mismatched for domains where evaluations are cheap. Problems resembling F2 and F7 are typically solved in practice by running thousands of rapid, inexpensive trials rather than treating each query as a precious resource. While a fixed weekly cadence serves as a useful pedagogical constraint for teaching sample efficiency, it does not reflect the operational reality of cheap-to-evaluate systems.

The boundary violation in F5 represents the most instructive event of the BBO, illustrating a domain where extrapolation carries real-world safety risks rather than merely numerical errors. While correcting the issue mid-BBO was necessary, the occurrence itself highlights that production systems require hard, pre-generation constraint enforcement to prevent candidates from leaving the physically valid design space entirely.

High dimensionality presents another critical constraint on sample budgets. As demonstrated by F7 and F8, higher-dimensional spaces prevent a small, fixed query budget from supporting reliable global optimisation. Real-world applications typically address this challenge through larger sample budgets, strong structural priors, or dimensionality reduction strategies.

To achieve true production readiness, several operational elements are required:

* **Governance Protocols:** Formal regulatory or human approval mechanisms before executing suggested queries.
* **Cost Accounting:** Explicit modeling of the financial or operational penalties associated with poor queries, rather than tracking only peak performance.
* **Multi-Objective Optimisation:** Frameworks capable of balancing competing objectives, such as trade-offs between yield and safety margin in F5 or throughput and labor costs in F4.

This exercise serves as an effective demonstration of BBO methodology across distinct domains. It models high-stakes optimisation compellingly for expensive or sensitive domains like F1, F3, F4, and F5, while remaining less representative of cheap-to-evaluate workflows like F2 and F7. Transitioning these models to production environments necessitates embedding strict safety boundaries directly into the candidate-generation process rather than relying on post-hoc validation.

[Back To The Top](#-content)

## 🎓 Cross-audience reflections

This BBO evaluation yields key takeaways tailored across three distinct perspectives: academic preparation, instructional design, and introductory machine learning practice.

### For Students: Applied Engineering Principles
Black-box optimisation (BBO) proves essential when evaluation steps involve slow, expensive, or destructive physical testing—such as structural simulations, wind-tunnel runs, or physical prototypes. In these settings, sample-efficient algorithms significantly outperform brute-force sweeps by maximizing the value of every trial.

Operating effectively in these environments requires embedding domain constraints directly into the candidate-generation engine. Post-hoc filtering risks executing unsafe proposals outside compliant physical boundaries. Similarly, while sensitivity analysis helps manage high-dimensional search spaces, early feature-importance rankings should be treated as dynamic hypotheses rather than permanent boundaries.

When optimisation runs hit extended plateaus, stagnation signals the need for diagnostic interventions—such as leave-one-out cross-validation and calibration checks—rather than repeated execution of identical search setups. Sequential single-query execution must match domain realities; when evaluations are inexpensive, workflows should transition to high-throughput batch optimisation. Practical deployments extend beyond single-scalar target metrics, requiring navigation across Pareto frontiers to manage complex, competing trade-offs.

### For Lecturers: Pedagogical Frameworks
The weekly single-query cadence functions as a strong pedagogical constraint, though framing it explicitly as a domain-dependent simplification helps contextualize its real-world relevance. While accurate for physical prototyping or drug discovery, it remains artificial for cheap computational simulations or automated hyperparameter tuning.

Evaluating student portfolios benefits from prioritising rigorous self-critique over raw benchmark scores alone. Structured reflection logs, leave-one-out cross-validation justifications, and the identification of invalid data points demonstrate critical engineering judgment. Boundary violations provide an ideal case study for bridging machine learning theory with safety-critical domain constraints.

High-dimensional performance degradation offers a natural transition into design-of-experiments concepts—such as Latin Hypercube sampling and sensitivity analysis—prior to introducing automated Bayesian optimisation toolkits. Additionally, mixed benchmark outcomes provide a key teaching moment: demonstrating that well-calibrated surrogates are necessary yet insufficient for search success, as calibrated uncertainty does not guarantee performance gains. The transition from single-objective benchmarking also establishes a logical foundation for subsequent modules on multi-objective optimisation and Pareto efficiency.

### For Machine Learning Beginners: Practical Insights
Iterative optimisation often exhibits non-linear progress patterns. Extended periods without improvement can precede dramatic performance jumps once the underlying surrogate accurately models the target landscape. Conversely, sudden performance spikes warrant careful validation to ensure models have not exploited unconstrained spatial boundaries.

Across diverse problem landscapes, functions display distinct behavioural profiles—ranging from sparse high-value regions and early performance plateaus to steady linear gains and sparse high-dimensional search spaces. Navigating anonymized benchmark functions mirrors exploratory data science, requiring dependency plots and feature-importance metrics to infer structural behaviour without prior domain intuition.

Methodological maturity develops iteratively as static modeling pipelines evolve toward dynamic model selection and calibrated uncertainty estimation. Ultimately, optimisation BBO highlight the stochastic nature of machine learning: even thoroughly validated models make probabilistic estimates rather than precise predictions. Documenting methodological shifts, discarded surrogate models, and corrected boundary assumptions reflects the realistic, non-linear progression of applied machine learning projects.

[Back To The Top](#-content)

## 🧪 Successful optimisation strategies by function

| Fn | Domain analogy | Successful strategy | Why it worked | Outcome |
| --- | --- | --- | --- | --- |
| **F1** | Radiation field (2D) | Rare-event detection | Log/signed-score-transformed GP + switch from UCB to EI once the dynamic-range problem was fixed | Matched the model to the data's extreme skew ($e^{-188}$ to $e^{-6}$) instead of fighting it with kernel tweaks |
| **F2** | Noisy log-likelihood (2D) | Statistical calibration | GP + UCB/Thompson sampling with distance-exclusion to avoid re-sampling known regions | Standard, well-matched approach for a smooth, noisy 2D surface |
| **F3** | Drug discovery (3D) | Expensive lab assay | ExtraTrees on polynomial features + LCB, with a late switch to a local bootstrapped-Ridge model confined to the record's basin | Cheap-per-query modelling appropriate for a small, expensive sample budget |
| **F4** | Warehouse business (4D) | Operations policy | GP selected over RF via LOO-CV, plus an SVM feasibility gate | Objective model selection rather than default preference; feasibility gate mirrors real operational constraints |
| **F5** | Chemical process (4D) | Production yield | SVR pre-filter + log-space GP, corrected after an out-of-bounds extrapolation produced an invalid "record" that had to be walked back | Catching and fixing the invalid result before acting on it protected every downstream decision |
| **F6** | Cake recipe (5D) | Product development | Early RF + Thompson sampling produced a big first jump; SVR was dropped once it reproduced a prior point and started overfitting | Willingness to retire an underperforming approach on evidence rather than sunk-cost defend it |
| **F7** | ML hyperparameters (6D) | ML tuning | RF+SHAP subspace narrowing, later audited and replaced with a continuous trust-region search once shown to under-predict the known record | Periodic re-validation of a restriction assumption, rather than trusting it indefinitely |
| **F8** | Unspecified (8D) | High-dimensional engineering | MC-Dropout MLP with $\kappa$ adapted from a measured calibration ratio, not manually tuned | Explicit uncertainty calibration compensates for GP unreliability in high dimensions with few points |

**Five strategies that generalised across functions:**

1. Transforming the model to match the underlying shape of the data provides the largest single lever for breaking through a performance plateau (F1, F5, F7). Matching structural realities directly outperforms forcing raw data into ill-fitting assumptions.

2. Search-space restrictions require periodic re-auditing rather than permanent trust (F7). Narrowing the search field works initially, but unvalidated constraints eventually trap performance in local minima as conditions change.

3. Invalid results must be caught and filtered out before triggering operational decisions (F5) . Establishing robust verification prior to action represents the single most vital governance control in any analytical workflow.

4. Underperforming approaches need to be discarded strictly on empirical evidence(F6) . Persisting with a failing pipeline out of familiarity or invested effort only drains resources without improving results.

5. Model selection relies on held-out cross-validation evidence rather than standard habits (F4). Evaluating options against strict validation protocols directly drives the largest performance jumps.

[Back To The Top](#-content)

## 📊 Most and least successful weeks by function

### Most successful week

The single largest record-breaking improvement for each function occurred during the following weeks:

| Fn | Most successful week | Jump produced | Why it stands out |
| --- | --- | --- | --- |
| **F1** | **Week 10** | 6.88e-15 → 3.17e-11 (~4,600×) | The single biggest relative leap in the BBO. Honourable mention: Week 13's final query pushed the record further, to 9.92e-09, a further ~313×. |
| **F2** | **Week 3** | −0.041 → 0.611 (+0.65) | Established almost the entire eventual gain in one query, right at the start; everything after (W5, W8) added smaller increments. |
| **F3** | **Week 3** | −0.051 → −0.0195 (+0.031) | Bigger than the only other improving week (W6→W7, +0.0044); this function's gains were almost entirely front-loaded. |
| **F4** | **Week 3** | −4.03 → 0.204 (+4.23) | The largest single jump of any function in absolute terms — recovered from a negative baseline in one query. Second-best: Week 7 (+0.45). |
| **F5** | **Week 12** | 6,470 → 8,233 valid (+1,763) | The largest *valid* jump once the Week 10 out-of-bounds spike was discounted — the real record-setting week for F5. |
| **F6** | **Week 3** | −1.242 → −0.353 (+0.889) | Nearly the whole BBO's improvement happened in this one query; six weeks of plateau followed. |
| **F7** | **Week 3** | 0.593 → 2.595 (+2.00) | The largest jump by far. Second-best: Week 13 (+0.265), which also beat the record after six flat weeks. |
| **F8** | **Week 4** | 9.598 → 9.760 (+0.162) | Narrowly the biggest jump, just ahead of Week 10's +0.148 — this function's gains were the smallest and slowest of the eight, consistent with its high dimensionality. |


Initial BBO queries delivered the single largest record improvements for five of the eight functions during Week 3, primarily by correcting poor manually calculated values established in Week 2. This early success applied to Functions 2, 3, 4, 6, and 7.

The remaining three cases—Functions 1, 5, and 8—followed a different pattern due to larger dynamic ranges and higher dimensionality. Major breakthroughs for these complex functions emerged only after several weeks of methodological refinement, such as data transform adjustments, validity checks, and calibration tuning, rather than from the initial setup.

### Least successful week

| Fn | Least successful week(s) | What happened |
| --- | --- | --- |
| **F1** | **Weeks 4–9** (6-week stagnation) | No query improved the tiny $7.71 \times 10^{-16}$ record for six consecutive submissions, despite active kernel and $\kappa$ tuning each week; the underlying dynamic-range mismatch was not addressed until the transform fix in W10. |
| **F2** | **Week 13** | The final, most careful pure-exploitation query (0.674) still fell short of the standing record (0.683) — the BBO's last chance to improve, and it missed, following a 6-week plateau. |
| **F3** | **Week 13** | The final exploitation query (−0.0335) missed the record (−0.0151) by the widest margin of the four functions that failed to close in Week 13, following the longest stagnation streak of the project (W7–W13, 7 weeks). |
| **F4** | **Week 13** | The final query (0.511) fell short of the record (0.664) by the largest absolute gap of the eight functions, after five flat weeks (W9–W13). |
| **F5** | **Week 10** | Nominally record-setting at submission time (14,653), the query was later found to extrapolate beyond the $[0,1]^4$ domain and had to be discounted — in hindsight the single most costly week in the BBO, since it consumed two further weeks (W11–W12) of correction before a valid record could be re-established. |
| **F6** | **Week 13** | The final query (−0.394) missed the record (−0.338), the third of four functions whose most careful, final-week query still lost ground relative to the standing best. |
| **F7** | **Weeks 10–12** | Following the Week 11 audit that found the Week 10 Random Forest funnel excluded the true optimum, three weeks passed with no improvement while the corrected trust-region search was brought online, before Week 13 finally beat the record. |
| **F8** | **Weeks 5–10** (6-week stagnation) | The longest stagnation streak of the project for the function with the smallest and slowest overall gains, consistent with sparse coverage of an 8D space from ~40–50 points. |


Performance regressed across four functions during final, conservative attempts due to surrogate models converging prematurely. High model calibration and low prediction errors caused search algorithms to settle into local optima, as these systems lacked mechanisms to mandate genuine re-exploration.

In contrast, successful trajectory gains across other functions resulted from deliberate, structural interventions. High-performing outcomes stemmed from fundamental adjustments—such as altering data transformations, auditing constraint validity, or correcting search-space boundaries—rather than minor refinements to already-settled methods.

[Back To The Top](#-content)

## 💼 Strategies in professional decision-making contexts

Extracting effective moves from raw optimisation trajectory data reveals five key patterns that drove major performance gains. Each pattern directly corresponds to a fundamental discipline in professional decision-making:

**1. Match underlying problem representation to data structure before optimizing**.
Dynamic output ranges spanning extreme scales fail under untransformed Gaussian Processes. Prolonged plateaus often result from kernel tinkering that ignores this core mismatch. Major breakthroughs occur only after applying transforms—such as log-space conversions—that align with the signal's true shape. In professional settings, stalled efforts usually stem from an incorrect representation of the problem rather than flawed execution.

**2. Audit and update operational constraints as new evidence emerges**.
Narrowing a search space effectively isolates high-potential areas, but fixed restrictions eventually trap performance. Auditing underlying assumptions late in a cycle often reveals under-predicted outcomes, and replacing outdated boundaries enables late-stage record gains. Scoping decisions in resource-constrained environments require scheduled re-validation checkpoints rather than permanent, unexamined boundaries.

**3. Verify result validity before taking action**.
Model extrapolations beyond valid boundaries can produce attractive figures that cannot be operationalised. Rejecting invalid metrics protects downstream choices, even at the cost of short-term apparent progress. In high-stakes decision-making and data governance, actionable accuracy outweighs inflated metrics; unverified numbers create false confidence that compromises subsequent choices.

**4. Eliminate underperforming approaches based on objective evidence**.
Dropping methods that overfit training data or reproduce existing data points maintains pipeline efficiency. Sound portfolio discipline requires terminating workstreams once marginal value halts, relying strictly on empirical performance indicators rather than invested effort.

**5. Select methodologies using held-out validation rather than default preferences**.
Choosing surrogate models based on leave-one-out cross-validation ($R^2$) rather than habit consistently yields superior outcomes and drives major performance leaps. Objective evidence based on unseen data must dictate strategic and methodological choices to ensure generalization to future scenarios.

**Failure Patterns and Strategic Synthesis**

Conversely, analytical efforts frequently plateau when surrogate models become well-calibrated around local optima without mechanisms to mandate re-exploration. Under these conditions, late-stage conservative adjustments often lead to regressive performance. This mirrors my executing established strategies competently while failing to question whether the overarching strategy remains correct.

Breakthroughs rely on structural interventions—such as transform shifts, validity audits, and search-space corrections—rather than minor refinements to settled methods. The most impactful gains stem not from highly complex models, but from robust governance behaviors: proper problem framing, periodic constraint re-validation, preemptive validity checks, evidence-based termination of failing efforts, and out-of-sample evaluation.

[Back To The Top](#-content)

## ✅ Overall outcome

The black-box optimisation (BBO) methodology demonstrates strong progression, evolving from ad hoc, function-specific pipelines toward a structured framework defined by leave-one-out cross-validation (LOO-CV) model selection, calibration checks, and fidelity tracking. A key operational strength lies in handling methodological corrections transparently—such as addressing dynamic range issues—rather than burying analytical shifts.

Final pure-exploitation phases yielded record improvements on half of the evaluated functions. Rather than limiting the findings, this mixed outcome provides crucial analytical value: it demonstrates directly that a well-calibrated surrogate model is a necessary baseline, but insufficient on its own to guarantee global optimisation success without active re-exploration mechanisms.

From an application standpoint, this framework aligns closely with high-cost physical or financial environments where evaluation budgets are strictly constrained. It offers less utility for low-cost, rapidly evaluable domains where high-throughput sampling is feasible.

Furthermore, candidate generation extrapolating beyond valid operational bounds highlights a critical requirement for high-stakes deployments. Ensuring reliability in safety- or compliance-critical environments requires hard operational constraints embedded directly into the generation algorithms, rather than relying solely on post-hoc verification and record-keeping discipline.

[Back To The Top](#-content)
