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

# 🤖 Model Card | SSOS

👩‍🔬 Author: May Zune | Imperial College Business School 2026

🆚 Version: v1 (31 August 2026)

©️ Licence: MIT License

#### 🧭 Content

This **model card** contains the following contents.

- [1. Model Overview](#-1-model-overview)
- [2. Intended Use](#-2-intended-use)
- [3. Model Architecture](#️-3-model-architecture)
- [4. Model Evolution](#-4-model-evolution)
- [5. Performance](#️-5-performance)
    - [Fidelity metrics comparison (W4, W9 and W13)](#-fidelity-metrics-comparison-w4-w9-and-w13)
- [6. Assumptions, Limitations, and Evaluation](#-6-assumptions-limitations-and-evaluation)
- [7. Ethical Considerations](#-7-ethical-considerations)
- [8. Model Life Cycle](#-8-model-life-cycle)


>*🔗 This model card is structured according to Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I.D. and Gebru, T., 2019. Model cards for model reporting. In Proceedings of the conference on fairness, accountability, and transparency (pp. 220-229).*

> 📌 This **model-card**  is intended to complement the accompanying [datasheet](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation), [week-summary](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation) and  [architecture-decision-record](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation). Together, these documents describe the project's development, key design decisions, model evolution, and evaluation.

## 📚 1. Model Overview

**Description:** SSOS comprises eight independently fitted surrogate models that optimise benchmark functions via sequential, Bayesian-style optimisation. Weekly refitting updates surrogate predictions and uncertainty estimates, enabling acquisition-guided candidate selection. Surrogate choices had largely converged by Week 10, with further, more targeted refinement (and one methodology audit that overturned a Week 10 decision for F7) continuing through Week 12, before Week 13 closed the project with a uniform pure-exploitation pass across all eight functions.

**Type:** A heterogeneous ensemble of supervised regression surrogates — Gaussian Processes (GP), tree ensembles, bootstrap-ensembled linear/SVR models, and Monte Carlo Dropout neural networks — embedded within Bayesian optimisation pipelines. Sequential acquisition functions drive adaptive, active-learning query selection.

**Tasks:** The framework primarily performs scalar regression with predictive uncertainty estimation, supporting sequential candidate selection under constrained query budgets. Auxiliary diagnostic tasks include feature-importance attribution and uncertainty calibration assessment, which enhance interpretability and reliability without influencing query selection.

> 📌 See  **input**, and **output**  in [datasheet](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation).

[Back To The Top](#-content)

## 🚀 2. Intended Use

SSOS performs sequential black-box optimisation, using regression-based surrogates and acquisition functions to iteratively locate inputs that maximise eight computationally expensive synthetic benchmark functions. It is designed primarily for students and instructors studying Bayesian optimisation, surrogate modelling, and optimisation under uncertainty within academic settings, and supports benchmarking and comparison of surrogate models (GPs, tree ensembles, SVR/bootstrap ensembles, Monte Carlo Dropout networks) and acquisition strategies under limited evaluation budgets. It is not intended for real-world decision-making: the benchmark functions are synthetic and carry function-specific limitations, including an unvalidated output transformation used for F3 in earlier weeks and retained duplicate query samples.

[Back To The Top](#-content)

## ⚙️ 3. Model Architecture

The framework consists of eight independent optimisation pipelines for eight functions rather than a shared architecture, each tailored to a specific benchmark function and each pairing a regression surrogate with an acquisition strategy in a weekly fit–sample–score–select cycle (see Datasheet). By the project's final weeks (W11–W13), surrogates included: 

- GP regression with Matérn and White kernels (F1, F2, F4, F6); 
- a locally-fitted bootstrap ensemble of Ridge regressors confined to a 0.15-radius basin around the incumbent best point (F3, from W12, replacing the earlier ExtraTrees-on-polynomial-features surrogate used W3–W11); 
- an SVR pre-filter followed by a GP, moving to pure exploitation once the response surface showed a consistent monotonic trend toward a domain corner (F5); 
- a continuous trust-region multi-start optimiser over a GP fit on `sqrt(Y)`, replacing the W10 two-stage Random Forest candidate filter after an audit found it excluded the true optimum (F7); 
- bootstrap ensembles (n=200–500) of SVR with grid-searched hyperparameters and LOO-calibrated uncertainty (F6, F7); and 
- a Monte Carlo Dropout multilayer perceptron in PyTorch with weighted loss favouring high-scoring points (F8). 

Acquisition functions — Upper Confidence Bound, Lower Confidence Bound, Expected Improvement, and posterior-mean maximisation — evaluate randomly sampled or Latin Hypercube candidate pools containing 5,000–100,000 points, with the highest-scoring candidate selected as the next query. In the final week (W13), every function switched to pure exploitation (no explore bonus): find the current best point across all accumulated data, fit a local weighted-linear model to estimate the ascent direction, check curvature via a local quadratic/Hessian fit, fit a local GP for the final predicted output and uncertainty, and submit a conservative step within the data-supported region.

**Data:** Initial input–output pairs for eight synthetic functions (`initial_inputs.npy`, `initial_outputs.npy`), expanded by one query–response pair per function each week, for eleven weeks of active optimisation (Weeks 3–13). Inputs are continuous vectors of 2–8 dimensions with a scalar output; no language data is involved.

**Preprocessing:** Function-specific output transformations (`arcsinh`, `logarithmic`, `log1p`, `square-root`, andn standardisation) applied during fitting; F3 used degree-2 polynomial feature expansion through Week 11, before switching to a local (untransformed, basin-restricted) Ridge ensemble from Week 12 onward.

> 📌 See  the **reasoning** behind each function's surrogate model, acquisition strategy, and any mid-project structural corrections  in [architecture-decision-record](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation).


[Back To The Top](#-content)

## 📈 4. Model Evolution

Development is documented for Weeks 3–13 (Week 1 excluded from the datasheet; Week 2 used handpicked queries). Weeks 3–4 emphasised exploration across diverse surrogates and acquisition functions: 
- F1 and F2 employed Expected Improvement combined with spatial diversity, 
- F3 combined random exploration, local exploitation, and hypothesis-driven selection, and 
- several functions trialled hybrid GP–MLP approaches. 

This exploratory phase revealed limitations, including an unsuccessful GP–UCB fit on transformed F1 outputs and an overly exploitative F4 search strategy. Between Weeks 5–7, surrogate selection became increasingly function-specific as observations accumulated: Random Forests and ExtraTrees replaced GPs for several functions, F5 adopted a two-stage SVR–GP pipeline, F7 added Random Forest-based candidate filtering ahead of GP optimisation, and F8 moved to a Monte Carlo Dropout network to support UCB-based acquisition; output transformations were integrated systematically into fitting during this stage. By Weeks 8–10, the framework had converged on stable, function-specific surrogate–acquisition pairings, with refinement focused on uncertainty estimation and acquisition-parameter calibration rather than model selection.

This convergence was not final, however. Week 11 reopened two of the Week 10 decisions: F1 reintroduced Expected Improvement (moving away from pure UCB) after diagnosing negative-output failures in earlier weeks, and, more substantively, F7's W10 two-stage Random Forest candidate filter (100k→10k→1k) was audited and found to fail, proposing candidates whose predicted value fell below the existing record; it was replaced with a continuous trust-region, multi-start L-BFGS-B optimiser. F8's dropout rate and hidden-layer width were also increased, with sample weighting added after leave-one-out calibration checks showed high-scoring points were systematically under-predicted. Week 12 brought a further, targeted surrogate change: F3 abandoned its global ExtraTrees-on-polynomial-features model (used since Week 3) in favour of a tightly local bootstrap ensemble of Ridge regressors fitted only within a 0.15-radius basin around the incumbent best point, reflecting a shift from modelling the whole response surface to modelling its neighbourhood around the optimum; F5 and F6/F7 similarly moved toward local, bootstrap-uncertainty-calibrated models (SVR ensembles) as the search space narrowed. By Week 12, F5's GP and SVR cross-check agreed strongly on a monotonic trend toward the domain corner (1,1,1,1), leading to a fully exploitative candidate (no exploration term) being submitted directly. Week 13, the final week, made this narrowing explicit and uniform: all eight functions switched to pure exploitation around a surrogate model built from every prior week's data, using a common local gradient–curvature–GP recipe rather than function-specific acquisition tuning.

Development thus progressed from broad exploration (W3–W4), through function-specific calibration (W5–W10), to a second, corrective refinement pass (W11–W12) that fixed at least one methodology error missed at the W10 "convergence" point, and finally to uniform exploitation (W13). Unresolved issues flagged early in the project — the unvalidated F3 transformation (in effect through W11) and F2/F4's exploitative acquisition strategies — were substantially addressed by the later weeks (F3's transform was retired at W12; F2 and F4 moved to GP-based exploitation with either a manual local search or a learned length-scale by W11–W12), though F1 and F3 continue to show large *relative* fidelity errors at the very small output magnitudes those functions produce (see [5. Performance](#️-5-performance))

[Back To The Top](#-content)

## ⚖️ 5. Performance

No unified metric applies, as each benchmark function is optimised independently against an unknown objective with no common ground truth. Performance is therefore assessed per function using optimisation-specific proxy and diagnostic measures rather than conventional predictive metrics.

| Function | Week 10 Method | Week 13 (Final) Method | Metric(s) | Outcome |
|---|---|---|---|---|
| F1 | GP (Matérn+White), arcsinh outputs, UCB (κ=1.5) | Local gradient/curvature + local GP, pure exploitation, signed-score ranking | Running best; query-to-best distance; rank sanity check | W10 stable, no anomalies; W11 reintroduced EI after diagnosing prior negative-output failures; W13 exploitation step improved on record (predicted 1.03E-09 vs. actual 9.92E-09) |
| F2 | GP (Matérn+White) + posterior-mean max, manual local search | Local gradient/curvature + local GP, pure exploitation | Running best; query clustering | Pure exploitation from W10 onward; W13 prediction (0.640) close to actual (0.674) |
| F3 | ExtraTrees on log(−Y) + polynomial features, LCB (κ=0.3) | Bootstrap ensemble (n=500) of local Ridge models, basin-restricted (r=0.15), pure exploitation | Running best; post-hoc transform check | Unvalidated log(−Y) transform stopped at W12 in favour of the local Ridge ensemble; duplicate queries from earlier weeks retained in the historical dataset |
| F4 | GP (fixed length-scale), pool biased to promising regions | Local gradient/curvature + local GP, pure exploitation | Running best; candidate-pool narrowing | Smooth convergence via subspace narrowing; W12 moved to a learned (rather than fixed) length-scale, selected via log-marginal-likelihood |
| F5 | SVR pre-filter → GP on log1p outputs | SVR/GP cross-check on log1p outputs, pure exploitation toward domain corner (1,1,1,1) | Running best; UCB-vs-EI comparison (W10); GP–SVR agreement (W12–13) | Stable two-stage staging by W10; by W12 GP and SVR agreed strongly on the domain-corner optimum, submitted directly with no exploration term |
| F6 | Two-stage GP (global LHS → local refinement), UCB (κ=2.5) | Local gradient/curvature + local GP, pure exploitation | Running best | Steady narrowing via RF-importance → GP; W11–W12 added a bootstrap SVR ensemble (LOO-calibrated) alongside the GP for interpretability/uncertainty checks |
| F7 | Two-stage RF filter (100k→10k→1k) → GP, EI | Trust-region multi-start GP optimisation on sqrt(Y), pure exploitation | Running best; subspace bound tracking | W10 RF filtering later audited (W11) and found to exclude the true optimum; replaced by a continuous trust-region optimiser from W11 onward |
| F8 | MC-Dropout MLP, adaptive-κ UCB via weekly LOO CV | MC-Dropout MLP (48 hidden, dropout=0.25), local gradient/curvature + local GP, pure exploitation | LOO calibration ratio; top-k calibration; SHAP | Weighted retrain improved calibration vs. ~0.68 unweighted baseline; W13 prediction (10.005) essentially matched actual (9.992) |

[Back To The Top](#-content)

### 💡 Fidelity metrics comparison (W4, W9 and W13)

Fidelity here measures how closely each surrogate's own prediction at its submitted query matched the true black-box value returned for that query — a sanity check on the surrogate, not on optimisation progress. Relative error (%) is reported per function at each of the three checkpoints; where a function's output is very close to zero, relative error becomes an unstable measure and the underlying weekly notebooks flag it explicitly (noted below).

| Function | W4 Rel. Error (%) | W9 Rel. Error (%) | W13 Rel. Error (%) | Note |
|---|---:|---:|---:|---|
| F1 | n/a (sign lost in log10 transform) | 99.6 | 89.6 | Output magnitude ~1e-9 to 1e-15; relative error is not a meaningful read at this scale — W9's notes recommend the z-score (0.51σ, well-calibrated) instead |
| F2 | 25.7 | 6.4 | 5.0 | Steady improvement in fidelity as the surrogate narrowed onto the relevant region |
| F3 | 45.1 (GP) / 144.1 (Polynomial) | 224.3 | 31.9 | W9's high figure is a near-zero-denominator artefact (notebook flags it as misleading); genuine improvement visible once the model changed to the local Ridge ensemble by W12 |
| F4 | 62.5 | 47.9 (UCB candidate) / 25.9 (exploit candidate) | 2.2 | Large fidelity gain by W13, consistent with the move to a learned length-scale GP and local exploitation |
| F5 | 80.1 (GP) / 54.8 (SVR) | 6.4 | 1.0 | Fidelity improved sharply once GP and SVR converged on the same domain-corner trend |
| F6 | 44.8 | 15.8 | 24.4 | Some fluctuation; W9 had no uncertainty estimate (SVR point prediction) for this function |
| F7 | 118.6 | 4.3 | 3.1 | Large early error reflects the log1p-transformed GP later shown (W11 audit) to be built on a flawed candidate-filtering step; fidelity stabilised once the filter was replaced |
| F8 | 5.3 (GP) / 7.4 (MLP) | 0.4 | 0.1 | Consistently the best-calibrated function throughout, reflecting the additional LOO calibration effort invested in the MC-Dropout model |

In line wtih [4. Model Evolution](#-4-model-evolution), the pattern is a broad improvement in surrogate fidelity from W4 to W13 for most functions (F2, F4, F5, F7, F8), consistent with the shift from exploratory, loosely-tuned models to locally-fitted, exploitation-focused surrogates by the final weeks. F1 and F3 are the exceptions: both operate at output magnitudes close to zero, where relative error is dominated by the denominator rather than genuine model quality, and where the underlying notebooks explicitly caution against reading the percentage figure at face value.

[Back To The Top](#-content)

## ✨ 6. Assumptions, Limitations, and Evaluation

**Assumptions:** The framework assumes that underlying response surfaces are sufficiently smooth for surrogate approximation — particularly GPs with Matérn kernels — from limited observations, and that function-specific output transformations (`logarithmic`, `square-root`, `arcsinh`) stabilise variance while preserving the relative ordering of objective values. Acquisition strategies further rely on well-calibrated predictive uncertainty, though formal calibration was evaluated only for F8, and the sequential process assumes independent query rounds, with incremental data updates not fundamentally altering the suitability of the selected surrogate. The Week 11 audit of F7's Random Forest filter shows this last assumption does not always hold in practice: a filtering strategy that appeared reasonable at Week 10 was found, on later review, to actively exclude the true optimum.

**Limitations and constraints:** Small training datasets increase sensitivity to individual observations, producing unstable surrogate fits, particularly for GPs and tree ensembles as new data are added. Exploitative acquisition strategies for F2 and F4 raised the risk of premature convergence to local optima during Weeks 8–10; both moved to GP-based exploitation with a learned or manually-checked local search by W11–W12. F2 additionally shows an unresolved discrepancy between its stated optimisation target and plotted results in earlier weeks, limiting confidence in that period's reported performance. F3's logarithmic transformation (used W3–W11) was unvalidated and would have failed for non-negative outputs; it was retired at Week 12 in favour of a local Ridge ensemble, but the retained duplicate query samples from earlier weeks continue to reduce the historical dataset's effective information content. F5's search drifted toward the domain boundary across W6–W10 and was ultimately submitted at a domain corner (1,1,1,1) from W12 onward without an explicit boundary-handling justification beyond GP–SVR agreement. Results are not directly comparable across functions, given distinct surrogates, strategies, and evaluation criteria; for the high-dimensional F8, outcomes should be read as high-quality local optima rather than guaranteed global convergence. Because Week 13 moved every function to pure exploitation with no exploration bonus, the final submitted queries carry no explicit safeguard against a locally, rather than globally, optimal result.

**Evaluation metrics:** Evaluation centres on optimisation performance and uncertainty calibration rather than conventional predictive metrics such as accuracy or F1-score. Optimisation performance is assessed by tracking the best objective value obtained each week, together with the proximity of newly selected query points to the best-known solution. For F8, additional calibration diagnostics compare mean absolute prediction error against mean predicted uncertainty (standard deviation), evaluated across the full dataset and within the highest-value samples; its weighted Monte Carlo Dropout model showed improved top-k calibration relative to an unweighted baseline (~0.68). From Week 4 onward, fidelity metrics (surrogate prediction vs. true black-box value at the submitted query) were computed weekly for most functions, providing a second, function-level check independent of the running-best trend (see [5. Performance](#️-5-performance)). Most functions were otherwise assessed qualitatively through surrogate fitting behaviour and query-selection patterns rather than a quantitative score. Performance is reported on a per-function basis rather than through a unified benchmark metric, and no demographic fairness or bias analysis was conducted, as the synthetic dataset contains no human-related attributes.

[Back To The Top](#-content)

## ✅ 7. Ethical Considerations

**Reproducibility:** Raw query–response pairs are preserved in untransformed .npy files, while all preprocessing, transformation, scaling, and filtering procedures are implemented within weekly documented notebooks using `NumPy`, `SciPy`, `scikit-learn`, and `PyTorch`, with fixed random seeds (`random_state=42`, `np.random.seed(42)`, `torch.manual_seed(42)`) applied throughout. Complete reproducibility is nevertheless not guaranteed, as query selection for F1, F3, and F5 occasionally relied on post-hoc comparison between alternative candidate strategies rather than a predefined selection criterion, and the dataset does not specify licensing or versioning information, limiting reuse beyond the course environment.

**Transparency:** Weekly notebooks document all major methodological decisions — surrogate model selection, preprocessing techniques, acquisition functions, and candidate pool sizes — with weekly developments summarised in the accompanying datasheet. Methodological revisions and known limitations are explicitly reported, including the abandoned exploration strategy for F1, the unvalidated transformation applied to F3 (used W3–W11, retired W12), duplicate query samples retained for historical accuracy, and the Week 11 audit that identified and corrected a flawed Random Forest filtering step in F7's Week 10 pipeline. Although the development process is transparent, the absence of explicit licensing terms restricts external reuse.

**Risks and mitigation:** Exploitative acquisition strategies for F2 and F4 increased the risk of premature convergence to local optima during Weeks 8–10, mitigated from W11–W12 onward by moving to GP-based exploitation with a learned length-scale and, for F2, a manual local-search cross-check. The now-retired logarithmic transformation applied to F3 through Week 11 and its retained duplicate query samples may introduce bias if the affected weeks' data is interpreted as reliable ground truth. Mitigation measures are implemented selectively, including masking near-zero observations for F1, filtering candidate points located near previously evaluated samples, and, most notably, replacing F7's flawed Random Forest candidate filter once it was audited in Week 11; however, corrections are not applied retroactively to earlier weeks' submitted queries, as known issues are retained to preserve historical reproducibility. The dataset contains no personal, sensitive, or human-subject data, and therefore presents no identifiable privacy concerns.

[Back To The Top](#-content)

## 🌱 8. Model Life Cycle

**Version control / repository:** [github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation) — weekly notebooks (W3–W13) and accompanying datasheet; see repository for version history.

**Monitoring plan:** Not applicable in production terms — SSOS was a fixed-budget academic exercise that concluded at Week 13, with no further weekly refitting planned. Any reuse of the surrogate pipelines beyond the course setting would require revisiting the unresolved items in [6. Assumptions, Limitations, and Evaluation](#-6-assumptions-limitations-and-evaluation) (in particular F5's boundary-corner result and the historical F3 duplicate/transform issues) before being monitored against new data.

[Back To The Top](#-content)