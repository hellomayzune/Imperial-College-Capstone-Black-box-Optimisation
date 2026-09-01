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

# 📅 Week Summary | SSOS

👩‍🔬 Author: May Zune | Imperial College Business School 2026

🆚 Version: v1 (31 August 2026)

©️ Licence: MIT License

#### 🧭 Content

This **week summary** contains the following contents.
- [Mapping Capstone Weeks and Module](#mapping-capstone-weeks-and-module)
- [Weekly Query Results](#-weekly-query-results)
    - [Weeks 1 and 2](#-weeks-1-and-2)
    - [Week 3](#-week-3)
    - [Week 4](#-week-4)
    - [Week 5](#-week-5)
    - [Week 6](#-week-6)
    - [Week 7](#️-week-7)
    - [Week 8](#️-week-8)
    - [Week 9](#-week-9)
    - [Week 10](#-week-10)
    - [Week 11](#-week-11)
    - [Week 12](#-week-12)
    - [Week 13](#-week-13)

> 📌 This document is a **chronological engineering log**: what was implemented and observed, week by week. It records events, not justification. Where a decision needed comparing named alternatives, that comparison and its outcome live in the [architecture decision record](architecture-decision-record.md) (e.g. "→ ADR-007"); the general procedure any such comparison should follow lives in [methodology.md](methodology.md). This log links to both rather than re-explaining them.

# Mapping Capstone Weeks and Module

| Week | Module | Module Description |
|------|--------|---------------------|
| W1 | M12 | Module 12: Bayesian Optimisation |
| W2 | M13 | Module 13: Logistic Regression |
| W3 | M14 | Module 14: Support Vector Machines |
| W4 | M15 | Module 15: Neural Networks and Deep Learning: Part One: Introduction |
| W5 | M16 | Module 16: Neural Networks and Deep Learning: Part Two: Advanced Concepts |
| W6 | M17 | Module 17: Neural Networks and Deep Learning: Part Three: Convolutional Neural Networks |
| W7 | M18 | Module 18: Hyperparameters and Hyperparameter Tuning |
| W8 | M19 | Module 19: Foundations of Generative AI and Large Language Models |
| W9 | M20 | Module 20: Advanced Generative AI and Large Language Models |
| W10 | M21 | Module 21: Transparency and Interpretability |
| W11 | M22 | Module 22: Unsupervised Learning: Part One: Clustering Techniques |
| W12 | M23 | Module 23: Unsupervised Learning: Part Two: Principal Component Analysis |
| W13 | M24 | Module 24: Reinforcement Learning |
| **W14** | **GitHub** | Module 25: Final BBO Capstone Project Submission: GitHub Repository |

[Back To The Top](#-content)

# 🎯 Weekly Query Results
🔗 This document summarises the results of weekly query [notebooks](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/codes).

## 📚 Weeks 1 and 2

(Module 12: Bayesian Optimisation and Module 13: Logistic Regression)

Week 1: no machine learning experiments were conducted. Week 2: initial manual analysis completed using `initial-data-exploratory-data-analysis.ipynb` and `initial-data-OLS-regression.ipynb`, establishing the initial understanding of the dataset.

### OLS Results by function

| Function | Description | Samples | Dimensions | R² | Adjusted R² | Prob(F-stat) | Significant Inputs |
|----------|-------------|--------:|-----------:|----:|------------:|-------------:|--------------------|
| F1 | Radiation field | 10 | 2D | 0.037 | -0.239 | 0.878 | None |
| F2 | Unspecified field | 10 | 2D | 0.569 | 0.446 | 0.0525 | Input_1 |
| F3 | Drug discovery | 15 | 3D | 0.368 | 0.196 | 0.154 | Input_3 |
| F4 | Warehouse business | 30 | 4D | 0.579 | 0.512 | 0.000164 | Input_1, Input_2, Input_4 |
| F5 | Chemical process | 20 | 4D | 0.662 | 0.572 | 0.00174 | Input_3, Input_4 |
| F6 | Cake recipe | 20 | 5D | 0.647 | 0.521 | 0.00696 | Input_4, Input_5 |
| F7 | Unspecified field | 30 | 6D | 0.346 | 0.176 | 0.103 | Input_5 |
| F8 | Unspecified field | 40 | 8D | 0.899 | 0.873 | 2.42 × 10⁻¹³ | Input_1, Input_2, Input_3, Input_7 |

**Observations (factual):** sample size grows only from 10 to 40 across F1→F8 while dimensionality grows from 2D to 8D — search-space volume grows exponentially against a linearly-growing budget (see [methodology.md §Shared experimental setting](methodology.md#-shared-experimental-setting) for why this rules out held-out validation). F1, F3, F5, F7 show outliers and mixed-sign linear trends; F2, F4, F6, F8 show little linear correlation. Neither group shows a consistently strong linear fit, motivating a surrogate-based approach over linear regression for all eight functions.

After Week 10, the first two weeks were revisited from a beginner's perspective, producing `w0-beginner-mind.ipynb` — a comparative framework contrasting GP-BO (explicit uncertainty via the acquisition function), Random Forest SMBO (tree-disagreement uncertainty), Logistic Regression (binary reframing), SVR (no native uncertainty), and MLPRegressor (flexible nonlinear surrogate, no native uncertainty).

[Back To The Top](#-content)

## 🔍 Week 3

(Modules 12–14: Bayesian Optimisation, Logistic Regression, Support Vector Machines)

| # | Query Strategy | Exploration–Exploitation | Key Limitations |
|---|---|---|---|
| **F1** | GP (Matérn + White kernel, default length-scale) + EI on 20k random candidates; top 15% by EI, then farthest-from-existing-samples selected | EI + diversity hybrid | Isotropic smoothness assumed with default kernel bounds; no transform/noise floor for near-zero values; 20 optimiser restarts from only 10 observations risks unstable hyperparameters |
| **F2** | Same GP + EI + distance-maximisation pipeline as F1 | Same as F1 | Direct reuse of the F1 pipeline without adapting to F2's response characteristics |
| **F3** | Degree-2 polynomial regression blended with GP + EI; candidate pool mixes global random, local perturbations, and directed candidates; blend weights from 5-fold CV R² | Mixes exploration (random), local exploitation (perturbations), and hypothesis-driven exploitation (directed candidates) | 5-fold CV R² highly variable at n=15; hardcoded "increase B and C" candidates encode an early, possibly misleading prior; quadratic regression cannot capture more complex nonlinearity |
| **F4** | GP kernel chosen via 5-fold CV over 3 kernels, then GP + UCB (κ=2.0) over 10k uniform candidates | Exploration-oriented (high κ, global candidates) | No diversity filtering — selected queries can sit close to previous samples; kernel selection via CV is noisy at n=30 |
| **F5** | SVR ranks a locally-generated candidate pool (perturbations + directional moves around incumbent); top 3000 candidates scored by GP + EI (ξ=0.005) | Strong exploitation — every candidate generated near the current best | Cannot discover optima outside the incumbent's neighbourhood; SVR pre-filter errors propagate since discarded candidates are never seen by the GP |
| **F6** | ARD GP + EI (ξ=0.05) over uniform + LHS + perturbation-around-top-5 candidate pool | Balanced: global coverage, space-filling, local exploitation | Larger ξ shifts EI toward exploration; no diversity constraint; GP hyperparameters hard to estimate reliably at n=20, 5D |
| **F7** | RF + SHAP identifies influential variables/directions, defining a reduced search region; GP (log1p outputs) + EI optimised within that subspace, 500 L-BFGS-B restarts | Strong exploitation — optimisation confined to a model-defined subspace | Subspace restriction depends entirely on one RF trained on 30 samples; incorrect importance/direction estimates could make the true optimum unreachable, with no global fallback |
| **F8** | Two pipelines: (a) ARD GP + UCB (κ=2.5), 4 dims pinned to historical extremes, 4 free, 3 random restarts; (b) SVR pre-filter → GP + EI over local perturbations near incumbent | (a) partial exploration + strong exploitation; (b) almost entirely exploitation | The two pipelines were never reconciled, making the submitted query ambiguous; fixing half the variables to historical extremes assumes boundary optima |

[Back To The Top](#-content)

## 🔍 Week 4

(Module 15: Neural Networks and Deep Learning — Part One)

Candidate-generation and optimisation strategies were compared across all eight functions this week.

| # | Prediction / Optimisation Method | Candidate Generation Strategy |
|---------:|----------------------------------|-------------------------------|
| **F1** | GP (Matérn ν=2.5) on log₁₀ outputs + UCB (κ=5.0, exploration-heavy) | Full-space LHS sweep (10k points); candidates within 0.15 of any existing sample masked out, forcing exploration into unsampled regions |
| **F2** | GP (Matérn+WhiteKernel noise) + Thompson sampling | Full-space LHS (15k points), no pinning |
| **F3** | GP EI (ξ=0.001, exploitation-biased) blended 50/50 with quadratic polynomial surrogate | Local cluster (σ=0.03) around the W3 best point + directed perturbations (B↓, C↑) from polynomial coefficients |
| **F4** | GP (Matérn) + Thompson sampling | Full-space LHS (20k points), deliberately unconstrained to test whether the W2 region merits further exploration |
| **F5** | Linear regression (search direction) → SVR pre-filter → GP EI (ξ=0.005) | Local (σ=0.03) + directed clusters around incumbent best; x3 fixed at 0.999999; SVR filters to top 3,000 before GP EI |
| **F6** | GP with ARD Matérn + EI (ξ=0.02) | Sensitivity-weighted: x1 full [0,1] range (highest ARD sensitivity); x2/x5 σ=0.10; x3/x4 σ=0.05 (lower sensitivity) |
| **F7** | GP EI (log1p outputs, ξ=0.01) + MLP regressor (50/50 blend) | HP1 restricted to [0,0.10] (highest RF importance, decreasing direction); HP5 to [0,0.25]; HP6 to [0.70,1.00]; HP2–HP4 uniform |
| **F8** | GP (ARD Matérn) UCB (κ=2.0) + MLP regressor (50/50 blend) | x2, x4, x8 fixed at 0; x5 fixed at 0.999999; x1, x6 sampled from Gaussians centred on the W1 solution; x3, x7 uniform |

W4 is when F1's two negative-valued outputs first appeared under high-κ UCB (→ [ADR-003](architecture-decision-record.md#-adr-003-response-to-two-negative-valued-query-outputs)) and when F2's surrogate-family churn began (→ [ADR-005](architecture-decision-record.md#-adr-005-surrogate-family-selection--gp--random-forest--pdp-informed-gp)).

[Back To The Top](#-content)

## 🔍 Week 5

(Module 16: Neural Networks and Deep Learning — Part Two: Advanced Concepts)

| # | Prediction / Optimisation Method | Hierarchical Feature Learning | Explore–Exploit Trade-off |
|---------|----------------------------------|-------------------------------|---------------------------|
| **F1** | GP (Matérn ν=2.5, length scale fixed at 0.15) on log-transformed \|outputs\|; UCB (κ=4.0) over 10,000 random candidates | None — raw 2D inputs used directly | Explicit via UCB κ; fixed length scale also stabilises exploration, addressing a previously observed over-smoothing failure mode |
| **F2** | Random Forest regression; greedy argmax of predicted mean over 50,000 random candidates | None — raw features only | Predominantly exploitative; exploration arises only indirectly from candidate-pool breadth |
| **F3** | Random Forest on degree-2 polynomial features (A, B, C, A·B, A·C, B·C, A², B², C²); argmax over 10,000 global candidates | Manual feature engineering via PolynomialFeatures | Predominantly exploitative; global candidate generation provides search coverage |
| **F4** | RF surrogate; uncertainty from tree-to-tree disagreement (`rf_predict_with_std`); EI (ξ=0.01) over Latin Hypercube candidates | None — raw inputs | Explicit: EI balances predicted gain against ensemble uncertainty; LHS promotes uniform exploration |
| **F5** | Two-stage: SVR pre-filters (top 3,000), then GP (log-transformed outputs, Matérn) via EI (ξ=0.005) | None | Exploration constrained by design — candidates generated near the current best; EI adds a modest uncertainty component |
| **F6** | RF surrogate; Thompson sampling using one randomly selected tree rather than the ensemble mean | None | Explicit stochastic exploration — tree disagreement is largest in sparsely sampled regions |
| **F7** | Two-stage: RBF interpolator (multiquadric) scores the full candidate set, then RF re-ranks the top 1,000 | None | Explicit two-stage: RBF extrapolation encourages exploration, RF filtering exploits learned patterns |
| **F8** | MC-Dropout MLP (2 hidden layers, 50 stochastic forward passes); UCB (κ=3.0) | Yes — hidden layers learn nonlinear representations directly, unlike F1–F7's raw/hand-engineered features | Explicit and model-native via MC-dropout uncertainty + UCB |

[Back To The Top](#-content)

## 🔍 Week 6

(Module 17: Neural Networks and Deep Learning — Part Three: Convolutional Neural Networks)

Four new samples were incorporated this week; feature engineering and acquisition design were reviewed per function.

| **#** | **Dim** | **Samples** | **Surrogate model(s)** | **Feature engineering** | **Acquisition** |
|---|:---:|:-------:|--------------------|---------------------|-------------|
| F1 | 2D | 14 | Gaussian Process (Matérn, fixed length scale) | Signal masking (drop \|y\| < 1e-30) | UCB |
| F2 | 2D | 14 | Random Forest (500 trees, depth 3) | — | Thompson sampling (per-tree) |
| F3 | 3D | 19 | ExtraTrees (200 trees) | Degree-2 polynomial interactions | Argmax over RF predictions |
| F4 | 4D | 34 | SVM classifier (gate) + ARD GP (refine) | Threshold-based binary labelling | Expected Improvement (EI) |
| F5 | 4D | 24 | SVR + GP (log space) | Log transform of output | EI |
| F6 | 5D | 24 | Random Forest (500 trees) | Maximin LHS candidate design | Full-ensemble UCB |
| F7 | 6D | 34 | RF (importance) → ARD GP | Importance-driven subspace narrowing | UCB, multi-start L-BFGS-B |
| F8 | 8D | 40 | Dropout MLP (8→24→24→1) | Standard scaling | UCB via MC-dropout uncertainty |

[Back To The Top](#-content)

## ⚙️ Week 7

(Module 18: Hyperparameters and Hyperparameter Tuning)

This week's question: *how has hyperparameter tuning changed the query strategy compared to earlier rounds?*

| **#** | **Surrogate** | **Key hyperparameters tuned** | **Week 7 tuning impact** | **Main risk in Week 7** |
|---|-----------|---------------------------|----------------------|---------------------|
| F1 | GP (Matérn+White) on log-absolute outputs | Kernel choice; UCB κ=2 | Reverted to the Week 3 kernel, recovering performance lost the previous iteration | No cross-validation; predictions remain nearly degenerate across weeks |
| F2 | Random Forest | Max depth=3; UCB κ=1.5 | Replaced Thompson sampling with full-ensemble UCB, improving uncertainty estimation | No held-out validation of generalisation |
| F3 | ExtraTrees + polynomial features | `max_features`: None vs. sqrt | Resolved missed higher-order feature interactions | Assessed on training data only; 9 features vs ~20 observations increases overfitting risk |
| F4 | SVM gate + GP (EI) | SVM probability threshold; GP α | Increased GP α to smooth noisy responses; added global LHS exploration | Repeated recommendations (W4, W5) indicate optimisation stagnation |
| F5 | SVR filter + GP (EI) | SVR pool=3000; EI ξ=0.005 | Restored the original candidate pool size, improving search coverage | Objective values plateau around 2700–3200, suggesting insufficient exploration |
| F6 | Random Forest | Max depth=5 (new); UCB κ=1.5 | First explicit regularisation of the RF model | Prior unconstrained model may have overfit; extent remains unquantified |
| F7 | Two Random Forests → subspace → GP + UCB | Subspace bounds; UCB κ=2 | RF-based filtering isolated the dominant HP1 region before GP optimisation | Subspace constructed from only ~10 observations |
| F8 | MC-Dropout MLP | Dropout=0.2; hidden=24; UCB κ=1.7; SHAP-scaled noise | Introduced SHAP-informed candidate generation | SHAP explanations from only 100 samples/20 background points; x5, x8 importance unreliable |

[Back To The Top](#-content)

## ⚙️ Week 8

(Module 19: Foundations of Generative AI and Large Language Models)

| # | Method | Exploration–Exploitation | Overfitting / Limitation | Reliability Step Taken |
|----------|---------------------------------|--------------------------|--------------------------|-------------------------|
| **F1** | GP (Matérn + White, optimised hyperparameters) + UCB (κ=0.5) on tight local candidates | Exploitation-heavy: local σ=0.015 search, low κ | Output spans e⁻¹⁸⁸ to e⁻⁶ raw, risking GP overconfidence | Predicted mean hard-clipped to [floor, 0] before ranking; floor from the largest gap in sorted log values |
| **F2** | Anisotropic GP (PDP-derived length-scale bounds from RF) + UCB (κ=0.1) | Exploitation-heavy (κ reduced to 0.1) | x₂ length-scale repeatedly hit its upper bound (10–100) | RF partial dependence used to derive GP length-scale bounds → ADR-005 |
| **F3** | ExtraTrees (WLS-weighted, polynomial features) + LCB (κ=0.5), exclusion filter | Exploitation-focused (UCB → LCB) | ExtraTrees underestimated best observations by ~37× | WLS-style weighting (1/residual²); exclusion filter prevents proposals within 0.01 of prior points |
| **F4** | GP (selected over RF via LOO R²) + UCB (κ=0.5) | Mixed: local pools around 3 promising regions, low κ | RF LOO R² (0.70) underperformed GP LOO R² (0.91) | Model selected by LOOCV, not assumption → **ADR-007 (✅ compliant)** |
| **F5** | SVR pre-filter (top 3000) → GP (log outputs) + EI (ξ=0.005) | Mostly exploitative; x₃/x₄ concentrated near upper boundary | SVR underpredicted by ~150 units, systematic bias | SVR ε retuned; GP/SVR predictions cross-checked at all historical samples |
| **F6** | Hybrid GP+RF (inverse-RMSE weighted) + EI, multi-restart L-BFGS-B | Explicit exploration: 30 multi-start EI runs + 0.05 min-separation filter | Only 20 observations in 5D — both GP and RF individually unreliable | SVR acquisition discarded after reproducing prior points; replaced with hybrid GP+RF + diversity filter |
| **F7** | Two-stage RF filtering → GP + UCB (κ=1.5) | More exploitative (κ 2.0→1.5) | RF impurity importance may exaggerate some variables | Impurity importance cross-checked with permutation importance and SHAP |
| **F8** | MC-Dropout MLP + UCB, adaptive κ (κ_base × calibration ratio) | SHAP-weighted perturbations (Pool A) + structured Pool B sweep | MC-dropout overconfident at dropout=0.20 (calibration ratio 1.10) | Dropout increased to 0.25 (ratio → 1.029); κ adapts from measured calibration → ADR-013 |

[Back To The Top](#-content)

## 🧮 Week 9

(Module 20: Advanced Generative AI and Large Language Models)

Each function's surrogate was assessed for robustness vs. compute cost:

| # | Method | Behaviour driver | Exploration balance | Robustness vs. cost |
|:-------:|--------------------------|------------------------------|:--------:|:--------------------------:|
| **F1** | GP, UCB/EI | GP uncertainty only | Very conservative | Low |
| **F2** | GP, RF-informed kernel bounds | Adaptive length scales from RF PDP | Mild | Low–medium |
| **F3** | ExtraTrees, polynomial features | Ensemble disagreement + feature interactions | Moderate | Medium |
| **F4** | GP with diagnostic visualisations | Human interpretation of response surfaces | Conservative | Medium |
| **F5** | GP, constrained local/boundary search | Boundary pinning + structured perturbations | Moderate | Medium–high |
| **F6** | RF surrogate + SHAP | Feature-importance explanations | Moderate | High |
| **F7** | Dataset filtering → RF optimisation | Adaptive data selection | High | High |
| **F8** | MC-Dropout NN | Learned representations + stochastic uncertainty | Highest | Highest |

**Note (factual, not a comparison):** these columns are qualitative judgements made per function at the time, not scores from a common metric — they should not be read as a ranking under [methodology.md's comparison protocol](methodology.md#-model-comparison-protocol).

[Back To The Top](#-content)

## 📝 Week 10

(Module 21: Transparency and Interpretability)

The pre-final review examined each function's optimisation strategy, key assumptions, data gaps/bias, and main limitation:

| **#** | **Strategy** | **Key Assumption** | **Main Data Gap / Bias** | **Key Limitation** |
|----------|-----------------------|----------------|----------------------|----------------|
| **F1** | GP with UCB/EI, signed monotonic transform | Follows W9 method | Low robustness | Sensible exploitation rather than a wide, poorly-calibrated jump |
| **F2** | GP (Matérn+White) + posterior-mean maximisation (multi-start L-BFGS-B), plus manual local search | Pure exploitation; domain extended to 1.2 (later reverted, ADR-004) | Queries cluster along one narrowing path | Target [0.70, 0.10] vs. plot label [0.69, 0.20] mismatch |
| **F3** | ExtraTrees (`log(-Y)`, polynomial features) + LCB (κ=0.3) | `log(-Y)` transform used before validated | Two weekly queries are exact duplicates | Transform breaks if Y ≥ 0; validation came too late |
| **F4** | GP (Matérn, fixed length-scale) + EI computed, query selected via `argmax(mu)` | Candidates biased toward W7/W8 neighbourhoods | Training mixes very different output regimes | Purely greedy on μ; EI computed for comparison only, not used → flagged in ADR-008 |
| **F5** | SVR pre-filter → GP (log1p, Matérn) + UCB (κ=2.5); query via `argmax(mu_log)` | Candidates pushed beyond [0,1] to 2.0 | No data exists above 1.0 for any input | `argmax(mu_log)` selected as the safer option |
| **F6** | Two-stage LHS (global+local) scored by GP (isotropic Matérn) + UCB (κ=2.5); SVR fit but unused | Isotropic length-scale (reverted from ARD) | SHAP computed but disconnected from search | Local refinement may miss a distant 5D optimum |
| **F7** | Two-stage RF filtering (full → HP1-filtered) → GP (`sqrt(Y)`, Matérn+DotProduct) + custom EI | Low-HP1 subset assumed to hold the optimum | Filtered RF trained on very few points | 100k→10k→1k funnel permanently excludes ~99% of search space → audited and reverted in W11, ADR-011 |
| **F8** | MC-Dropout MLP (weighted loss) + LHS pool (50k) + UCB (adaptive κ from LOO calibration) | Up-weighting high-y points improves precision | Weighting thresholds set after seeing the data | SHAP importance computed but unused this week |

Datasheet and model card development began this week, alongside the above review.

[Back To The Top](#-content)

## 🧮 Week 11

(Module 22: Unsupervised Learning — Part One: Clustering Techniques)

Each surrogate was refined further; query history since W2 was also reviewed through a clustering lens (do successive queries converge, split, or sit as unresolved outliers).

### W11 configuration

| # | Surrogate | Acquisition | Key hyperparameters | Notes |
|---|---|---|---|---|
| **F1** | GP, Matérn ν=2.5 | EI (ξ=0.5), signed-score transform | Fixed length-scale (≈0.05, 3-point distance estimate), `OFFSET=300` | Assertion added to guard against `OFFSET` too small for dynamic range |
| **F2** | GP, Matérn (anisotropic/ARD) | UCB (κ=1.0), min-distance exclusion | `min_dist=0.1` | Length-scale bounds from RF PDP output ranges |
| **F3** | ExtraTrees (200 trees), degree-2 polynomial features, `log(-Y)` | LCB (κ=0.3), min-distance filter | `max_features='sqrt'`, `min_samples_leaf=2`, `min_distance=0.001` | LOOCV comparison across none/sqrt/log transforms picked log → ADR-006 (✅ compliant) |
| **F4** | GP, Matérn ν=2.5 | EI (ξ=0.01), switched from UCB | `ConstantKernel × Matern(length_scale=0.4, fixed)`, α=1e-4, 20 restarts | Candidate pool: local perturbations (σ=0.08) + global random → ADR-008 |
| **F5** | SVR pre-filter → GP (log1p outputs) | EI (ξ=0.005) | SVR: C=100, γ=scale, ε=0.01; GP: length_scale=0.2, 25 restarts | Out-of-bounds historical points excluded from training; hard [0,1] assertion added |
| **F6** | GP, ARD Matérn+White (SVR for SHAP only) | UCB (κ=2.5), two-stage LHS | GP length_scale=ones(5), bounds (1e-2,1e2); noise=1e-3; refine radius=0.05, top_k=50 | SVR diagnosed as overfitting via train-vs-LOO MAE gap → ADR-010 (⚠️ partial) |
| **F7** | Trust-region GP, Matérn+DotProduct on `sqrt(Y)` | EI (ξ=0.01), multi-start L-BFGS-B, adaptive trust region | length_scale=[0.3]×6, bounds (0.05,5.0); trust region from top-8 points + 15% margin | Two-stage RF funnel audited against historical data and found to underperform; replaced this week → ADR-011 |
| **F8** | MC-Dropout MLP (hidden=48, dropout=0.25) | UCB, adaptive κ=κ_base(1.7) × calibration ratio | lr=1e-3, 1500 epochs, weighted loss (5×/10×), 50k LHS pool | Dropout 0.2→0.25 and hidden 24→48 for calibration → ADR-013 |

**Fidelity check (post-hoc, per [methodology.md](methodology.md#-post-hoc-fidelity-checks)):** comparing W11 surrogate predictions against the actual returned outputs for the 8 submitted queries gave overall R²=0.930 (MAE≈201.1, RMSE≈567.3), dominated by F5's larger output scale. Excluding F5, R²≈0.888 with Spearman ρ=0.64 — the surrogates ranked functions less reliably than raw error scores suggest; F4 predicted the wrong sign entirely, while F2 and F8 tracked within ~2% and ~1% respectively. As noted in methodology.md, this is a single-week observation, not a significance test.

[Back To The Top](#-content)

## 🧮 Week 12

(Module 23: Unsupervised Learning — Part Two: Principal Component Analysis)

Several functions moved to local, bootstrap-based ensembles (F3, F6, F7) in place of a single global model; F5 committed fully to pure exploitation toward a domain corner.

### W12 configuration

| # | Surrogate | Acquisition | Key hyperparameters | Notes |
|---|---|---|---|---|
| F1 | GP, Matérn ν=2.5, fixed length scale (from spread of top-3 recent points); target=signed_score | Dual-arm: UCB (exploit, local grid) + EI (explore, global) | κ=0.2, ξ=0.5 | Length scale set by geometry, not MLE |
| F2 | GP, anisotropic Matérn ν=1.5+WhiteKernel, MLE-fit (100 restarts) | EI vs. observed record, 300k candidates+L-BFGS-B, min-distance exclusion | ξ=0.005, length_scale bounds (0.03–1.0), min-dist=0.015 | Tiny ξ since goal is "stay above max" after 3 failed exploration weeks |
| F3 | Bootstrap ensemble (n=500) of local Ridge models, 0.15-radius basin around best | UCB on local perturbations | basin_radius=0.15, α=0.02, n_boot=500, κ=1.0 | Switched from global ExtraTrees to local model + bootstrap uncertainty → ADR-006 |
| F4 | GP, Constant×Matérn ν=2.5 (learned length scale)+WhiteKernel, chosen via higher log-marginal-likelihood | EI vs. best-so-far, mixed pool; final pick restricted to tight σ=0.02 band | length_scale bounds (0.05–2.0) | Learned length scale improved fit vs. fixed prior |
| F5 | SVR (RBF)+GP (Matérn ν=2.5, log1p target), both in-bounds only, cross-checked via LOO | Pure exploitation — argmax(GP mean); SVR as agreement check | SVR: C=100, γ=scale, ε=0.01; GP length_scale=0.3 | No exploration term — monotonic trend toward (1,1,1,1) → ADR-009 (❌ not compliant) |
| F6 | Bootstrap ensemble (n=300) of SVR, GridSearchCV-tuned; σ calibrated against LOO residuals | EI (ξ=0.01), 2-stage LHS pool | grid-searched C/γ/ε | Two-step tuning: hyperparameters (grid search) + calibration (LOO) |
| F7 | Bootstrap ensemble (n=200) of SVR (C=10, γ=0.5, ε=0.01), selected over RF/GP-residual via LOOCV | EI (ξ=0.15, from LOO error scale), blended pool | C=10, γ=0.5, ε=0.01, ξ=0.15 | ξ scaled ~15× default since CV error (~0.24) makes near-zero margin meaningless → ADR-012 (✅ compliant) |
| F8 | MC-Dropout MLP (48 hidden, dropout=0.25), weighted MSE (5×/10×), σ from 100 MC passes, SHAP | UCB, adaptive κ=κ_base × (LOO error/σ ratio), 50k LHS candidates | hidden=48, dropout=0.25, lr=1e-3, epochs=1500, κ_base=1.0 | κ self-calibrates from full weighted LOO retraining → ADR-013 |

For F5, cross-checking the GP against the SVR agreement model showed close agreement near the top-performing region — including the corner candidate (1,1,1,1) — diverging more at low-to-mid predicted yields, consistent with their different loss functions.

**Fidelity check:** against the actual W12 outputs, the surrogate portfolio achieved R²=0.972 (MAE≈161.3, RMSE≈455.8). Per-function relative error ranged from ~1.3% (F8) to ~223% (F4, surrogate mean 0.62 vs. near-zero actual 0.192), with F1's error dominated by its ~5×10⁻¹³ actual magnitude.

[Back To The Top](#-content)

## ✅ Week 13

(Module 24: Reinforcement Learning)

The final query round: every function switched to pure exploitation (no UCB/EI exploration bonus), per the project-wide [ADR-001](architecture-decision-record.md#-adr-001-global-shift-from-exploration-to-pure-exploitation). A single general procedure was applied consistently across F1–F8:

1. **Find the current best point** — combine original samples with all prior-week observations, identify the highest-output input.
2. **Focus on the local neighbourhood** — examine points close to the current best; remove negative/failure and noise-floor values where appropriate.
3. **Estimate the direction of improvement** — fit a weighted local-linear model for the ascent direction.
4. **Check the local curvature** — fit a regularised local-quadratic model, inspect the Hessian/eigenvalues for consistency with a local maximum.
5. **Build a local GP** — fit to nearby, signal-bearing observations for predicted output and uncertainty.
6. **Choose a conservative exploitation step** — move from the current best along the estimated improvement direction, within the data-supported region.
7. **Predict the new output** — use the GP for the point estimate, compare against the current best, report prediction intervals.
8. **Evaluate/rank observations** — apply the `signed_score` transform where outputs span extreme ranges, so successes rank above failures.

F5 continued its W9–12 methodology: SVR (RBF) agreement check alongside a Matérn GP decision-maker, both trained on `log1p`-transformed, in-bounds data; final query via pure `argmax(GP posterior mean)` over ~21,000 local candidates around `[0.976764, 1, 1, 1]`.

### Fidelity Metrics (Surrogate Mimicry) — Week 13

| Function   | Prediction |   Actual |
| ---------- | ---------: | -------: |
| Function 1 |   1.03E-09 | 9.92E-09 |
| Function 2 |   0.639679 | 0.673566 |
| Function 3 |  -2.28E-02 | -0.03348 |
| Function 4 |   0.499878 | 0.511339 |
| Function 5 |  8575.9543 | 8662.483 |
| Function 6 |  -2.98E-01 | -0.39403 |
| Function 7 |    2.91953 | 3.013542 |
| Function 8 |  10.005407 | 9.991664 |

By the final week, surrogates tracked black-box outputs closely across the board: F2, F4, F5, F8 landed within ~1–5% of actual; F3, F6, F7 stayed within the same order of magnitude and correct sign. As stated in [methodology.md](methodology.md#-post-hoc-fidelity-checks), this closing check is prospective confidence-building for the deployed models, not retrospective justification for how each week's model was originally selected — that evidentiary basis is recorded per-function in the [ADR](architecture-decision-record.md).

[Back To The Top](#-content)
