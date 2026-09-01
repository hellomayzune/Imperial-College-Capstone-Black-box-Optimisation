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

# 🧪 Methodology | SSOS

👩‍🔬 Author: May Zune | Imperial College Business School 2026

🆚 Version: v1.1 (1 September 2026) — added Tutorials section, cross-referencing the two diagnostics notebooks

©️ Licence: MIT License

#### 🧭 Content

This **methodology** document contains the following contents.
- [Document overview](#-document-overview)
- [Tutorials](#-tutorials)
- [Experimental setting](#-experimental-setting)
- [Standard model comparison protocol](#-standard-model-comparison-protocol)
- [Acquisition function protocol](#-acquisition-function-protocol)
- [Global exploration-to-exploitation Policy (ADR-001)](#-global-exploration-to-exploitation-policy-adr-001)
- [Post-hoc fidelity checks](#-post-hoc-fidelity-checks)
- [Protocol adherence across functions](#-protocol-adherence-across-functions)
- [Methodological limitations](#-methodological-limitations)
- [References](#-references)


## 🎯 Document overview

**Methodology:** Details the standard procedure for validating and comparing candidate models and acquisition strategies.

**Architecture Decision Record (ADR):** Documents specific weekly functional decisions, comparing available options under a set protocol.

**Week Summary:** Provides a chronological timeline of implementation and weekly observations.

**Datasheet / Model Card:** Outlines the current state of the dataset and model portfolio.

[Back To The Top](#-content)

## 📓 Tutorials

Two notebooks document the exploratory work behind the protocol below. They are walkthroughs, not audited weekly submissions—read them to understand how the standard model comparison protocol evolved, not as evidence for specific ADR entries.


**[`w0-beginner-mind-diagnostics-tutorial.ipynb`](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/codes): Breadth-First Diagnostics**

* **`w0-beginner-mind-diagnostics-tutorial.ipynb`** — A breadth-first comparison of five surrogate families (Gaussian Process + EI, Random Forest SMBO + UCB, SVR, MLPRegressor, and Logistic Regression) across all eight functions ($F_1$–$F_8$) from `CapstoneFunction_from_Numpy.xlsx`. It ranks GP + EI first due to its explicit, calibrated uncertainty estimates, while placing Random Forest SMBO second for its ensemble-based uncertainty. This notebook provides the informal rationale for selecting GP + EI as the baseline surrogate in the protocol below. 

**[`w0_bayesopt-diagnostics-tutorial.ipynb`](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/codes): Depth-First Pipeline**

* **`w0_bayesopt-diagnostics-tutorial.ipynb`** — The depth-first follow-up: a complete GP-BO pipeline per function featuring a one-at-a-time search across 19 kernels, 2 $x$-transforms, and 8 $y$-transforms. It handles model selection using analytical (closed-form) LOO-NLPD instead of brute-force refitting, generates 3-point Kriging-Believer query batches per function, ranks dimension sensitivity for $d > 2$, and contrasts EI, UCB, and PI alongside Kriging-Believer and Local-Penalization diversity strategies. Its analytical-LOOCV pipeline directly underpins the protocol's LOOCV step, while its output (`proposed_queries.csv`) defines the template for weekly query submissions. 

**Relationship to the Protocol**
Neither notebook serves as an individual ADR entry. Whenever a function's actual weekly evaluations deviated from or bypassed the tutorial pipeline, those departures are recorded transparently under *Protocol Adherence Across Functions* rather than obscured here.

[Back To The Top](#-content)

## 🧱 Experimental setting 

- **Query budget:** Exactly 1 query per function per week over 13 weeks. Evaluation decisions are non-repeatable with a strict, hard budget.
- **Sample Size:** Starts between 10 (F1, F2) and 40 (F8) points, adding 1 point per week to a maximum of $n \le 53$. Because samples remain small throughout, standard train/test splits cannot be used for validation.
- **Dimensionality:** Ranges from 2D (F1, F2) to 8D (F8). The search space expands exponentially while the query budget only grows linearly, introducing the curse of dimensionality from Week 1.
- **Ground Truth:** True outputs are unknown prior to submission. Query selection protocols must rely solely on previously observed data.

[Back To The Top](#-content)

## 🔬 Standard model comparison protocol

* **Step 1: Define a Small Candidate Set**
  * Limit comparisons to 2–3 models max (the incumbent plus 1–2 challengers).
  * *Reason:* Small sample sizes ($n \le 53$) increase the risk of choosing a model that wins purely by luck.

* **Step 2: Use Leave-One-Out Cross-Validation (LOOCV)**
  * Standard train/test splits remove too much data to fit stable surrogates.
  * LOOCV maximizes training data while ensuring the model is never tested on points it has already seen.


* **Step 3: Measure Primary Metrics**
  * Primary accuracy metrics are **LOO $R^2$** and **LOO MAE**.
  * Secondary diagnostics complement—but cannot replace—these metrics:

| Diagnostic | Primary Use | Cannot Be Used For |
| --- | --- | --- |
| **Log-marginal-likelihood** | Tuning GP kernels/hyperparameters within the same model family | Comparing across different model families |
| **Train vs. LOO error gap** | Spotting overfitting in a single model | Ranking two different candidate families |
| **Calibration ratio** (error / predicted $\sigma$) | Validating confidence intervals | Judging point-prediction accuracy |

* **Step 4: Apply the Decision Rule**
  * A challenger only replaces an incumbent if it shows a large, meaningful metric improvement (e.g., LOO $R^2$ jumps from $0.70$ to $0.91$). Small gaps (like $0.02$–$0.03$) do not justify a swap.
  * Defaults to the incumbent when results are close or unmeasured (parsimony principle).
  * **Exception:** Structural failures—such as excluding the top observed point or generating identical duplicate queries across consecutive weeks—trigger an immediate replacement regardless of metrics.

* **Step 5: Scope Boundaries**
  * This protocol applies solely to surrogate model selection. It is **not** used to evaluate acquisition functions (e.g., UCB vs. EI vs. Thompson sampling).

[Back To The Top](#-content)

## 🎯 Acquisition function protocol

Given a fitted surrogate mean μ(x) and, where available, an uncertainty estimate σ(x), each function used one of:

| Acquisition | Form | Used when |
|---|---|---|
| Upper Confidence Bound (UCB) | argmax μ(x) + κ·σ(x) | Uncertainty estimate is trusted; κ set the exploration bonus |
| Expected Improvement (EI) | argmax E[max(0, f(x) − f_best − ξ)] | Similar role to UCB; ξ sets the required improvement margin |
| Thompson sampling | Sample one plausible function from the posterior, argmax that sample | Ensemble/tree models where per-tree or per-sample draws are cheap |
| Pure exploitation | argmax μ(x) | Budget nearly exhausted, or the model's own error made an exploration margin meaningless (see ADR-012) |

No formal comparison protocol (analogous to the LOOCV rule above) was defined for choosing *between* these four; the choice was made per function, per week, based on remaining budget and observed surrogate behaviour, and is recorded as a discrete engineering decision in the ADR and week summary rather than as a controlled comparison.

[Back To The Top](#-content)

## 📉 Global exploration-to-exploitation Policy (ADR-001)

* **Weeks 1–11 (Exploration):** All functions used an active exploration bonus ($\text{UCB } \kappa > 0$ or $\text{EI } \xi > 0$).
* **Weeks 12–13 (Exploitation):** Functions transitioned to pure exploitation (transition began in Week 12 and completed across all functions in Week 13).
* **Project-Wide Application:** Applied centrally as a unified policy across all eight functions, documented under **ADR-001**.

[Back To The Top](#-content)

## ✅ Post-hoc fidelity checks

* **Definition:** A secondary evaluation performed *after* submitting a query, comparing the true observed output against the surrogate model's prior prediction (recorded in the weekly fidelity tables).
* **Timing & Scope:** This occurs after query selection using newly available data. It is **not** part of the model comparison protocol.

**Key Constraints & Purpose**

* **Single-Point Validation:** Tests the active surrogate against reality one observation at a time.
* **No Statistical Weight:** Because each week adds only a single data point, fidelity checks cannot prove one model is statistically superior to another.
* **Forward-Looking Calibration:** Used to calibrate confidence in future predictions, not to justify past choices retrospective to query generation.
* **ADR Role:** Any fidelity metrics cited in an ADR entry serve as evidence for future model confidence, not retroactive validation of the selection.

[Back To The Top](#-content)

## 📋 Protocol adherence across functions

Reported here once, honestly, rather than re-litigated in each ADR entry:

| Function | Comparison protocol actually used                                                          | Metric                      | Compliance with Standard model comparison protocol |
| -------- | ------------------------------------------------------------------------------------------ | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **F1**   | None recorded                                                                              | —                           | ❌ Not compliant                                                                                                                                                                                                                             |
| **F2**   | Qualitative anisotropy diagnostic (RF partial-dependence informing GP length-scale bounds) | —                           | ⚠️ Partial — informed a hyperparameter, not a model choice                                                                                                                                                                                  |
| **F3**   | LOOCV comparison of output transforms (none / sqrt / log)                                  | Unspecified LOO metric      | ✅ Compliant in form; metric value not preserved                                                                                                                                                                                             |
| **F4**   | LOOCV R² (GP vs RF)                                                                        | LOO R²: 0.91 vs 0.70        | ✅ Compliant                                                                                                                                                                                                                                 |
| **F5**   | None recorded                                                                              | —                           | ❌ Not compliant                                                                                                                                                                                                                             |
| **F6**   | Train-vs-LOO MAE gap                                                                       | Gap magnitude not preserved | ⚠️ Partial — overfitting diagnostic, not a family comparison                                                                                                                                                                                |
| **F7**   | LOOCV comparison (SVR ensemble vs RF vs GP-residual)                                       | Unspecified LOO metric      | ✅ Compliant in form; metric value not preserved                                                                                                                                                                                             |
| **F8**   | Top-k LOO calibration check                                                                | Calibration ratio           | ⚠️ Partial — calibration diagnostic, used to justify reweighting, not family choice                                                                                                                                                         |

**Summary:** Three of eight functions (**F1, F2, F5**) never ran a documented quantitative comparison before choosing or changing surrogate family; their ADR entries say this directly rather than implying otherwise.


[Back To The Top](#-content)

## ⚠️ Methodological limitations

* **High-Variance LOO Estimates:** LOOCV on small samples ($n = 10\text{--}53$) is noisy. Large gaps (e.g., $R^2$ of $0.91$ vs. $0.70$) indicate clear signals, but small differences are within standard noise margins.
* **Potential Data Leakage:** In some instances, features or output transforms were selected after inspecting full datasets before running LOOCV. This introduces mild optimistic bias, which is explicitly noted in those ADR entries.
* **Lack of Multiple-Comparison Corrections:** No formal adjustments (such as Bonferroni corrections) were applied when evaluating more than two models in a single week. Keeping candidate sets small (Step 1) was the only mitigation.
* **Single-Point Fidelity Checks:** Weekly fidelity metrics reflect single observations rather than statistical distributions and cannot support significance claims.
* **Retrospective Protocol Creation:** Drafted in Week 14 after all submissions concluded, this protocol serves as an auditing framework for recorded gaps rather than a retroactive revision of past decisions.

[Back To The Top](#-content)

## 📄 References

- Rasmussen, C.E. and Williams, C.K.I., 2006. Gaussian processes for machine learning. Cambridge, MA: MIT Press.
- Frazier, P.I., 2018. A tutorial on Bayesian optimization. arXiv preprint arXiv:1807.02811.
- Shahriari, B., Swersky, K., Wang, Z., Adams, R.P. and de Freitas, N., 2016. Taking the human out of the loop: a review of Bayesian optimization. Proceedings of the IEEE, 104(1), pp.148–175.
- Srinivas, N., Krause, A., Kakade, S.M. and Seeger, M., 2010. Gaussian process optimization in the bandit setting: no regret and experimental design. In: Proceedings of the 27th International Conference on Machine Learning (ICML 2010). Haifa, Israel, 21–24 June 2010.
- Hastie, T., Tibshirani, R. and Friedman, J., 2009. The Elements of Statistical Learning. 2nd ed. New York: Springer. (LOOCV and small-sample validation.)

[Back To The Top](#-content)
