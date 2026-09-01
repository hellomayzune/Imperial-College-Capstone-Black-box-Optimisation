<p align="center">
<img src="https://raw.githubusercontent.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/main/figures/ssos-image.png" alt="SSOS" width="600" />
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
Repositories for the <a href="https://www.imperial.ac.uk/lifelong-learning/courses/professional-certificate-ml-and-ai/"><strong>Professional Certificate in Machine Learning and Artificial Intelligence</strong></a> BBO Capstone Project 
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
  <a href="https://pytorch.org/docs/stable/index.html">
    <img src="https://img.shields.io/badge/PyTorch-v2.2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
</p>


# ⭐ Sequential Surrogate-Optimisation Suite for the 8 Black-Box Functions (SSOS)

**✨ Rationale and objectives**

How Artificial Intelligence (AI) shapes everyday life over the coming decades will depend not only on better Machine Learning (ML) models, but on the ability to optimise and improve them in practice. This Black-Box Optimisation (BBO) Capstone Project takes that as its starting point, using a black-box optimisation problem to build data-driven decision-making skills for improving systems whose internal workings are unknown—a capability that matters more as modern AI systems grow more complex and widely used. Approached from an early stage in the ML/AI field, the project develops practical skills in applying surrogate optimisation methods and deep learning models under a limited evaluation budget, refining strategies iteratively through experimentation and evaluation, working through real-world challenges like noisy, non-linear, and high-dimensional problems, and understanding the exploration–exploitation trade-off, all captured in a well-documented optimisation framework.

**📌 Highlight**

- SSOS is a sequential surrogate optimisation framework for maximising eight black-box functions under a severely constrained evaluation budget.

- Eight black-box functions (F1–F8), 2D → 8D, each solved with a tailored surrogate model — Gaussian Processes, Random Forest/ExtraTrees, SVR, or MC-Dropout MLP. Acquisition strategies matched to each problem: UCB, Expected Improvement, Thompson sampling, posterior-mean argmax.

- A real, imperfect optimisation record — including a documented mistake (Week 10 candidate-filtering on F7 excluded the true optimum, audited and fixed in Week 11) with full transparency: [Methodology](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/documentation/methodology.md), [Datasheet](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/documentation/datasheet.md), [Model Card](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/documentation/model-card.md), [Week Summary](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/documentation/week-summary.md), [Architecture Decision Record](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/documentation/architecture-decision-record.md), and [BBO Evaluation](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/documentation/bbo-evaluation.md) explaining why, not just what.
All 8 functions improved on their initial best after 13 weeks — see [Results summary](#-results-summary).

> ✍️ This [blog](https://mayzune.com/2026/08/29/finding-a-mountain-peak-in-zero-visibility/) reflects the journey of finding a mountain peak in zero visibility.

> 🏆 The presentation **bbo-capstone-project-presentation-week12-module23.md** was submitted at Week 12, Module 23 for Stage 2: Required capstone component 23.2. It received Exemplary Assignment Badge.


👩‍🔬 Author: May Zune | Imperial College Business School 2026


🆚 Version: v1 (31 August 2026)

©️ Licence: MIT License

📬 Contact: https://www.linkedin.com/in/mayzune/

---

### 🌐 Repositories

This **repository** contains the following contents. 

- [Project overview](#-project-overview)
  - [Challenge objectives and methodology](#-challenge-objectives-and-methodology)
  - [Results summary](#-results-summary)
  - [Initial data](#-initial-data)
  - [Extended data and reproducibility](#-extended-data-and-reproducibility)
- [Repository map](#-repository-map)
- [Documentation](#-documentation)
- [Software packages](#-software-packages)
- [AI use disclosure](#-ai-use-disclosure)
- [References and acknowledgement](#-references-and-acknowledgement)

[Back To The Top](#-repositories)


## 🚀 Project overview

This project applies a **Sequential Surrogate-Optimisation Suite (SSOS)** to optimise eight black-box functions over a 13-week capstone. Each week, new input variables are selected based on previous observations with the goal of maximising the function output. As the project progresses, the 2D to 8D dimensionality of the search space increases, introducing greater optimisation complexity. Rather than focusing solely on finding the global optimum, this project emphasises a systematic, data-driven optimisation process. The objective is to demonstrate practical machine learning and black-box optimisation skills through thoughtful experimentation, iterative model refinement, and well-documented decision-making. 

<img src="https://raw.githubusercontent.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/main/figures/mz-bbo-abstract.png" alt="abstract" width="900" />

[Back To The Top](#-repositories)

### 🎯 Challenge objectives and methodology

This **SSOS** capstone project applies black-box optimisation to maximise 8 unknown functions (F1–F8, ranging from 2D to 8D) under a strict budget of one query per function per week. 

An Exploratory Data Analysis (EDA) study is conducted via [initial-data-exploratory-data-analysis.ipynb](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/codes/initial-data-exploratory-data-analysis.ipynb) and [initial-data-OLS-regression.ipynb](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/codes/initial-data-OLS-regression.ipynb) using the initial data to uncover variable correlations, boundaries, and landscape structures before running iterative optimisation algorithms.

The [initial-data-pca-gpr-rbf.ipynb](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/codes/initial-data-pca-gpr-rbf.ipynb) notebook addresses two related but fundamentally different questions:

1. **PCA:** Which directions explain the greatest amount of variation in the input data?
2. **GPR:** Which input variables actually matter for predicting the output?

The central conclusion of `initial-data-pca-gpr-rbf.ipynb` is **"high variance does not necessarily mean high predictive importance."** This preliminary study argues that PCA-based variance reduction should not simply replace supervised feature selection using GPR/ARD.

The two tutorial notebooks detail the development of a Bayesian optimization framework: **[`w0-beginner-mind-diagnostics-tutorial.ipynb`](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/codes): Breadth-First Diagnostics** broadly compares five surrogate models across eight functions to justify using Gaussian Process + EI as the baseline, while **[`w0_bayesopt-diagnostics-tutorial.ipynb`](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/codes): Depth-First Pipeline** deeply evaluates kernel and transform combinations to establish the protocol's LOOCV selection and submission pipeline. Neither notebook serves as an audited weekly submission, but together they provide the structural rationale and methodology for the protocol's execution.

Each function is tackled with a tailored surrogate model (Gaussian Processes, Random Forest/ExtraTrees, SVR, or MC-Dropout MLP) paired with an acquisition strategy (UCB, EI, Thompson sampling, or posterior-mean argmax) to balance exploration and exploitation. The main challenges are the severe query budget, complete uncertainty about the underlying function structure, and the risk of surrogate misspecification or biased search-space narrowing (e.g. faulty output transforms, over-aggressive filtering, or clustering around prior queries) that could permanently exclude regions containing the true optimum.

[Back To The Top](#-repositories)

### ⚡ Results summary

The table summarises the input dimensionality, number of initial samples, and search domain for each function, alongside a comparison of the weekly BBO best results against the initial best, highlighting the corresponding normalised score achieved from the ceiling (defined from best competition results).

Leaderboard scores (where #1 is the best among 34 participants) are included to transparently show which ADR performs well—or poorly—across different functions.

| Function | Input | Output | Samples | Study Area | Initial Best | BBO Best | Ceiling | Normalised Score | Week Best | Leaderboard |
|:---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| F1 | 2D | 1D | 10 | 📡 Radiation field | 7.71E−16 | 9.90E−09 | 2 | 0.000000005 | 13 | 15 |
| F2 | 2D | 1D | 10 | 🎲 Noisy log-likelihood | 0.611205 | 0.683188 | 0.76 | 0.4838 | 8 | 11 |
| F3 | 3D | 1D | 15 | 💊 A drug discovery | −0.034835 | −0.015072 | 0 | 0.5673 | 6 | 16 |
| F4 | 4D | 1D | 30 | 🏭 Warehouse business | −4.025542 | 0.664078 | 0.68 | 0.9966 | 8 | 4 |
| F5 | 4D | 1D | 20 | ⚗️ Chemical process in a factory | 1088.86 | 14653 | 8663 | 1.7908 (outperform) | 10 | 1 |
| F6 | 5D | 1D | 20 | 🍰 A cake recipe | −0.714265 | −0.338381 | −0.130 | 0.6433 | 9 | 17 |
| F7 | 6D | 1D | 30 | 🔧 Tuning six hyperparameters | 1.364968 | 3.013542 | 3.4 | 0.8101 | 13 | 5 |
| F8 | 8D | 1D | 40 | 🧠 Eight input parameters | 9.598482 | 9.991664 | 10 | 0.9792 | 13 | 5 |

Review the results against initial data analysis.

| Function | Archetype | Evidence from initial-data-pca-gpr-rbf.ipynb |
|---|---|:---|
| **F1** | `hairline_spike` | The outlier is a razor-thin stripe against a near-zero baseline — "hairline" captures how narrow it is in only 2D. |
| **F2** | `rising_edge` | Clean monotone ramp toward a high band near the domain's edge. |
| **F3** | `corner_lock` | The optimum sits in a smooth bowl tucked in one corner of the input space, and the search locks onto it almost instantly. |
| **F4** | `hidden_bowl` | A smooth single-peak shape that's invisible in 1D/linear views and only emerges under a nonlinear fit. |
| **F5** | `edge_surge` | Same ramp-to-boundary behaviour as F2, but with a much sharper, larger-magnitude surge near the edge (the 1088.9 outlier). |
| **F6** | `noise_ceiling` | Best value is hit almost by chance on the first sample; everything after is noisy scatter below that ceiling. |
| **F7** | `ghost_spike` | Same spike signature as F1, but higher-dimensional and noisier — the spike is harder to "see," more like a ghost in the data. |
| **F8** | `deep_lock` | Same early-converge behaviour as F3, but the surrogate locks onto it with much higher confidence (CV R² = 0.996) despite 8 dimensions — a "deeper," more robust lock. |

<img src="https://raw.githubusercontent.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/main/figures/final-bbo-result-plot-summary.png" alt="result" width="1050" />



[Back To The Top](#-repositories)

### 📁 Initial data

Imperial College (classroom.emeritus.org) provides initial data as .npy files for each black-box function; use `np.load()` in Python to examine them.

- `initial_inputs.npy` — contains the initial input samples.
- `initial_outputs.npy` — contains the corresponding function outputs.


### 📝 Extended data and reproducibility

Each week, a new set of query points is generated for each black-box function, and their predicted outputs are estimated by the optimisation model. After submission, the corresponding true function outputs are returned as `inputs.txt` and `outputs.txt` files, allowing the predictions to be evaluated. The newly acquired input-output pairs are then appended to the existing dataset, progressively increasing the number of training samples and enabling the optimisation model to improve over successive weeks.

Weekly submissions are available as `w1-functions-all.ipynb`, `w2-functions-all.ipynb`, etc.

For transparency and easier comparison across weeks, the returned output values are manually recorded and appended to the existing dataset instead of being imported directly from the weekly `.txt` files. This approach provides a clear history of optimisation progress and makes it straightforward to compare each week's performance against the initial maximum value. The example below demonstrates how the weekly query points and their corresponding outputs are combined with the initial dataset using NumPy:

```python
f1_inputs = np.load('initial_data/function_1/initial_inputs.npy')
f1_outputs = np.load('initial_data/function_1/initial_outputs.npy')

# Weekly queries and returned outputs
prev_queries_1 = np.array([...])
prev_outputs_1 = np.array([...])

# Combine with the initial dataset
all_inputs_F1 = np.vstack([f1_inputs, prev_queries_1])
all_outputs_F1 = np.hstack([f1_outputs, prev_outputs_1])
```
This cumulative dataset is then used to train the surrogate model in subsequent optimisation iterations.

> 💡 This [w0-beginner-mind.ipynb](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/codes/w0-beginner-mind.ipynb)  provides a comparative experimental framework for evaluating several different approaches to surrogate-based black-box optimisation.


[Back To The Top](#-repositories)

## 📦 Repository map

In total, the project started with 16 NumPy files (8 input files and 8 output files), organised by function as shown in the project structure below. This **repository** is structured as follows:
```
Imperial-College-Capstone-Black-box-Optimisation
│
├── 📖 readme.md                                            ← 👋 hello! 
│
├── 🚀 codes                        
│   ├── w1-functions-all.ipynb                              ← 🧮 see weekly code file
.... the same format for every week.
│   ├── w13-functions-all-added-fidelity-metrics.ipynb      
....
│   ├── w0-beginner-mind-diagnostics-tutorial.ipynb         ← 🪷 Shoshin! Breadth-First Diagnostics. Tutorial for auxiliary evidence,.
│   ├── w0_bayesopt-diagnostics-tutorial.ipynb              ← ✏️ Depth-First Diagnostics Pipeline. Tutorial using EI + Kriging-Believer
│   ├── initial-data-exploratory-data-analysis.ipynb        ← worked in week 1/2
│   ├── initial-data-OLS-regression.ipynb                   ← worked in week 1/2
│   ├── initial-data-pca-gpr-rbf.ipynb                      ← review pca - initial data
│   ├── extended-data-weekly-outputs-compare.ipynb          ← weekly query result
│   ├── extended-data-weekly-outputs-pca-week27.ipynb       ← review pca - extended data
│   ├── analytical-comparison-caroline-may.ipynb            ← ⚖️ academic credibility
│
├── 📚 documentation                        
│   ├── methodology.md                                            ← Methodology
│   ├── architecture-decision-record.md                           ← Architecture decision record
│   ├── bbo-evaluation.md                                         ← BBO evaluation
│   ├── datasheet.md                                              ← Datasheet
│   ├── model-card.md                                             ← Model card
│   ├── week-summary.md                                           ← Week summary
│   ├── analytical-comparison.md                                  ← ⚖️ academic credibility
│   ├── bbo-capstone-project-presentation-week12-module23.md      ← 🏆 Exemplary assignment badge
│
├── 🧩 figures                                              ← see all visualisation outputs
│
├── 💾 initial_data/                                       ← available via https://classroom.emeritus.org/          
│   ├── function_1/
│   │   ├── initial_inputs.npy
│   │   └── initial_outputs.npy
│   ├── function_2/ ... function_8/
```

[Back To The Top](#-repositories)


## 📚 Documentation

| Document | What it covers |
|---|---|
| **🧪 Methodology** | The standard procedure for validating and comparing candidate models and acquisition strategies: the LOOCV-based model comparison protocol (candidate set size, primary/secondary metrics, decision rule, structural-failure exceptions), the acquisition function protocol (UCB, EI, Thompson sampling, pure exploitation), the project-wide exploration-to-exploitation policy (ADR-001), post-hoc fidelity checks, an honest per-function table of protocol adherence, and methodological limitations (small-sample LOO variance, potential leakage, no multiple-comparison correction). Drafted retrospectively in Week 14 as an auditing framework, not a revision of past decisions. |
| **⚙️ Datasheet** | Motivation, composition, collection, preprocessing, intended uses, and limitations of the SSOS dataset — eight synthetic black-box optimisation benchmarks (radiation detection, drug discovery, warehouse planning, chemical process optimisation, ML hyperparameter tuning, etc.), each starting from a small query–response set and growing incrementally as new evaluations are added. Structured per Gebru et al. |
| **🔧 Model card** | The SSOS framework itself: eight independently fitted surrogate models (Gaussian Processes, tree ensembles, SVR, Monte Carlo Dropout neural networks) paired with acquisition strategies (UCB, Expected Improvement, posterior-mean maximisation), plus weekly retraining and intended use as a teaching/research benchmarking tool rather than a real-world deployment system. Structured per Mitchell et al. |
| **📅 Week summary** | Week-by-week narrative (Weeks 1–13) of how the methodology evolved — from exploratory analysis and simple regression to GPs, Random Forests, ExtraTrees, SVR, and MC Dropout NNs — including the rationale behind each change, the exploration/exploitation balance, feature engineering, hyperparameter tuning, and lessons learned. |
| **🧩 Architecture decision record** | 14 decisions total: 1 project-wide decision (global shift from exploration to pure exploitation late in the project), 1 cross-cutting model-selection reference table, and 12 function-specific decisions (F1–F8). Each entry logs the final decision, alternatives considered, risks accepted, and later revisions — notably the **F7 Week 10/11 correction**, where a candidate-filtering strategy was audited and replaced after it was found to exclude the true optimum. Grounded in Rasmussen & Williams, Frazier, Srinivas et al., and Shahriari et al. Focuses on *why*, complementing the datasheet/model card's *what*. |
| **🔥 BBO evaluation** | Full-timeline assessment (Week 2–13) of what worked and what didn't — tracking a maturity curve from ad hoc, per-function pipelines to a structured framework with cross-validation, calibration checks, and fidelity tracking. Notes that the final pure-exploitation phase gave record gains on half the functions, illustrating that good calibration alone isn't sufficient for global convergence without re-exploration. |
| **🏷️ Analytical comparison** | Week-by-week code-similarity check ([analytical-comparison-caroline-may.ipynb](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/codes) and [analytical-comparison.md](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation)) against Caroline's reference pipeline scripts, establishing academic integrity requirement, similar problem structure and independent development of the modelling approach and custom fidelity metrics while acknowledging the referenced work. |


[Back To The Top](#-repositories)

## 🧰 Software packages
The following software packages are used in this project.
| Package | Used for |
|---------|----------|
| **NumPy** | Array operations and numerical computing. |
| **Matplotlib** | Data visualisation and plotting. |
| **scikit-learn** | Machine learning models and utilities, including SVR, SVC, Pipeline, MLPRegressor, ExtraTreesRegressor, StandardScaler, LinearRegression, LogisticRegression, Ridge, RandomForestRegressor, GradientBoostingRegressor, KNeighborsRegressor, PolynomialFeatures, KFold, LeaveOneOut, GaussianProcessRegressor (Matérn, WhiteKernel, ConstantKernel, DotProduct), Partial Dependence, and evaluation metrics (`r2_score`, `mean_absolute_error`, `mean_squared_error`). |
| **SciPy** | Statistical distributions, optimisation, distance calculations, ranking functions, and Latin Hypercube Sampling (`norm`, `rankdata`, `minimize`, `cdist`, `scipy.stats.qmc.LatinHypercube`). |
| **scikit-optimize (skopt)** | Design of experiments using Latin Hypercube (`Lhs`) and search space definition (`Space`). |
| **SHAP** | Model explainability and feature importance analysis. |
| **PyTorch** | Monte Carlo Dropout neural network implementation (`torch.nn`) for uncertainty-aware surrogate modelling. |

[Back To The Top](#-repositories)


## 📢 AI use disclosure

In accordance with [Imperial College London's Generative AI guidance](https://www.imperial.ac.uk/admin-services/library/learning-support/generative-ai-guidance/), the following statements acknowledge the use of generative AI tools in this project.

I acknowledge the use of [Gemini](https://gemini.google.com) for conceptual clarification and architecture planning, [ChatGPT](https://chatgpt.com/) for abstract image generation, and [Claude](https://claude.ai) for code development, debugging, and drafting and reviewing project documentation. I confirm that no content generated by AI has been presented as my own work.

All AI-assisted code and text were treated as a starting point rather than a final output: outputs were reviewed, tested, and edited by me, technical claims and figures were verified against the actual optimisation results before inclusion, and the surrounding analysis, decision-making, and written narrative reflect my own reasoning rather than unedited AI output.


[Back To The Top](#-repositories)


## 📄 References and acknowledgement 
**References**
- Gebru, T., Morgenstern, J., Vecchione, B., Wortman Vaughan, J., Wallach, H., Daumé, H. and Crawford, K., 2021. Datasheets for datasets. Communications of the ACM, 64(12), pp.86–92.
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I.D. and Gebru, T., 2019. Model cards for model reporting. In Proceedings of the conference on fairness, accountability, and transparency (pp. 220-229).
- Frazier, P.I., 2018. A tutorial on Bayesian optimization. arXiv preprint arXiv:1807.02811.
- Rasmussen, C.E. and Williams, C.K.I., 2006. Gaussian processes for machine learning. Cambridge, MA: MIT Press.
- Shahriari, B., Swersky, K., Wang, Z., Adams, R.P. and de Freitas, N., 2016. Taking the human out of the loop: a review of Bayesian optimization. Proceedings of the IEEE, 104(1), pp.148–175.
- Srinivas, N., Krause, A., Kakade, S.M. and Seeger, M., 2010. Gaussian process optimization in the bandit setting: no regret and experimental design. In: Proceedings of the 27th International Conference on Machine Learning (ICML 2010). Haifa, Israel, 21–24 June 2010.

**Acknowledgement**
- A special thank you to Wolfram Wiesemann, Alex Ribeiro-Castro, Ruth Misener, and Christopher L. Tucci for their inspiring teaching approach, dedication, and excellence in delivering the Professional Certificate in Machine Learning and Artificial Intelligence at Imperial College London.
- Thank you to [@carolinebryant](https://github.com/carolinebryant), Intelligence Analyst [**Caroline Bryant**](https://www.carolinebryant.com/) for sharing inspiring references and resources that helped shape my understanding of Black-Box Optimisation (BBO).
- Thank you to my study mates, [**Chase Bender**](https://github.com/chasebender) and [**Eduardo**](https://github.com/Wizen-Labs), for their insightful discussions, encouragement, and support throughout the BBO Capstone Project for the Professional Certificate in Machine Learning and Artificial Intelligence at Imperial College London. It was a pleasure learning and tackling this challenge together!

[Back To The Top](#-repositories)
