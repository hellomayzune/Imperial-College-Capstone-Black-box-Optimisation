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

# 📊 Datasheet | SSOS

👩‍🔬 Author: May Zune | Imperial College Business School 2026

🆚 Version: v1 (31 August 2026)

©️ Licence: MIT License

#### 🧭 Content

This **datasheet** contains the following contents.
- [Motivation](#-motivation)
- [Composition](#-composition)
    - [Format and size](#format-and-size)
    - [Label and target](#label-and-target)
    - [Weekly data growth](#weekly-data-growth)
    - [Error, noise and missing data](#error-noise-and-missing-data)
- [Collection process](#-collection-process)
    - [Example of function-specific methods (Week 10)](#example-of-function-specific-methods-week-10)
    - [Final collection round (Week 13)](#final-collection-round-week-13)
    - [Common design choices](#common-design-choices)
    - [Post-hoc validation or comparison of strategies](#post-hoc-validation-or-comparison-of-strategies)
- [Preprocessing](#️-preprocessing)
    - [Output transforms (Week 10 example)](#output-transforms-week-10-example)
    - [Feature engineering](#feature-engineering)
    - [Errors, duplicates, and degenerate points](#errors-duplicates-and-degenerate-points)
    - [Raw data and software](#raw-data-and-software)
- [Extended analysis of weekly optimisation trajectories and output performance](#️-extended-analysis-of-weekly-optimisation-trajectories-and-output-performance)
- [Uses](#️-uses)
    - [Intended use](#intended-use)
    - [Inappropriate uses](#inappropriate-uses)
- [Distribution and maintenance](#-distribution-and-maintenance)
    - [Distribution](#distribution)
    - [Maintenance](#maintenance)

>*🔗 This datasheet is structured according to Gebru, T., Morgenstern, J., Vecchione, B., Wortman Vaughan, J., Wallach, H., Daumé, H. and Crawford, K., 2021. Datasheets for datasets. Communications of the ACM, 64(12), pp.86–92.*

> 📌 This **datasheet**  is intended to complement the accompanying [week-summary](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation), [model-card](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation) and  [architecture-decision-record](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/documentation). Together, these documents describe the project's development, key design decisions, model evolution, and evaluation.


## ✨ Motivation

Black-box optimisation addresses problems where the objective function is unknown, costly to evaluate, or otherwise inaccessible, limiting the use of conventional optimisation methods. This datasheet supports evaluation and comparison of algorithms for such problems, enabling users to assess how effectively different methods locate optimal solutions under these constraints.

The datasheet was produced for the Stage 2 Capstone submission to Imperial College London's Computing Department and Emeritus Business School, as part of the Professional Certificate in Machine Learning and Artificial Intelligence. It received no external funding and reflects the student's personal interest in machine learning.

The dataset supports black-box function optimisation, where the goal is to find a function's maximum using few evaluations. It enables testing of search strategies such as Bayesian optimisation, using benchmark problems inspired by real-world applications including radiation detection, robot control, and drug discovery.

[Back To The Top](#-content)


## 🌱 Composition

The dataset comprises eight synthetic functions, each simulating a distinct real-world optimisation scenario. Every function is supplied with a small initial set of query–response pairs (initial_inputs.npy, initial_outputs.npy), reflecting the low-data setting typical of expensive black-box evaluations. Each instance is an input vector (2–8 dimensions, depending on the function) and a corresponding scalar output. The table below summarises each function.

| Function | Input | Output | Goal | Application |
|---|---|---|---|---|
| F1 | 2D | 1D | Maximise | Radiation-source detection: locate contamination where only proximity gives a non-zero reading. |
| F2 | 2D | 1D | Maximise | Noisy log-likelihood surface with multiple local optima; tests exploration vs. exploitation. |
| F3 | 3D | 1D | Maximise | Drug discovery: minimise adverse reactions across three compounds (negated for maximisation). |
| F4 | 4D | 1D | Maximise | Warehouse placement: ML surrogate approximates an expensive biweekly calculation over four hyperparameters. |
| F5 | 4D | 1D | Maximise | Chemical process yield: typically unimodal, single optimum. |
| F6 | 5D | 1D | Maximise | Recipe optimisation across five ingredients (negative score reframed as maximisation). |
| F7 | 6D | 1D | Maximise | ML hyperparameter tuning (e.g. learning rate, regularisation). |
| F8 | 8D | 1D | Maximise | High-dimensional (8D) hyperparameter tuning; global optimum is hard to find, so strong local maxima are an accepted goal. |

### Format and size

Data is stored as NumPy binary files (.npy), loaded via `np.load()`. Each function has an inputs array (n × d) and a matching 1D outputs array of length n. Arrays are deliberately small, reflecting evaluation cost, and grew by one row per function each week across the full W3–W13 collection window (11 weekly rounds), on top of the initial samples and the W2 handpicked round.

### Label and target

The scalar output is the target to be maximised. Where the real-world objective is naturally a minimisation problem (e.g. adverse reactions in F3, negative recipe score in F6), the output is transformed, typically by negation, so higher is always better.

### Weekly data growth

Each function begins with a small initial sample; new query–response pairs are appended incrementally each round (labelled  W3, W4, ..., W13) using `np.vstack/np.hstack`, with the running best tracked via np.argmax. Growth continued for the full 13-week Capstone period: by the close of Week 13, each function's dataset held its initial samples plus 12 additional weekly points (W2–W13). 

### Error, noise and missing data

Some functions are explicitly noisy (e.g. F2, which can produce misleading local optima), while others (e.g. F5) are smoother and largely unimodal. Week 1 queries were excluded in this work. A separate data-quality issue surfaced mid-project rather than at collection time: F7's Week 10 candidate-filtering pipeline was audited in Week 11 and found to have proposed candidates whose predicted value fell below the existing record, meaning at least one week's search procedure (though not the appended data point itself) was flawed until corrected — see [Post-hoc validation](#post-hoc-validation-or-comparison-of-strategies).

[Back To The Top](#-content)

## 🎯 Collection process

Queries were generated sequentially, one round per week, over a planned 13-week period, which ran to completion. Each function follows the same loop each week: load the cumulative dataset; fit a surrogate model as a probabilistic proxy for the true function; generate a candidate pool (random uniform, Latin Hypercube, or perturbations around strong points); score candidates with an acquisition function trading predicted value against uncertainty; and select, submit, and append the top-scoring candidate. This loop held through Week 12; Week 13 replaced the per-function acquisition step with a single shared exploitation-only procedure (below).

### Example of function-specific methods (Week 10)

The surrogate model and acquisition function differ by function and evolved across weeks as evidence accumulated. By Week 10, choices had converged toward each function's demonstrated response surface:

| Function | Surrogate model | Acquisition strategy |
|---|---|---|
| F1 | GP (Matérn + White kernel) on arcsinh-transformed outputs | UCB (κ = 1.5) |
| F2 | GP (Matérn + White kernel) | Posterior-mean maximisation (multi-start L-BFGS-B) |
| F3 | ExtraTrees on log-transformed, polynomially-expanded features | Lower confidence bound (κ = 0.3) |
| F4 | GP (fixed length-scale), candidate pool biased to recent promising areas | Posterior-mean maximisation |
| F5 | SVR pre-filter → GP on log1p outputs | Highest predicted GP mean |
| F6 | Two-stage GP: global LHS, then local refinement | UCB (κ = 2.5) |
| F7 | Two-stage Random Forest filter (100k→10k→1k) → GP | Expected improvement (original scale) |
| F8 | Monte Carlo-Dropout MLP surrogate | UCB, adaptive κ via weekly leave-one-out CV |

This snapshot was not the final word for every function. Week 11 reopened F1 (reintroducing Expected improvement) and, more substantively, audited and replaced F7's two-stage Random Forest filter after it was found to exclude the true optimum from its candidate pool. Week 12 replaced F3's ExtraTrees-on-polynomial-features surrogate (used continuously since Week 3) with a locally-fitted bootstrap ensemble of Ridge regressors restricted to a 0.15-radius basin around the incumbent best point, and F5 moved to fully exploitative selection once its GP and SVR models agreed on a monotonic trend toward the domain corner (1,1,1,1).

[Back To The Top](#-content)

### Final collection round (Week 13)

Week 13 was the last data-collection round and used a single, shared procedure across all eight functions rather than function-specific acquisition tuning: 

1. combine all prior weeks' observations and identify the current best point; 

2. restrict attention to its local neighbourhood, discarding negative/failure values and noise-floor points where relevant; 

3. fit a weighted local-linear model to estimate the ascent direction; 

4. fit a regularised local-quadratic model and inspect its Hessian to check the region is consistent with a local maximum; 

5. fit a local GP on the nearby signal-bearing observations for a final predicted output and uncertainty; 

6. take a conservative step in the estimated ascent direction, staying inside the region the GP's uncertainty still supports; 

7. report the GP's predicted output and interval alongside the query; 

8. rank/sanity-check using the `signed_score` transform where outputs span extreme positive/negative ranges. 

No exploration bonus (UCB/EI margin) was applied in this final round for any function, as its aimas was a pure-exploitation.

### Common design choices

Across Weeks 3–12, three patterns recur. 

First, the surrogate model class shifted over successive weeks (GP, Random Forest, ExtraTrees, SVR/bootstrap ensembles, or MC-Dropout neural network); most functions settled by Week 7–9, though F3 changed surrogate again at Week 12 and F7's Week 10 pipeline was substantively revised in Week 11. 

Second, acquisition was primarily UCB or Expected Improvement, though several functions used pure posterior-mean maximisation — an explicitly exploitative choice with no uncertainty term — increasingly from Week 10 onward, culminating in Week 13's uniform pure-exploitation round. 

Third, several functions applied output transforms (`log`, `log1p`, `arcsinh`, or `square-root`) to stabilise the surrogate fit; F3's transform was retired at Week 12 in favour of an untransformed, basin-local model. Candidate pools ranged from 5,000 to 100,000 points (Weeks 3–12) or ~21,000 locally-generated points (Week 13, F5) and were typically filtered to exclude points too close to previously queried locations.

### Post-hoc validation or comparison of strategies

Strategy choices were not always fixed before a query was generated. 

- For F1 and F5, both an exploitation-leaning and an exploration-leaning candidate were computed and compared before submission in several weeks. 
- For F3, the `transform log(−Y)` was applied to select the live query before its validity was cross-checked against alternatives, and this transform remained unvalidated in production use from Week 3 through Week 11. 
- A further, more consequential instance of post-hoc validation occurred in Week 11: F7's Week 10 two-stage Random Forest candidate filter (100k→10k→1k) was reviewed retrospectively and found to propose candidates whose predicted output fell below the existing record — i.e. the filtering stage had been actively excluding the region containing the true optimum. It was replaced with a continuous trust-region, multi-start L-BFGS-B optimiser from Week 11 onward. 
- Query selection therefore sometimes involved iterative, post-hoc comparison — and, in F7's case, a post-hoc correction of a previously-submitted week's methodology — rather than a fixed a priori rule; this is a relevant detail for reproducibility, and means the Week 10 F7 row in this datasheet should be read as a documented historical decision rather than a validated best practice.

[Back To The Top](#-content)

## 🛠️ Preprocessing

Raw query–output pairs are stored unmodified in the .npy files. Before each surrogate model is fitted, function-specific transforms are applied to outputs and, in some cases, engineered features are added to inputs. No manual labelling was required, since every instance already carries a numeric target returned directly by the black-box function.

### Output transforms (Week 10 example)

| Function | Transform | Purpose |
|---|---|---|
| F1 | Sign-preserving arcsinh; mask \|y\| < 1e-30 (W6) | Compresses magnitude range while preserving zero-crossings for GP fitting |
| F3 | log(−Y) + degree-2 polynomial interaction features | Better-behaved scale for tree models; NOTE — breaks if Y ≥ 0, validated only after use |
| F5 | log1p(Y); StandardScaler on inputs/outputs | Stabilises GP fit on skewed yields; required by SVR's distance kernel |
| F7 | √Y | Variance-stabilising; EI computed back on original Y scale |
| F8 | StandardScaler on inputs/outputs | Required for stable MC-Dropout MLP training |

This table reflects the Week 10 state and is retained as a historical snapshot. The one material change afterward: F3's log(−Y) + polynomial-feature pipeline was dropped at Week 12 in favour of an untransformed, basin-local Ridge ensemble (no output transform, no polynomial expansion), once the unvalidated log(−Y) transform was judged an unnecessary source of risk relative to a simpler local model. F1, F5, F7, and F8's transforms above remained in use, largely unchanged, through Week 13.

[Back To The Top](#-content)

### Feature engineering

- For F3, the three raw inputs were expanded into degree-2 polynomial interaction features before being passed to the ExtraTrees surrogate (Weeks 3–11), letting a tree-based model capture interaction effects; this expansion was dropped at Week 12 along with the rest of that pipeline. 
- For F5 and F8, inputs (and outputs, where applicable) were standardised before being passed to the SVR and MC-Dropout MLP surrogates, since these models are sensitive to input scale; scaling was inverted before reporting predictions. This remained the case through Week 13.

### Errors, duplicates, and degenerate points

Handling was partial and inconsistent across functions, and remained so through to the end of the project. 
- F1's Week 6 pipeline masked near-zero readings (|y| < 1e-30) before modelling. Several functions filtered new candidates too close to an already-queried location, but this did not apply retrospectively: 
- F3's log records show two weekly queries (W5, W6) are exact duplicates that were never removed from the dataset, even after F3's surrogate changed at Week 12. 
- F7's Week 10 candidate-filtering stage is a documented case of a systematic (not random) collection error: it was found in Week 11 to structurally exclude the region containing the true optimum, rather than simply adding noise. As with the F3 duplicates, the Week 10 query itself was not retroactively removed or re-collected — only the going-forward methodology was corrected. It is acknowledged that the historical W10 data should be aware of function's search direction history, for that one week, based on a flawed filter.

### Raw data and software

Transforms are applied only in memory at fitting time; the .npy files retain original, untransformed outputs throughout, and any transform is inverted before reporting or selecting a query. All transform, scaling, and filtering logic is implemented in the weekly Jupyter notebooks using NumPy, scikit-learn, SciPy, and (for F8) PyTorch, making every step inspectable and reproducible.

[Back To The Top](#-content)

## 🛠️ Extended analysis of weekly optimisation trajectories and output performance

> 📌 This **extended-data-weekly-outputs-pca-week27.ipynb**  [notebook](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/codes) presents an extended analysis of weekly optimization trajectories and output performance across eight benchmark functions (`F1`–`F8`) from the BBO capstone project. 

Using clustering and PCA-based dimensionality reduction, the analysis addresses two primary questions:

1. Whether the optimization algorithm explores new decision-space regions over time.
2. Whether specific input regions correlate with superior output performance.

### 1. Output Distribution (SymLog Scale)

Baseline (`Given`) inputs show wide ranges across all functions, with `F1`–`F4` tightly clustered around lower baseline ranges and `F6`–`F8` displaying higher output variance. Iterative `Weekly Updates` systematically shift output distributions upward, confirming that the optimizer successfully navigates away from initial random samples toward higher-performing output configurations.

### 2. Temporal Performance Trajectories (Weeks 2–13)

* **Low-Dimensional (`F1`–`F4`):** Exhibit rapid gains in early iterations (Weeks 2–5) before leveling off asymptotically, indicating quick convergence toward local/global optima.
* **High-Dimensional (`F7`, `F8`):** Display non-linear, step-like improvements requiring extended weekly iterations to escape sub-optimal plateaus.

### 3. PCA Projection & Decision-Space Exploration (`F8`)

Projecting the 8D input space of `F8` onto its first two principal components (`PC1` and `PC2`) reveals that weekly update points (W2–W13) progressively migrate away from the baseline input cloud, providing evidence of active exploration of previously under-sampled, high-performing decision regions.

### 4. Exploration Distance Metrics

Nearest-neighbor Euclidean distance measurements show that higher-dimensional functions (`F7`, `F8`) require significantly larger exploration distances from baseline points than lower-dimensional functions (`F1`, `F2`), supporting the conclusion that the algorithm is capable of searching well beyond the initial design regions.

[Back To The Top](#-content)

## ▶️ Uses

### Intended use

The dataset supports evaluation and comparison of black-box optimisation algorithms for the Capstone Project Stage 2 submission described above. Its primary use is benchmarking sequential decision-making strategies (Bayesian optimisation, tree-ensemble and neural-network surrogates) against eight synthetic functions of varying dimensionality, under a limited, expensive-to-evaluate query budget. Beyond the immediate assessment audience, the dataset and notebooks also suit teaching or self-study in optimisation under uncertainty, including as a worked example of how a search methodology can be audited and corrected mid-project (see F7, Week 11).

### Inappropriate uses

Although each function is framed around a real-world scenario, all eight are synthetic benchmarks, not measurements from a deployed system, and should not inform decisions about any real radiation-detection, drug-trial, warehouse, chemical, or production ML system. The dataset carries limited direct risk of social harm, as it involves no human subjects or protected attributes. Several methodological risks apply if results are generalised: some functions used exploitative, near-greedy acquisition (e.g. F2, F4 in Weeks 8–10, and every function in the pure-exploitation Week 13 round) that can converge prematurely to a local rather than global optimum; F3's unvalidated log(−Y) transform (in use Weeks 3–11) and uncleaned duplicate queries could propagate error if reused as ground truth elsewhere without accounting for these issues; and F7's Week 10 data point was selected using a candidate filter later shown to exclude the true optimum, so that week's result specifically should not be treated as a validated near-optimal query.

[Back To The Top](#-content)

## 🌐 Distribution and maintenance

### Distribution

Initial data points are hosted by the course provider (Imperial College London / Emeritus) as .npy files for enrolled students; weekly query results are submitted back through the course platform, with no plan for independent redistribution. No DOI, registry entry, or public repository listing exists, as this is a course artefact rather than a published dataset. No explicit licence terms are stated in the materials reviewed — an identified gap. No fees apply within the course context, and the dataset contains no personal, sensitive, or export-controlled content.

### Maintenance

The growing set of weekly queries was maintained by the student, **May Zune**, as sole author of the weekly notebooks; the course provider maintained the original initial data and submission platform. Updates followed a fixed weekly schedule for the 13-week Capstone period and concluded on schedule: this document reflects the completed state as of Week 13, the final collection round, and no further scheduled updates are planned. Known issues (e.g. F3's duplicate queries, F3's since-retired unvalidated transform, and F7's Week 10 filtering error) are documented in-line rather than corrected retroactively in the historical record, to preserve reproducibility of what was actually submitted each week. Anyone reusing the dataset after course completion should treat it as a static, closed snapshot and verify current availability of the original hosted files.

[Back To The Top](#-content)