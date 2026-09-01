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

# 🎤 BBO Capstone Project Presentation

👩‍🔬 Author: May Zune | Imperial College Business School 2026

🆚 Version: v1 (31 August 2026)

©️ Licence: MIT License

> 🏆 This **presentation** is submitted at Week 12, Module 23 for Stage 2: Required capstone component 23.2. It received Exemplary Assignment Badge.

> ⭐ **Exemplary Assignment Overview** At Imperial, we encourage participants to study, research, and apply their learning to support their professional development. In this cohort, we seek to identify participant assignments that exhibit this spirit. Participants who submit non-plagiarised assignments on time that indicate their in-depth understanding and application of key concepts along with relevant examples and correct usage of jargon and terminologies will be eligible for the Exemplary Assignment Badge. The award is a digital badge that can be displayed on social media, such as LinkedIn and other social channels. The best assignments will be identified post-grading.

## 1. An Overview of Your BBO Approach

### What I Am Trying to Achieve

Optimising an unknown function is like finding a mountain peak in zero visibility: no map, no view of the slope, just drop, measure, guess again. When each measurement costs real money or hours of compute, guessing isn't an option. That's **Black-box Optimisation (BBO)**.

Over 13 weeks, I worked across eight benchmark functions, from simple 2D surfaces to complex 8D spaces. The goal wasn't to brute-force the space; it was a sample-efficient strategy for finding global maximums with as few expensive calls as possible.

This mapped directly onto my background in architectural engineering. Window ratios, shading, insulation, thermal mass, HVAC controls, dozens of coupled variables, and a single EnergyPlus simulation can take hours. Brute-force sweeps burn compute, time, and budget fast. BBO gives a structured way through that trade-off.

When I worked on evaluation, it wasn't about chasing a leaderboard score. It came down to decision quality, hypothesis testing, clear governance, and how I handled uncertainty when the surrogate model misbehaved.

### My Strategy and Overall Process

An adaptive Bayesian Optimisation pipeline that learns from every sample and picks the next point deliberately.

1. **Data preprocessing:** Raw outputs were often chaotic: skewness, variance spikes. Standardising features and log-transforming targets kept the surrogate model (F7) from over-indexing on outliers.
2. **Surrogate modelling:** Gaussian Processes mapped the unknown terrain, giving both a predicted value and a confidence level at every point.
3. **Candidate selection & filtering:** Balancing exploitation and exploration meant scoring millions of candidates, which didn't scale. A two-stage filter—coarse sweep, then localised optimisation—fixed that (F5, F6, F8).
4. **Model updating:** Compared ground truth against GP predictions after each evaluation. Residual errors showed where the model misjudged variance or local trends and fed into the next round's hyperparameters.

### Bridging Abstract Optimisation to Real-World Engineering

Three places this now shows up in my building design work:

* **Faster thermal simulations:** Surrogate models approximate EnergyPlus performance from a fraction of the samples: thousands of design variations in minutes, not days.
* **High-dimensional retrofits:** Intuition works for two or three variables; it breaks down at six or eight. Bayesian optimisation finds non-obvious configurations across envelope, glazing, and mechanical controls that heuristics miss.
* **Managing uncertainty:** Real performance drifts from static predictions as climate and occupant behaviour change. Exploration plus targeted evaluation lets us stress-test designs early without over-conservative assumptions.

---

## 2. How My Strategy Has Evolved

### From Blind Guesswork to Precision Engineering

**Week 1:** Scatter guesses, run the same baseline model across all eight functions, and hope for the best.

It broke down quickly; some functions showed wild swings, others hit serious computational bottlenecks, and at times the model even directed the search towards solutions that were worse than those I had already found.

Thirteen weeks of fixes later, the workflow looks nothing like where it started.

### Tailoring the Strategy to Each Problem

Six fixes, one per failure mode:

* **F1 & F2 — Repetitive guesses:** Distance rules to force new territory.
* **F3 — Missing the big picture:** Trend-line model for the slope, local model for the detail.
* **F4 — Fixed pacing:** Explore broadly early, narrow later.
* **F5 & F8 — Slow evaluation:** Two-stage filter; cheap screen kills 90% of candidates before the main model runs.
* **F6 — Treating variables equally:** Model now learns which ones actually drive performance.
* **F7 — Skewed data:** Stabilise the data and restrict search to high-impact ranges.

### Five Core Rules I Live by Today

1. Explore wide, then narrow as patterns emerge.
2. Focus on key drivers; ignore the noise.
3. Never test the same spot twice.
4. Filter early, evaluate late.
5. Stabilise data before modelling.

---

## 3. Patterns, Data and Insights

### Five Core Lessons from the Optimisation Landscape

Treating this as a living system, not just a dataset, changed how I approach it.

1. **Smart search beats blind search.** Uniform random sampling wasted budget on hard landscapes (F5, F7, F8). Switching to local refinement around the best-known points did better.
2. **Manage noisy data before it misleads you.** Spikes in F7 fooled the model into chasing false uncertainty. A `log1p` transform fixed it; the search stopped chasing noise.
3. **Separate trend from detail.** F3 has a global slope plus local bumps, too much for one model. A polynomial curve handles the slope; the Gaussian Process handles the bumps.
4. **Not all dimensions matter.** F6 and F8 are driven by a handful of variables. ARD lets the model stretch/shrink dimensions by actual influence, instead of wasting queries everywhere.
5. **Screen before you commit compute.** A lightweight SVR "bouncer" filters weak candidates before the expensive model runs, with full evaluation only for serious contenders.

### The Blueprint Behind the Improvements

Three categories drove the gains:

#### Architecture

* Scaling fixes on F7
* Trend/residual separation on F3
* ARD on F6/F8

Each cut wasted search effort.

#### Candidate Quality & Filtering

* Localised search around top performers gave the biggest convergence gains.
* Pre-screening (F5, F8) plus SHAP-defined boundaries kept the search focused.

#### Explore/Exploit Balance

* Wide early, narrow later (F4).
* Spatial distance constraints on F1/F2 stopped candidates from clustering too close together.

### A Mindset Shift

Four principles now guide how I approach optimisation:

1. Candidate quality beats model complexity.
2. Clean variance (via transforms, trend separation) beats fighting noisy data downstream.
3. A small set of dimensions usually drives most of the outcome; find them early.
4. Multi-stage filtering lets you explore widely without paying the full cost everywhere.

---

## 4. Decision-Making and Iteration

### Navigating the Unknown: A Dynamic Optimisation Strategy

Every search over unknown territory faces the same trade-off: explore new ground or exploit what you've already found.

My framework handles this the way you'd explore a new city: wander broadly on day one, then spend more time in the neighbourhoods that turn out to matter.

Early on, the algorithm samples high-uncertainty regions. As data builds up, it shifts compute toward refining the strongest candidates.

### Day-to-Day Tactics

* **Variable candidate generation:** Spread wide for global exploration; cluster tightly around top performers for local refinement.
* **Spatial distance constraints:** A minimum distance between evaluated points stops the search from getting stuck re-testing familiar ground.
* **Sensitivity-based dimensional reduction:** Not every variable matters equally. Sensitivity analysis finds the key drivers and pins the rest to baseline.
* **Pre-evaluation filtering:** A cheap screening model cuts the candidate pool before expensive evaluation runs.

### Key Strategic Experiments

#### 1. Macro Trends + Local Detail

One model for global trendlines, one for local contours, like mapping a mountain range:

* The macro model tells you, *"We're going uphill."*
* The local model finds the exact peak.

Splitting the two cut baseline error and speeds up convergence.

**Trade-off:** A wrong macro model biases the whole search. It needs flexible starting assumptions.

#### 2. Two-Stage Filtering

A cheap filter screens thousands of candidates before the expensive engine sees them.

**Trade-off:** False negatives. A bad early screen can kill a good region before the precise model ever gets a look. Filter sensitivity needs ongoing calibration.

### Adaptive Feedback Loops

Three ways the framework recalibrates when things don't go to plan:

* **Model recalibration:** Adjust noise parameters or rescale targets when prediction variance shifts.
* **Perturbation & constraint loosening:** Loosen spatial constraints and widen the sampling radius when progress stalls.
* **Dynamic feature selection:** Drop uninformative parameters and redirect compute when high-dimensional performance plateaus.

---

## 5. Next Steps and Reflection

### Future Roadmap: Multi-Round Optimisation

For future rounds, I will focus less on tuning the algorithm and more about making it faster, sharper, and scalable.

### 1. Adaptive Modelling & Ensembling

Swap the fixed surrogate model for one that adapts to the search phase:

* Broad exploration early.
* Precise local estimates as it converges.
* Ensembling GPs, Random Forests, and Gradient Boosted Trees cuts blind spots and improves uncertainty estimates.

### 2. Batch Testing & Filtering

* Evaluate multiple candidates in parallel.
* Expand search regions when results are consistently strong.
* Tighten regions around challenging or uncertain areas.
* Use a lightweight surrogate to screen candidates first, reserving expensive computation for the most promising contenders.

### 3. Feature Weighting & Preprocessing

* Drop or deprioritise parameters that stop contributing.
* Detect skewed/heavy-tailed data automatically.
* Transform data before fitting to maintain model stability.

### Bridging ML and Built Environment Science

The same upgrades can be applied to building performance, decarbonisation, and indoor air quality:

* **High-dimensional optimisation:** EnergyPlus and CFD simulations are too expensive to brute-force. Adaptive acquisition finds Pareto-optimal designs—energy, comfort, air quality—on a fraction of the simulation budget.
* **Hybrid physical-ML models:** Physics handles the knowns (thermodynamics, geometry, climate zone). GPs handle the messy stuff (occupant behaviour, microclimate).
* **Geospatial active learning:** Spatial diversity sampling picks representative building archetypes instead of redundant ones, for city-scale insight without city-scale simulation.
* **Data robustness:** Automated transforms keep models stable through energy spikes, weather extremes, and shifting climate conditions.
* **Scalable screening:** Two-stage filtering lets policymakers scan millions of retrofit options fast, saving CFD/thermal modelling for the finalists.

### Translating ML to Real-World Impact

For planners, facility managers, and investors: less about the algorithm, more about what it saves them—**energy costs, retrofit risk, and compliance checks.**




