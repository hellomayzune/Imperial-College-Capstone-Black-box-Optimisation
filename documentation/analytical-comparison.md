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

# 🪞 Reflection: Analytical Comparison (Caroline vs May)

👩‍🔬 Author: May Zune | Imperial College Business School 2026

🆚 Version: v1 (31 August 2026)

©️ Licence: MIT License

> 📌 This [analytical-comparison-caroline-may.ipynb](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/tree/main/codes) serves as an analytical comparison between **Caroline's BBO notebooks** and **May's BBO notebooks** across **Weeks 3 through 13**, which underpins the [Analytical Comparison section](https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/README.md#️-analytical-comparison-caroline-vs-may) of the SSOS README.

## Why I ran this comparison

My BBO codebase started from Caroline’s work, so as the project developed, I wanted to see how much my implementation had actually changed. Rather than relying on my own impression that “it looks different” I wanted a clear, evidence-based way to track those changes over time.

I therefore compared our notebooks week by week, from Week 3 to Week 13, to see where they still overlapped, where they had diverged, and how those differences developed.

I used `nbformat` to parse both sets of `.ipynb` files into code and Markdown cells. I then used `difflib.SequenceMatcher` and `thefuzz` to measure both structural similarity and line-by-line similarity for each week.

## Why I chose her work as the starting point for my codebase

I chose Caroline’s work as the starting point for my codebase because it gave me a solid foundation to build on while still leaving room for me to develop my own approach.

* **Strong BBO foundation:** Her approach used a standardised Gaussian Process (GP) with Matérn and White kernels, Expected Improvement (EI) evaluated on candidate pools, and distance-maximisation filters to maintain spatial separation between points.

* **Fast experimentation:** Her weekly notebooks showed that she was willing to try different ideas quickly and test several approaches rather than staying with one method for too long.

* **Clear documentation:** Her changelogs were short but useful. The `# changes:` comment above each function made it easy to see what had been changed without adding unnecessary detail.

* **Comfort with uncertainty:** She was willing to make practical decisions when the results were close. For example, she documented a near-tie between 0.741 and 0.727 and moved forward rather than spending too much time trying to force a clear winner. I found this useful because it reflects the kind of decisions that often have to be made when working under time constraints.

* **Consistent validation:** Most of her changes were followed by some form of validation, such as a leave-one-out (LOO) or training diagnostic comparison. This gave me confidence that the changes were being assessed using results rather than intuition alone.

* **Willingness to change direction:** She was also prepared to revert changes and try different acquisition-sampling strategies. This gave me several alternative ideas for handling spatial distance constraints and helped me think about which parts of the approach I wanted to keep or develop differently.


## What the numbers showed

| Phase       | Code similarity | Markdown similarity | Reading                                                                            |
| ----------- | --------------: | ------------------: | ---------------------------------------------------------------------------------- |
| Weeks 3–7   |         51%–63% |              3%–12% | Shared foundations, including GP regressors, EI, and query formatting              |
| Weeks 8–10  |         61%–68% |                 ~4% | Highest overlap, mainly from shared acquisition functions and surrogate routines   |
| Weeks 11–13 |       58% → 38% |               5%–8% | Increasing divergence as I developed my own fidelity metrics and modular structure |

Two things stood out from the results:

* **Code similarity was highest in the middle of the project, then dropped.** It reached 61%–68% in Weeks 8–10, fell to 58% in Week 11, and reached 38% by Week 13. This reflects the point where I moved away from adapting shared components and started developing my own fidelity metrics, element-wise metrics, and modular function files.

* **Markdown similarity stayed low throughout.** It never went above 12%, even during the weeks when the code was most similar. This was important because it showed that, although we were working with some of the same core ideas — such as Gaussian Processes, EI/UCB acquisition functions, and Matérn kernels — the way I organised, described, and explained the work was different.

## What are found in structural and analytical differences

Across Weeks 3–12, my approach gradually became more structured and validation-focused, while Caroline’s remained more centred on rapid experimentation and targeted code-level validation. My work moved towards a structured, self-auditing and highly documented process, while Caroline’s remained focused on rapid experimentation, targeted validation, and concise code-level diagnostics.

* **Weeks 3–6:** I made few direct changes but explored a flexible modelling approach using multiple models, SHAP feature selection, domain knowledge, and hybrid candidate generation. This offered adaptability but introduced risks such as overfitting and overly restrictive search spaces. Caroline used a more standardised GP-based approach with EI and distance criteria, providing consistent uncertainty modelling and spatial coverage but less adaptation to different response profiles.

* **Weeks 7–9:** Our approaches began to differ more in how we worked. I focused on validation, auditability, and detailed documentation, using measures such as LOO R², bias, and calibration checks. Caroline focused on faster experimentation, testing more modelling options and keeping her notebooks concise. 

* **Weeks 10–12:** The difference became more structural. I introduced formal planning tables, documented assumptions and query changes, and added end-of-notebook fidelity reports using MAE, RMSE, and R². Caroline continued to use concise code comments with before-and-after diagnostics and validated changes individually, including reverting changes when they did not improve results.

The overlap in the middle of the project did not concern me. It made sense because we were working on the same problem, using the same standard methods, such as GPs, EI, UCB, and Latin Hypercube Sampling. A 50–68% code similarity across Weeks 3–10 is therefore not surprising when the underlying methods and tasks are shared. The fact that the similarity decreased as I added these changes was useful evidence. The divergence was not something that appeared randomly; it increased as I introduced more of my own extensions and design choices. This gave me more confidence that the later differences reflected the way I had developed the project myself.

## What I’d do differently

* **Separate code from comments and variable names.** At the moment, similarity is measured across complete code cells. Breaking this down further would make it easier to see how much of the similarity comes from the actual algorithm and how much comes from things like variable names or comments.

* **Add a trend chart.** The week-by-week tables show the changes, but plotting similarity over time would make the drop from Week 10 to Week 13 much easier to see at a glance.


##  References and appreciation 
- A special thank you to [@carolinebryant](https://github.com/carolinebryant), Intelligence Analyst [Caroline Bryant](https://www.carolinebryant.com/). for sharing inspiring references and resources that helped shape my understanding of black-box optimisation.

