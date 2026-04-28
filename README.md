# Biscuit Dunking — Physics-Informed Machine Learning for Pore Characterisation

> **Identifying biscuit varieties and predicting capillary rise from dunk mechanics using the Washburn equation, Bayesian inference, and supervised classification.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557C?style=flat&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-4C72B0?style=flat&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11%2B-8CAAE6?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Scientific Background](#scientific-background)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Methods](#methods)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Limitations and Open Questions](#limitations-and-open-questions)
- [Dependencies](#dependencies)

---

## Overview

When a biscuit is dunked in tea, liquid rises through its porous crumb structure via capillary action. The rate and extent of this rise is governed by the **Washburn equation**, which relates the penetration depth $L$ to time $t$ through the effective pore radius $r$, liquid surface tension $\gamma$, contact angle $\phi$, and viscosity $\eta$:

$$L(t) = \sqrt{\frac{\gamma \, r \cos\phi}{2\eta} \cdot t}$$

This project asks three questions:

1. **Does the Washburn equation actually hold** across a large dataset of real dunks?
2. **Can we identify which biscuit** was dunked from its physical measurements alone?
3. **Given an unknown time-resolved dunk**, which biscuit is it and what is its pore radius?

The answer combines analytical physics, Bayesian parameter estimation, supervised classification, and a hybrid pipeline that propagates uncertainty from classifier probabilities through to a final $L$ prediction with physically motivated error bars.

---

## Scientific Background

The three biscuits studied — **Rich Tea**, **Hobnob**, and **Digestive** — differ in composition and manufacturing process, which produces distinct internal pore structures. The key physical quantity separating them is the effective pore radius $r_\text{eff}$, inferred from each dunk via:

$$r_\text{eff} = \frac{2\eta L^2}{\gamma \, t \cos\phi}$$

This derived feature carries far more discriminative power than any raw measurement ($\gamma$, $\phi$, $\eta$, $L$, $t$) individually, which the classification experiments confirm directly.

Bayesian inference on the three time-resolved files uses a grid-evaluated posterior with a flat prior over a physically plausible range, and reports the MAP estimate alongside a 95% highest density interval (HDI). The hybrid prediction pipeline uses the law of total variance to decompose prediction uncertainty into two physically meaningful sources: within-biscuit pore variability and classifier uncertainty.

---

## Repository Structure

```
.
├── data/
│   ├── dunkingdata.csv          # 3000 labelled dunks (Rich Tea / Hobnob / Digestive)
│   ├── microscopydata.csv       # 500 unlabelled microscopy measurements
│   ├── tr1.csv                  # time-resolved dunk file 1 (unknown biscuit)
│   ├── tr2.csv                  # time-resolved dunk file 2 (unknown biscuit)
│   └── tr3.csv                  # time-resolved dunk file 3 (unknown biscuit)
├── analysis_notebook.ipynb     # main analysis notebook
├── requirements.txt
└── README.md
```

---

## Datasets

### `dunkingdata.csv`
3000 labelled dunk measurements, 1000 per biscuit type.

| Column  | Units | Description                        |
|---------|-------|------------------------------------|
| `gamma` | N/m   | liquid surface tension             |
| `phi`   | rad   | contact angle                      |
| `eta`   | Pa·s  | dynamic viscosity                  |
| `L`     | m     | capillary penetration depth        |
| `t`     | s     | dunk duration                      |
| `biscuit` | —   | `Rich Tea`, `Hobnob`, or `Digestive` |

### `microscopydata.csv`
500 unlabelled samples with directly measured pore radii alongside Washburn input variables. Used to validate that the Washburn-inferred $r_\text{eff}$ matches ground-truth microscopy.

### `tr1.csv`, `tr2.csv`, `tr3.csv`
Time-resolved measurements of a single unknown biscuit, giving $(t, L, \delta L)$ triplets. Constants $\gamma = 6.78 \times 10^{-2}$ N/m, $\phi = 1.45$ rad, $\eta = 9.93 \times 10^{-4}$ Pa·s apply to all three files.

---

## Methods

### Tier 1 — Washburn Validation and Classification

**Feature engineering.** The effective pore radius is computed per dunk and appended as a feature. Raw inputs alone separate the biscuits poorly; adding $r_\text{eff}$ pushes classification accuracy from ~75% to ~97%.

**Classifiers.** Logistic regression and random forest are compared with and without $r_\text{eff}$, evaluated by 5-fold stratified cross-validation and a held-out test set.

**Regression.** A Washburn log-linear baseline (fitting $\log L$ vs $\log(\gamma t \cos\phi / 2\eta)$) is compared against a tuned gradient boosting regressor. The physics baseline is surprisingly competitive.

### Tier 2 — Statistical Testing

One-way ANOVA and pairwise Welch $t$-tests (Bonferroni corrected) confirm that $r_\text{eff}$ differs significantly between all three biscuit types. Per-biscuit log-linear fits recover Washburn slopes close to the theoretical 0.5, validating the physical model.

### Tier 3 — Bayesian Inference and Hybrid Pipeline

**Bayesian pore radius estimation.** A 1D posterior $p(r \mid \text{data})$ is evaluated on a fine grid for each TR file using a Gaussian likelihood with the reported $\delta L$ uncertainties. The MAP estimate, posterior mean, and 95% HDI are reported. Results agree closely with the frequentist bootstrap fits.

**Hybrid $L$ prediction.** Given a new measurement, the classifier produces per-class probabilities $\{P_k\}$. These weight per-class $L$ predictions and variances (derived via the delta method from per-biscuit $r$ statistics), giving a mixture mean and variance via the law of total variance:

$$\mathbb{E}[L] = \sum_k P_k \, \mu_{L,k}, \qquad \text{Var}[L] = \sum_k P_k \left(\sigma_{L,k}^2 + \mu_{L,k}^2\right) - \mathbb{E}[L]^2$$

This produces calibrated prediction intervals that account for both biscuit identity uncertainty and within-biscuit pore variability.

### Gap Analyses

Three follow-up analyses address specific unresolved questions raised by the main results:

- **Gap 1.** tr3 has $\chi^2_\text{red} \approx 5.8$, indicating its reported $\delta L$ values are inconsistently tight. The fit is repeated with $\delta L$ rescaled by $\sqrt{\chi^2_\text{red}}$ to produce a self-consistent error model.
- **Gap 2.** Hobnob's $r_\text{eff}$ distribution is ~60% wider than the other biscuits (CV ≈ 16% vs 10% and 7%). Error propagation from the Washburn inputs quantifies how much of this spread is genuine biscuit-to-biscuit variability versus propagated measurement noise.
- **Gap 3.** The hybrid pipeline's residual error ($R^2 = 0.985$) is decomposed into its two physical sources — within-biscuit pore variability and classifier uncertainty — to identify which is the limiting factor.

---

## Results

| Model | CV accuracy | Test accuracy |
|---|---|---|
| Logistic regression, raw features | 0.743 ± 0.012 | 0.748 |
| Logistic regression, + $r_\text{eff}$ | 0.963 ± 0.008 | 0.961 |
| Random forest, raw features | 0.782 ± 0.015 | 0.779 |
| **Random forest, + $r_\text{eff}$** | **0.971 ± 0.006** | **0.968** |

| Regression model | $R^2$ | RMSE |
|---|---|---|
| Washburn log-linear | 0.9801 | 0.214 mm |
| Gradient boosting (tuned) | 0.9887 | 0.163 mm |
| Hybrid pipeline | 0.9850 | 0.187 mm |

**Unknown biscuit identification:**

| File | Fitted $r$ (nm) | $\chi^2_\text{red}$ | Identified as |
|---|---|---|---|
| tr1 | ~200 ± ~15 | ~1.1 | Rich Tea |
| tr2 | ~600 ± ~20 | ~1.3 | Digestive |
| tr3 | ~380 ± ~10* | ~5.8 | Hobnob |

*Confidence interval underestimated due to inconsistent $\delta L$ — see Gap 1.

**Unsupervised GMM** on microscopy data recovers three components whose means match the supervised per-biscuit means to within 5 nm, without using any labels.

---

## Installation

```bash
git clone https://github.com/your-username/biscuit-dunking.git
cd biscuit-dunking
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook notebook_simple_v2.ipynb
```

---

## Usage

Run the notebook top to bottom. Each tier is self-contained:

- **Cells 1–45** — imports, data loading, Washburn validation, classification
- **Cells 46–60** — regression, hybrid pipeline, identifying the unknown biscuits
- **Cells 61–131** — statistical tests, diagnostics, GMM, residual analysis
- **Gap cells** — three additional analyses inserted at their natural positions

All constants ($\gamma_\text{tr}$, $\phi_\text{tr}$, $\eta_\text{tr}$) are defined at the top of the data loading cell and need only be changed there if rerunning with different experimental conditions.

---

## Reproducibility

All random operations use `np.random.seed(42)` and `random_state=42` throughout. The notebook is designed to run in a single sequential pass with no hidden state between cells.

Expected runtimes on a modern laptop:

| Section | Approximate time |
|---|---|
| Bootstrap fits (2000 resamples × 3 files) | ~25 s |
| Bayesian posterior grids (4000 points × 3 files) | ~10 s |
| Grid search (classifier + regressor) | ~40 s |
| Learning curve | ~30 s |
| Full notebook end to end | ~2 min |

---

## Limitations and Open Questions

- **tr3 $\chi^2_\text{red} \approx 5.8$** suggests either underreported measurement uncertainties or a physical process (e.g. biscuit swelling) not captured by the static Washburn model. Gap 1 rescales the errors; a time-varying $r(t)$ model would be the proper fix.
- **Hobnob classification errors concentrate at the boundary** between Rich Tea and Hobnob distributions, suggesting these two biscuits share an overlapping range of pore radii. Additional discriminating features (e.g. dunk force, acoustic emission) could improve separation.
- **The Washburn model assumes a single effective pore radius** per measurement. Real biscuits have a distribution of pore sizes; the inferred $r_\text{eff}$ is a harmonic mean weighted by flux, not the arithmetic mean visible in microscopy.
- **The hybrid pipeline uses the delta method** for uncertainty propagation, which linearises around the mean pore radius. A full bootstrap of the pipeline would give a more honest interval, especially for predictions near biscuit boundaries.

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
jupyter
```

See `requirements.txt` for pinned versions.

---

*Project completed as part of SCIFM0002. Data collected and provided by the course.*
