# Biscuit Dunking - Physics Informed Machine Learning for Pore Characterisation


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

## What is project ?

When you dunk a biscuit, tea climbs through its pores. The [Washburn equation](https://doi.org/10.1103/PhysRev.17.273) describes that climb:

$$L^2 = \frac{\gamma \, r \cos\varphi \, t}{2\eta}$$

This project tests how well the equation holds, builds a classifier to identify biscuit types from dunking measurements, and combines both into a hybrid pipeline that outperforms either approach alone.

---

## Key results

| | |
|--|--|
| Washburn R² (pore radius known) | **0.9996** |
| Classifier accuracy (with physics feature) | **96%** |
| Hybrid pipeline R² | **0.985** |
| Pure physics / pure ML (no radius) | ~0.79 each |

Blind samples: **tr1 = Hobnob** (519 nm), **tr2 = Rich Tea** (280 nm), **tr3 = Digestive** (1005 nm)

---

## How it works

1. **Washburn is nearly exact** when pore radius `r` is known - any failure comes from missing that input, not the physics.
2. **Inverting Washburn** gives an effective pore radius per measurement that cleanly separates biscuits: ~300 nm (Rich Tea), ~500 nm (Hobnob), ~800 nm (Digestive).
3. **A random forest** using that derived feature classifies biscuit type at 96%. Without it: 81%.
4. **The hybrid** classifies first, then feeds the matching mean radius into Washburn - R² jumps from 0.79 to 0.985 with realistic propagated uncertainties.

ML here fills in the missing physics input, not replace the physics.

---

## Repo structure

```
├── data/
│   ├── dunking-data.csv       # 3,000 labelled dunk measurements
│   ├── microscopy-data.csv    # 500 microscopy samples (includes r)
│   ├── tr-1.csv / tr-2.csv / tr-3.csv   # blind time-resolved files
├── Analysis_notebook.ipynb    # full analysis
```
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
---

## Run it

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
jupyter notebook Analysis_notebook.ipynb
```

---

## References

- Washburn (1921) *Phys. Rev.* **17**, 273
- Breiman (2001) *Mach. Learn.* **45**, 5
- Pedregosa et al. (2011) *JMLR* **12**, 2825
