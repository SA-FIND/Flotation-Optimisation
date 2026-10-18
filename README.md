# Flotation ML: Leak-Free Grade Forecasting in Iron Ore Flotation

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Framework: XGBoost](https://img.shields.io/badge/XGBoost-2.0+-0082f0.svg)](https://xgboost.ai/)

## Overview

**Flotation ML** forecasts hourly concentrate grade (% Fe and % SiO2) in an industrial iron ore reverse flotation circuit from its process historian. The work focuses on getting an honest number: removing the forward-fill and target leakage that make published soft-sensor results on this dataset look near-perfect, and measuring every model against a naive persistence baseline.

---

## Core Features

- **Assay-event extraction:** Replaces the forward-filled 20-second historian (737,453 rows) with one row per real laboratory assay (3,381 rows), so no target value is copied across rows.
- **Past-only features:** Autoregressive lags of earlier assays plus window statistics of the process sensors over the interval before each assay.
- **Leakage tests:** `tests/test_leakage.py` fails if a feature contains the current target; `scripts/make_synthetic_flotation.py` builds a historian with a known ceiling to check the whole pipeline.
- **Model comparison:** Persistence, Ridge, XGBoost and LSTM, with model choice made by `TimeSeriesSplit` cross-validation on the training period only.

---

## Results

### Assay-level forecasting (leak-free)

> **Correction (October 2026).** Earlier versions of this README reported $R^2 = 0.993$ (Fe) and $0.994$ (SiO₂). Those numbers came from target leakage: the momentum features `y.diff(lag)` contained the current assay, and the test set was passed to XGBoost as `eval_set`. Both are fixed; `tests/test_leakage.py` fails on the old features. `scripts/make_synthetic_flotation.py` builds a synthetic historian with a known ceiling (best $R^2 \approx 0.5$) to check the pipeline for leaks.

Temporal 80/20 split: train Mar 10 to Aug 7 2017 (2,704 assays), test Aug 7 to Sep 9 2017 (677 assays). Skill = $1 - MSE_{model}/MSE_{persistence}$.

| Model (test period) | Fe conc. $R^2$ | Fe skill | SiO₂ conc. $R^2$ | SiO₂ skill |
|---|---|---|---|---|
| Persistence ($y_t = y_{t-1}$) | 0.293 | 0 | 0.439 | 0 |
| Sensor-only XGBoost (soft sensor) | -0.150 | n/a | -0.241 | n/a |
| Hybrid XGBoost (sensors + AR lags) | 0.377 | +0.119 | 0.429 | -0.018 |
| **Ridge on past assays (selected by CV)** | **0.435** | **+0.200** | **0.516** | **+0.138** |
| LSTM (sequence length 20) | 0.014 | n/a | 0.012 | n/a |

The selected models were chosen by 5-fold `TimeSeriesSplit` on the training period only; the test period was scored once (`scripts/improve_case_study_2.py`, output in `results_improved.txt`).

**Finding:** in this plant's data, hourly concentrate grade is predictable mainly from its own recent history. Adding the process sensors (window means, previous-interval states, all window statistics) did not lower cross-validated error for either target.

---

## Project Structure

```text
├── data/                  # Raw and processed datasets (CSV, XLSX) - gitignored
├── docs/                  # Academic manuscripts and literature reviews
├── models/                # Serialized model weights (.pkl, .pt) - gitignored
├── notebooks/             # Jupyter notebooks for interactive EDA and training
│   ├── model.ipynb        # Baseline legacy models
│   ├── model3.0.ipynb     # Initial Iron Ore sequence models
│   ├── model4.0.ipynb     # Iron Ore assay-level AR lags (Final)
├── scripts/               # CV model selection and synthetic leak-check data
├── src/                   # Python source code for data pipelines and executable models
│   ├── run_case_study_2.py # Executable pipeline for Iron Ore time-series models
│   ├── analyse_iron.py    # EDA script for Iron Ore data
├── tests/                 # Unit tests for ML models and preprocessing logic
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## Tech Stack

- **Core Engineering:** Python 3.13+, Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, PyTorch

---

## Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/SA-FIND/Flotation-Optimisation.git
cd Flotation-Optimisation

# Create and activate virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Models

The core modeling workflows are housed within the `notebooks/` directory. You can launch Jupyter to interact with the pipelines:

```bash
pip install jupyter
jupyter notebook
```

- Open `notebooks/model4.0.ipynb` to explore the Iron Ore time-series forecasting.

Alternatively, the Python scripts in `src/` can be run directly from the command line to execute the full pipeline.

---

## Author
**Solomon Ahedor**  
BSc Metallurgical Engineering (2026), Kwame Nkrumah University of Science and Technology (KNUST)

## License

This project is open-source under the **MIT License**.
