# Flotation ML: Machine Learning for Grade and Recovery Prediction in Froth Flotation

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Framework: XGBoost](https://img.shields.io/badge/XGBoost-2.0+-0082f0.svg)](https://xgboost.ai/)

## Overview

**Flotation ML** is an applied machine learning pipeline designed to optimize grade and recovery predictions in industrial froth flotation circuits. Addressing the persistent challenges of non-linear process dynamics and sampling limitations, this project empirically validates sample-size thresholds on UG2 platinum circuits and implements robust time-series forecasting for iron ore circuits without data leakage.

---

## Core Features

- **Small-Batch Empirical Validation (UG2 Circuit):** Evaluates Leave-One-Out Cross-Validation (LOOCV) across three structural data tiers (run-level $n=3$, stream-level $n=33$, and assay-level $n=32$) to determine minimum $n/p$ thresholds for generalizable signal extraction.
- **Assay-Level Time-Series Forecasting (Iron Ore Circuit):** Replaces deeply flawed, forward-filled continuous sensor datasets (1 million rows) with explicit assay-event datasets (3,552 rows) engineered with Autoregressive (AR) lag features.
- **Algorithm Comparison:** Benchmarks Ridge Regression (L2 regularization), XGBoost (Gradient Boosted Trees), and Long Short-Term Memory (LSTM) networks across both small batch and sequential batch paradigms.
- **Process Optimization:** Identifies actionable what-if scenarios, such as manipulating feed $Cr_2O_3$ and Underflow (U/F) split ratios, to maximize PGM+Au recovery.

---

## Model Performance & Discoveries

### 1. Iron Ore Time-Series Models (Assay-Level AR Lags)
By extracting actual assay events and utilizing Autoregressive lag features, we eliminated data leakage, resulting in highly robust models.
- **XGBoost (Iron %Fe):** $R^2 = 0.993$
- **XGBoost (Silica %SiO₂):** $R^2 = 0.994$
- **LSTM (Sequence Length 20):** Failed to outperform simple AR logic ($R^2 \approx 0.54$) due to limited sequence history depth relative to the strength of explicit lag features.

### 2. UG2 Small-Batch Models
- **Assay-Level ($n=32$):** Ridge Regression achieved $R^2 = 0.954$, proving highly effective at extracting linear chemical relationships for use as an XRF soft sensor.
- **Stream-Level ($n=33$):** XGBoost achieved $R^2 = 0.788$, successfully modeling the non-linear physics of the circuit.
- **Run-Level ($n=3$):** All models failed (negative $R^2$), establishing that an $n/p < 1$ ratio produces no generalizable signal in metallurgical models.

---

## Project Structure

```text
├── data/                  # Raw and processed datasets (CSV, XLSX) - gitignored
├── docs/                  # Academic manuscripts and literature reviews
├── models/                # Serialized model weights (.pkl, .pt) - gitignored
├── notebooks/             # Jupyter notebooks for interactive EDA and training
│   ├── model.ipynb        # Baseline legacy models
│   ├── model2.0.ipynb     # UG2 small-batch (Ridge vs XGBoost vs LSTM)
│   ├── model3.0.ipynb     # Initial Iron Ore sequence models
│   ├── model4.0.ipynb     # Iron Ore assay-level AR lags (Final)
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

- Open `notebooks/model2.0.ipynb` to explore the UG2 small-batch analysis.
- Open `notebooks/model4.0.ipynb` to explore the Iron Ore time-series forecasting.

Alternatively, the Python scripts in `src/` can be run directly from the command line to execute the full pipeline.

---

## Author
**Solomon Ahedor**  
BSc Metallurgical Engineering (2026), Kwame Nkrumah University of Science and Technology (KNUST)

## License

This project is open-source under the **MIT License**.
