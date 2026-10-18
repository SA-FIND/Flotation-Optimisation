# Flotation ML: leak-free grade forecasting in iron ore flotation

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project forecasts hourly concentrate grade (% Fe and % SiO2) in an industrial iron ore reverse flotation circuit from its process historian, and measures every model against a persistence baseline that repeats the last assay.

The historian (737,453 rows at about 20-second intervals) stores each hourly laboratory assay as a value repeated on every row until the next one. The pipeline rebuilds it as one row per real assay (3,381 rows), with lags of earlier assays and statistics of the process sensors over the interval before each assay. `tests/test_leakage.py` fails if any feature is computed from the current target, and `scripts/make_synthetic_flotation.py` writes a synthetic historian whose best achievable R² is about 0.5, so a near-perfect score on it points to a leak. Models are chosen by `TimeSeriesSplit` cross-validation on the training period only.

## Results

Correction, October 2026: earlier versions of this README reported R² = 0.993 (Fe) and 0.994 (SiO₂). Those numbers came from target leakage. The momentum features `y.diff(lag)` contained the current assay, and the test set was passed to XGBoost as `eval_set`. Both are fixed, and `tests/test_leakage.py` fails on the old features.

Temporal 80/20 split: train 10 Mar to 7 Aug 2017 (2,704 assays), test 7 Aug to 9 Sep 2017 (677 assays). Skill is 1 - MSE(model) / MSE(persistence), so any positive value beats repeating the last assay.

| Model (test period) | Fe conc. R² | Fe skill | SiO₂ conc. R² | SiO₂ skill |
|---|---|---|---|---|
| Persistence, y(t) = y(t-1) | 0.293 | 0 | 0.439 | 0 |
| Sensor-only XGBoost (soft sensor) | -0.150 | below 0 | -0.241 | below 0 |
| Hybrid XGBoost (sensors + AR lags) | 0.377 | +0.119 | 0.429 | -0.018 |
| **Ridge on past assays (selected by CV)** | **0.435** | **+0.200** | **0.516** | **+0.138** |
| LSTM (sequence length 20) | 0.014 | below 0 | 0.012 | below 0 |

The 95% moving-block bootstrap interval on the Ridge skill is +0.146 to +0.249 for Fe and +0.073 to +0.194 for SiO₂ (`results_paper.txt`).

Hourly concentrate grade on this plant is predictable mainly from its own recent history. Adding the process sensors (window means, previous-interval states, all window statistics) did not lower cross-validated error for either target, and models on the sensors alone score below persistence.

## Reproduce the numbers

1. Download the Kaggle dataset described in `data/README.md` and save it as `data/iron_ore_flotation.csv`.
2. Install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the tests, then the three scripts behind the results:

```bash
python -m pytest -q
python src/run_case_study_2.py > results_fixed.txt
python scripts/improve_case_study_2.py > results_improved.txt
python scripts/make_paper_figures.py > results_paper.txt
```

`run_case_study_2.py` gives the persistence, sensor-only, hybrid and LSTM rows. `improve_case_study_2.py` gives the cross-validated model selection, and `make_paper_figures.py` gives the bootstrap intervals and the leakage figure. On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Project structure

```text
├── data/          # datasets, gitignored (see data/README.md)
├── notebooks/     # exploratory notebooks; the scripts are the reference results
├── scripts/       # CV model selection, paper figures, synthetic leak-check data
├── src/           # assay-event pipeline, models, utilities
├── tests/         # unit tests, including the leakage test
└── results_*.txt  # outputs of the scripts above
```

## Author

Solomon Ahedor, BSc Metallurgical Engineering (2026), Kwame Nkrumah University of Science and Technology (KNUST)

## License

MIT
