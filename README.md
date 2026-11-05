# Flotation ML: leak-free grade forecasting in iron ore flotation

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project forecasts hourly concentrate grade (% Fe and % SiO2) in an industrial iron ore reverse flotation circuit from its process historian, and measures every model against a persistence baseline that repeats the last assay.

The historian (737,453 rows at about 20-second intervals) stores each hourly laboratory assay as a value repeated on every row until the next one. In 320 hours at least one concentrate assay takes a new value on every row instead, on a straight line; those values are interpolated, not measured, so the pipeline leaves those hours out (`results/interpolated_hours.txt`). It rebuilds the rest as one row per real assay (3,381 rows), with lags of earlier assays and statistics of the process sensors over the interval before each assay. `tests/test_leakage.py` fails if any feature is computed from the current target, and `scripts/make_synthetic_flotation.py` writes a synthetic historian whose best achievable R² is about 0.5, so a near-perfect score on it points to a leak. Models are chosen by `TimeSeriesSplit` cross-validation on the training period only.

## Results

Correction, October 2026: earlier versions of this README reported R² = 0.993 (Fe) and 0.994 (SiO₂). Those numbers came from target leakage. The momentum features `y.diff(lag)` contained the current assay, and the test set was passed to XGBoost as `eval_set`. Both are fixed, and `tests/test_leakage.py` fails on the old features.

Correction, November 2026: the October results (Ridge R² = 0.435 Fe, 0.516 SiO₂) still scored 84 assays that fall in interpolated hours, 12 of them in the test period. The interpolated silica values sit 0.03 from the previous assay on average, against 0.54 for real assays, so they flattered the silica scores. The numbers below exclude those hours.

Temporal 80/20 split: train 10 Mar to 7 Aug 2017 (2,704 assays), test 7 Aug to 9 Sep 2017 (677 assays). Skill is 1 - MSE(model) / MSE(persistence), so any positive value beats repeating the last assay.

| Model (test period) | Fe conc. R² | Fe skill | SiO₂ conc. R² | SiO₂ skill |
|---|---|---|---|---|
| Persistence, y(t) = y(t-1) | 0.247 | 0 | 0.394 | 0 |
| Sensor-only XGBoost (soft sensor) | -0.120 | below 0 | -0.123 | below 0 |
| Hybrid XGBoost (sensors + AR lags) | 0.350 | +0.137 | 0.419 | +0.041 |
| **Ridge on past assays (selected by CV)** | **0.402** | **+0.206** | **0.496** | **+0.168** |
| LSTM (sequence length 20) | 0.050 | below 0 | -0.280 | below 0 |

The 95% moving-block bootstrap interval on the Ridge skill is +0.157 to +0.251 for Fe and +0.117 to +0.217 for SiO₂ (`results_paper.txt`).

Hourly concentrate grade on this plant is predictable mainly from its own recent history. No model with process sensor inputs (window means, previous-interval states, all window statistics) reached a lower cross-validated error than the best model on past assays alone, for either target, and models on the sensors alone score below persistence.

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
python scripts/split_ladder.py
python scripts/check_interpolated_hours.py > results/interpolated_hours.txt
```

`run_case_study_2.py` gives the persistence, sensor-only, hybrid and LSTM rows. `improve_case_study_2.py` gives the cross-validated model selection, and `make_paper_figures.py` gives the bootstrap intervals and the leakage figure. `split_ladder.py` scores the same models for % SiO₂ under random-row, random-hour, time-ordered and rolling splits (`results/split_ladder.csv`): random 20-second rows give a random forest R² of 0.92, and a time-ordered split with the interpolated hours removed takes every sensor-only model below 0. On Linux or macOS, activate the environment with `source .venv/bin/activate`.

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
