# Flotation ML

This project forecasts the hourly concentrate grade (% Fe and % SiO₂) of an iron ore flotation plant from its process historian, and tests whether the process sensors make the forecast any better.

The data is a public historian from an iron ore reverse flotation plant: 737,453 rows at about 20-second intervals, March to September 2017, with hourly laboratory assays.

## Results

I compare every model with persistence, which predicts that the next assay equals the last one. Skill is 1 - MSE(model) / MSE(persistence), so a model with skill above 0 beats persistence.

The split is in time order: training from 10 Mar to 7 Aug 2017 (2,704 assays), testing from 7 Aug to 9 Sep 2017 (677 assays).

| Model (test period) | Fe R² | Fe skill | SiO₂ R² | SiO₂ skill |
|---|---|---|---|---|
| Persistence, y(t) = y(t-1) | 0.247 | 0 | 0.394 | 0 |
| XGBoost on sensors only | -0.120 | below 0 | -0.123 | below 0 |
| XGBoost on sensors and past assays | 0.350 | +0.137 | 0.419 | +0.041 |
| Ridge on past assays (chosen by CV) | 0.402 | +0.206 | 0.496 | +0.168 |
| LSTM (sequence length 20) | 0.050 | below 0 | -0.280 | below 0 |

The best model is a Ridge regression on past assays alone. Its 95% moving-block bootstrap interval on skill is +0.157 to +0.251 for Fe and +0.117 to +0.217 for SiO₂. No model with sensor inputs had a lower cross-validated error, and the sensor-only models do worse than persistence. On this plant, the grade mostly follows its own recent history.

Two choices change these numbers a lot:

- How the data is split. If the 20-second rows are split at random, a random forest on the sensors gets R² = 0.92 for SiO₂, because neighbouring rows carry the same assay. With a time-ordered split, every sensor-only model falls below 0 (`results/split_ladder.csv`).
- Features that contain the target. A momentum feature written as y(t) - y(t-1) includes the value being predicted. With it, XGBoost gets R² = 0.98 for both targets. Built from past assays only, the same feature gives 0.38 (Fe) and 0.48 (SiO₂). `tests/test_leakage.py` fails if any feature uses the current target.

## Data preparation

The historian repeats each hourly assay on every row until the next one comes in. I rebuild it as one row per real assay (3,381 rows), with lags of earlier assays and statistics of the sensors over the interval before each assay.

In 320 hours, at least one concentrate assay changes on every row along a straight line. Those values were interpolated, not measured, so I leave those hours out (`results/interpolated_hours.txt`).

Features, target form and hyperparameters are chosen by 5-fold `TimeSeriesSplit` on the training period only. The test period is scored once, after that.

`scripts/make_synthetic_flotation.py` writes a synthetic historian where the best possible R² is about 0.5. If a model scores close to 1 on it, a feature is leaking the target.

## Running it

1. Download the Kaggle dataset described in `data/README.md` and save it as `data/iron_ore_flotation.csv`.
2. Set up the environment, then run the tests and scripts:

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q
python src/run_pipeline.py > results/pipeline.txt                 # persistence, sensor-only, XGBoost and LSTM rows
python scripts/select_models.py > results/model_selection.txt     # cross-validated model selection
python scripts/make_paper_figures.py > results/paper_figures.txt  # bootstrap intervals and the leakage comparison
python scripts/split_ladder.py                                    # random vs time-ordered splits
python scripts/check_interpolated_hours.py > results/interpolated_hours.txt
```

## Repo layout

```text
data/       the dataset goes here (not tracked, see data/README.md)
src/        assay-event pipeline, models and utilities
scripts/    model selection, figures, split comparison, synthetic data
tests/      unit tests, including the leakage test
results/    outputs of the scripts above
```

## Data source

"Quality Prediction in a Mining Process" by Eduardo Magalhães Oliveira, on Kaggle.

## License and contact

MIT License. Solomon Ahedor, Department of Materials and Metallurgical Engineering, KNUST, Ghana. ahedorsolomon@gmail.com
