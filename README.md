# Flotation ML

Can the sensors on an iron ore flotation plant forecast the next concentrate assay better than just repeating the last one? I tested this on a public plant historian: 737,453 rows at about 20-second intervals, with hourly laboratory assays of % Fe and % SiO₂ in the concentrate.

Every model is scored against persistence, y(t) = y(t-1). Skill is 1 - MSE(model) / MSE(persistence), so any positive value beats repeating the last assay.

## What I found

**1. Concentrate grade follows its own recent history.** On a time-ordered split (train 10 Mar to 7 Aug 2017, 2,704 assays; test 7 Aug to 9 Sep 2017, 677 assays), the best model is a Ridge regression on past assays alone. No model with sensor inputs had a lower cross-validated error, and models on the sensors alone score below persistence.

| Model (test period) | Fe R² | Fe skill | SiO₂ R² | SiO₂ skill |
|---|---|---|---|---|
| Persistence, y(t) = y(t-1) | 0.247 | 0 | 0.394 | 0 |
| XGBoost on sensors only | -0.120 | below 0 | -0.123 | below 0 |
| XGBoost on sensors and past assays | 0.350 | +0.137 | 0.419 | +0.041 |
| **Ridge on past assays (chosen by CV)** | **0.402** | **+0.206** | **0.496** | **+0.168** |
| LSTM (sequence length 20) | 0.050 | below 0 | -0.280 | below 0 |

The 95% moving-block bootstrap interval on the Ridge skill is +0.157 to +0.251 for Fe and +0.117 to +0.217 for SiO₂.

**2. The way you split the data decides the answer.** If the 20-second rows are split at random, a random forest on the sensors gets R² = 0.92 for SiO₂, because neighbouring rows share the same assay. Split in time order and every sensor-only model drops below 0 (`results/split_ladder.csv`).

**3. One leaky feature is enough to fake a near-perfect model.** A momentum feature written as y(t) - y(t-1) contains the value being predicted. With it, XGBoost reaches R² = 0.98 for both targets; with the same feature built from past assays only, it gets 0.38 (Fe) and 0.48 (SiO₂). `tests/test_leakage.py` fails if any feature is computed from the current target.

## How the data is prepared

The historian repeats each hourly assay on every row until the next one arrives. I rebuild it as one row per real assay (3,381 rows), with lags of earlier assays and statistics of the process sensors over the interval before each assay. In 320 hours, at least one concentrate assay changes on every row along a straight line. Those values are interpolated, not measured, so I leave those hours out (`results/interpolated_hours.txt`).

Model choices (features, target form, hyperparameters) are made by 5-fold `TimeSeriesSplit` on the training period only. The test period is scored once, after selection.

`scripts/make_synthetic_flotation.py` writes a synthetic historian whose best possible R² is about 0.5, so a near-perfect score on it points to a leak.

## Run it yourself

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

## What is in the repo

```text
data/       the dataset goes here (not tracked, see data/README.md)
src/        assay-event pipeline, models and utilities
scripts/    model selection, figures, split comparison, synthetic data
tests/      unit tests, including the leakage test
results/    outputs of the scripts above
```

## What comes next

- Run the same protocol on a second plant's historian to see whether the result holds.
- Forecast more than one assay ahead, with intervals.

## Data and credits

Data: "Quality Prediction in a Mining Process" by Eduardo Magalhães Oliveira, Kaggle. A public historian from an iron ore reverse flotation plant, March to September 2017.

## License and contact

MIT License. Solomon Ahedor, Department of Materials and Metallurgical Engineering, KNUST, Ghana. ahedorsolomon@gmail.com
