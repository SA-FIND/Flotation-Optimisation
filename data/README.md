# Data

Raw data files are not committed (size and licensing).

## Iron ore flotation historian (Case Study 2)

- Source: Kaggle, "Quality Prediction in a Mining Process" by Eduardo Magalhaes Oliveira:
  https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process
- File: `MiningProcess_Flotation_Plant_Database.csv`, 737,453 rows, March to September 2017.
- Save it here as `data/iron_ore_flotation.csv` (comma separator, comma decimals). Do not open and re-save it in Excel: Excel truncates at 1,048,575 rows and can change number formats.

## UG2 platinum circuit data (Case Study 1)

Private plant data used with permission. Not redistributed. Contact the author.

## Synthetic data

`python scripts/make_synthetic_flotation.py data/iron_ore_flotation.csv` writes a synthetic historian with a known, limited signal (best achievable R2 about 0.5). Use it to check the pipeline for leakage: any R2 near 1.0 on this file means a feature contains the target.
