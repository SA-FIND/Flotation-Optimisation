"""
Split ladder: how the same models score on the same data under different splits.

Target: % Silica Concentrate. % Iron Concentrate is never an input because it is
the target's near twin (both come from the same hourly assay).

Data rules:
  * Rows before the 16-29 March 2017 gap are dropped (a short, separate block).
  * Hours in which the silica or iron assay changes on every row are linear
    interpolations between real assays, so they are left out of training and
    scoring ("filled" hours). One variant keeps them to show what they do.
  * Persistence repeats the previous real hourly assay. Skill is
    1 - MSE(model) / MSE(persistence) on the same test rows.

Splits:
  A  random 80/20 over 20-second rows (about 180 rows share one assay)
  B  random 80/20 over hourly assays
  C  time order, first 80% of hours train, last 20% test
  D  rolling origin, 5 folds, 24 h gap between train and test

Run:  python scripts/split_ladder.py
Writes results/split_ladder.csv and results/split_ladder.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "iron_ore_flotation.csv"
OUT = ROOT / "results"
SEED = 42
TARGET = "% Silica Concentrate"
TWIN = "% Iron Concentrate"
FEED = ["% Iron Feed", "% Silica Feed"]
GAP_END = pd.Timestamp("2017-03-29 12:00")
CALENDAR = ["hour", "dow", "doy"]


def models(n_rows):
    # Fully grown forests on ~590k rows need tens of GB; subsampling per tree keeps
    # split A within laptop memory and still lets the forest memorise assays.
    rf_kw = dict(max_samples=0.25) if n_rows > 100_000 else {}
    return {
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RF": lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1,
                                            random_state=SEED, **rf_kw),
        "GBM": lambda: HistGradientBoostingRegressor(random_state=SEED),
    }


def load():
    df = pd.read_csv(CSV, decimal=",", parse_dates=["date"])
    df = df[df.date >= GAP_END].reset_index(drop=True)
    g = df.groupby("date")
    filled = (g[TARGET].nunique() > 1) | (g[TWIN].nunique() > 1)
    sensors = [c for c in df.columns if c not in ("date", TARGET, TWIN)]
    hours = g[sensors].mean()
    hours[TARGET] = g[TARGET].first()
    hours["filled"] = filled
    hours = hours.reset_index()
    for frame in (df, hours):
        frame["hour"] = frame.date.dt.hour
        frame["dow"] = frame.date.dt.dayofweek
        frame["doy"] = frame.date.dt.dayofyear
    return df, hours, sensors


def add_last_assay(hours, keep_filled):
    h = hours if keep_filled else hours[~hours.filled]
    h = h.copy()
    h["last_assay"] = h[TARGET].shift(1)
    return h.dropna(subset=["last_assay"]).reset_index(drop=True)


def dm_test(e_model, e_ref):
    """Diebold-Mariano on squared-error loss, Newey-West variance. Negative stat: model better."""
    d = e_model ** 2 - e_ref ** 2
    n = len(d)
    lag = int(n ** (1 / 3))
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def score(train, test, features, order_col="date"):
    test = test.sort_values(order_col)
    y = test[TARGET].values
    ref = test["last_assay"].values
    row = {"n_train": len(train), "n_test": len(test), "n_test_assays": test.date.nunique(),
           "Persistence": r2_score(y, ref)}
    mse_ref = mean_squared_error(y, ref)
    best = None
    for name, make in models(len(train)).items():
        pred = make().fit(train[features], train[TARGET]).predict(test[features])
        row[name] = r2_score(y, pred)
        skill = 1 - mean_squared_error(y, pred) / mse_ref
        if best is None or skill > best[1]:
            best = (name, skill, y - pred)
    row["best_model"], row["best_skill"] = best[0], best[1]
    row["dm_stat"], row["dm_p"] = dm_test(best[2], y - ref)
    return row


def rolling(hours, features, n_splits=5, gap=24):
    folds = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(hours):
        folds.append(score(hours.iloc[tr], hours.iloc[te], features))
    out = {}
    for k in ["Ridge", "RF", "GBM", "Persistence", "best_skill"]:
        v = np.array([f[k] for f in folds])
        out[k], out[f"{k}_min"], out[f"{k}_max"] = v.mean(), v.min(), v.max()
    out["best_model"] = max(["Ridge", "RF", "GBM"], key=lambda m: out[m])
    out["n_train"], out["n_test"] = folds[-1]["n_train"], sum(f["n_test"] for f in folds)
    out["n_test_assays"] = out["n_test"]
    return out


def main():
    df, hours, sensors = load()
    no_feed = [c for c in sensors if c not in FEED]
    h_clean = add_last_assay(hours, keep_filled=False)
    h_kept = add_last_assay(hours, keep_filled=True)
    print(f"Hours after the gap: {len(hours)} | filled: {int(hours.filled.sum())} | "
          f"clean hourly assays with a previous assay: {len(h_clean)}")

    rows = df[df.date.isin(h_clean.date)].merge(h_clean[["date", "last_assay"]], on="date")
    results = {}

    tr, te = train_test_split(rows, test_size=0.2, random_state=SEED)
    results["A: random 20 s rows"] = score(tr, te, sensors)
    results["A + calendar inputs"] = score(tr, te, sensors + CALENDAR)

    tr, te = train_test_split(h_clean, test_size=0.2, random_state=SEED)
    results["B: random hourly assays"] = score(tr, te, sensors)
    results["B + calendar inputs"] = score(tr, te, sensors + CALENDAR)

    def time_split(h):
        cut = int(0.8 * len(h))
        return h.iloc[:cut], h.iloc[cut:]

    tr, te = time_split(h_clean)
    results["C: time order 80/20"] = score(tr, te, sensors)
    results["C + calendar inputs"] = score(tr, te, sensors + CALENDAR)
    results["C without feed assays"] = score(tr, te, no_feed)
    results["C + last assay as input"] = score(tr, te, sensors + ["last_assay"])
    results["C, last assay only"] = score(tr, te, ["last_assay"])
    results["C + filled hours kept"] = score(*time_split(h_kept), sensors)

    results["D: rolling, 5 folds, 24 h gap"] = rolling(h_clean, sensors)
    results["D + last assay as input"] = rolling(h_clean, sensors + ["last_assay"])

    table = pd.DataFrame(results).T
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "split_ladder.csv")
    meta = {"rows_after_gap": int(len(df)), "hours_after_gap": int(len(hours)),
            "filled_hours": int(hours.filled.sum()), "clean_hourly_assays": int(len(h_clean)),
            "seed": SEED, "results": json.loads(table.to_json(orient="index"))}
    (OUT / "split_ladder.json").write_text(json.dumps(meta, indent=2))
    cols = ["Ridge", "RF", "GBM", "Persistence", "best_skill", "dm_p", "n_test", "n_test_assays"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(table[cols])


if __name__ == "__main__":
    main()
