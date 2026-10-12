"""
Case Study 2: can we beat persistence honestly?

Protocol (no test-set peeking):
  * Same assay-event extraction and 80/20 temporal split as src/run_case_study_2.py.
  * Every model choice (features, target form, hyperparameters) is made by
    5-fold TimeSeriesSplit on the TRAIN period only. Selection metric: CV MSE.
  * The test period is scored once, for all candidates, after selection.
    The "selected" model is the one with the lowest CV MSE, not the best test score.

Run:  python scripts/improve_case_study_2.py > results_improved.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "iron_ore_flotation.csv"
CACHE = ROOT / "data" / "cache_cs2_assay_events.pkl"
TARGETS = ["% Iron Concentrate", "% Silica Concentrate"]
LAB_FEED = ["% Iron Feed", "% Silica Feed"]


def extract_events():
    """Vectorised version of STEP 2 in run_case_study_2.py (same windows, same stats)."""
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    df = pd.read_csv(CSV, sep=",", decimal=",", parse_dates=["date"])
    lab_all = TARGETS + LAB_FEED
    sensors = [c for c in df.columns if c not in lab_all + ["date"]]
    change = df[lab_all].diff().fillna(1).abs().sum(axis=1) > 0
    # Event k owns rows (event_{k-1}, event_k]; the first event owns rows [0, event_0].
    group = change.cumsum().shift(1).fillna(0).astype(int)
    group = group.clip(upper=change.sum() - 1)
    g = df[sensors].groupby(group.values)
    stats = g.agg(["mean", "std", "min", "max"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    ev = df.loc[change].reset_index(drop=True)
    inst = ev[sensors].add_suffix("_inst")
    out = pd.concat([inst, stats.reset_index(drop=True).iloc[: len(ev)]], axis=1)
    for c in lab_all:
        out[c] = ev[c].values
    out["date"] = ev["date"].values
    out["hour"] = ev["date"].dt.hour.values
    out["dow"] = ev["date"].dt.dayofweek.values
    out.to_pickle(CACHE)
    return out


def add_features(a):
    a = a.copy()
    for t in TARGETS:
        for lag in [1, 2, 3]:
            a[f"{t}_lag{lag}"] = a[t].shift(lag)
        a[f"{t}_mom12"] = a[f"{t}_lag1"] - a[f"{t}_lag2"]
        a[f"{t}_mom13"] = a[f"{t}_lag1"] - a[f"{t}_lag3"]
    base = a.dropna().reset_index(drop=True)  # identical rows to the pipeline
    # Extra past-only features (computed after dropna so the row set is unchanged)
    for t in TARGETS:
        past = base[t].shift(1)
        base[f"{t}_rm6"] = past.rolling(6, min_periods=1).mean()
        base[f"{t}_rm24"] = past.rolling(24, min_periods=1).mean()
        base[f"{t}_ewm"] = past.ewm(halflife=3).mean()
    for c in [c for c in base.columns if c.endswith("_mean")]:
        base[f"{c}_prev"] = base[c].shift(1)  # previous interval's process state
    return base.bfill()  # only fills the first row's new rolling/prev features


def cv_mse(make_model, X, y, offset, n_splits=5):
    errs = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        m = make_model().fit(X[tr], y[tr])
        errs.append(mean_squared_error(y[va] + offset[va], m.predict(X[va]) + offset[va]))
    return float(np.mean(errs))


def main():
    a = add_features(extract_events())
    split = int(0.8 * len(a))
    train, test = a.iloc[:split], a.iloc[split:]
    print(f"Rows {len(a)} | train {len(train)} ({train.date.min()} to {train.date.max()}) | "
          f"test {len(test)} ({test.date.min()} to {test.date.max()})")

    sensor_mean = [c for c in a.columns if c.endswith("_mean") and not c.startswith("%")]
    sensor_all = [c for c in a.columns if c.endswith(("_inst", "_mean", "_std", "_min", "_max"))
                  and not c.startswith("%")] + ["hour", "dow"]
    prev_cols = [c for c in a.columns if c.endswith("_prev")]

    for t in TARGETS:
        other = [x for x in TARGETS if x != t][0]
        ar_basic = [f"{t}_lag{k}" for k in (1, 2, 3)] + [f"{t}_mom12", f"{t}_mom13"]
        ar_ext = ar_basic + [f"{t}_rm6", f"{t}_rm24", f"{t}_ewm", f"{other}_lag1"]
        y_tr, y_te = train[t].values, test[t].values
        lag_tr, lag_te = train[f"{t}_lag1"].values, test[f"{t}_lag1"].values
        zero_tr, zero_te = np.zeros(len(train)), np.zeros(len(test))

        xgb_reg = lambda **k: (lambda: xgb.XGBRegressor(n_estimators=k.get("n", 300), max_depth=k.get("d", 3),
                                                        learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
                                                        min_child_weight=k.get("mcw", 10), reg_lambda=5.0,
                                                        n_jobs=2, random_state=0))
        ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))

        # (name, feature columns, model factory, predict delta y - lag1?)
        cands = [
            ("Ridge AR-basic", ar_basic, ridge, False),
            ("Ridge AR-ext", ar_ext, ridge, False),
            ("Ridge AR-ext + sensor means", ar_ext + sensor_mean, ridge, False),
            ("Ridge AR-ext + means + prev", ar_ext + sensor_mean + prev_cols, ridge, False),
            ("XGB AR-ext", ar_ext, xgb_reg(), False),
            ("XGB AR-ext + means, d2", ar_ext + sensor_mean, xgb_reg(d=2), False),
            ("XGB AR-ext + means, d3", ar_ext + sensor_mean, xgb_reg(d=3), False),
            ("XGB AR-ext + means + prev, d3", ar_ext + sensor_mean + prev_cols, xgb_reg(d=3), False),
            ("XGB AR-ext + all sensors, d3", ar_ext + sensor_all, xgb_reg(d=3), False),
            ("XGB delta, AR-ext + means, d3", ar_ext + sensor_mean, xgb_reg(d=3), True),
            ("XGB delta, AR-ext + means + prev, d2", ar_ext + sensor_mean + prev_cols, xgb_reg(d=2), True),
        ]

        mse_p_te = mean_squared_error(y_te, lag_te)
        cv_p = float(np.mean([mean_squared_error(y_tr[va], lag_tr[va])
                              for _, va in TimeSeriesSplit(5).split(y_tr)]))
        print(f"\n=== {t} ===")
        print(f"{'model':40s} {'CV MSE':>8s} {'CV skill':>9s} | {'test R2':>8s} {'test skill':>10s}")
        print(f"{'Persistence (y = lag1)':40s} {cv_p:8.4f} {0:+9.3f} | {r2_score(y_te, lag_te):8.4f} {0:+10.3f}")
        rows = []
        for name, cols, mk, delta in cands:
            Xtr, Xte = train[cols].values, test[cols].values
            off_tr, off_te = (lag_tr, lag_te) if delta else (zero_tr, zero_te)
            cv = cv_mse(mk, Xtr, y_tr - off_tr, off_tr)
            m = mk().fit(Xtr, y_tr - off_tr)
            pred = m.predict(Xte) + off_te
            r2 = r2_score(y_te, pred)
            skill = 1 - mean_squared_error(y_te, pred) / mse_p_te
            rows.append((cv, name, r2, skill))
            print(f"{name:40s} {cv:8.4f} {1 - cv / cv_p:+9.3f} | {r2:8.4f} {skill:+10.3f}")
        best = min(rows)
        print(f"--> Selected by CV (lowest CV MSE): {best[1]}  ->  test R2 = {best[2]:.4f}, "
              f"skill vs persistence = {best[3]:+.3f}")


if __name__ == "__main__":
    sys.exit(main())
