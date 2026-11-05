"""
Figures and uncertainty estimates for the iron ore manuscript.

Uses the same assay events, features and 80/20 temporal split as
scripts/improve_case_study_2.py. Models are the ones selected there by
TimeSeriesSplit CV on the training period (Ridge AR-basic for Fe, Ridge
AR-ext for SiO2); nothing here is tuned on the test period.

Outputs (docs/figures/, results_paper.txt):
  fig1_test_forecast.png  test-period assays, persistence and Ridge forecasts
  fig2_skill.png          test skill vs persistence with 95% block-bootstrap CI
  fig3_leakage.png        R2 with the leaky momentum feature vs the past-only one
  fig4_acf.png            autocorrelation of assay grade by lag

Run:  python scripts/make_paper_figures.py > results_paper.txt
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from improve_case_study_2 import TARGETS, add_features, extract_events  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
SHORT = {"% Iron Concentrate": "Fe", "% Silica Concentrate": "SiO2"}
BLOCK = 24          # assays per bootstrap block (about one day)
N_BOOT = 2000


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))


def hybrid_xgb():
    return xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.5, min_child_weight=10, reg_lambda=5.0,
                            n_jobs=2, random_state=0)


def block_bootstrap_skill(y, pred, base, rng):
    """95% CI of 1 - MSE(pred)/MSE(base) under moving-block resampling of the test period."""
    n = len(y)
    starts = np.arange(n - BLOCK + 1)
    e_m, e_b = (y - pred) ** 2, (y - base) ** 2
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, n // BLOCK + 1)])[:n]
        out[i] = 1 - e_m[idx].mean() / e_b[idx].mean()
    return np.percentile(out, [2.5, 97.5])


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    a = add_features(extract_events())
    split = int(0.8 * len(a))
    train, test = a.iloc[:split], a.iloc[split:]
    sensor_mean = [c for c in a.columns if c.endswith("_mean") and not c.startswith("%")]

    selected = {"% Iron Concentrate": "basic", "% Silica Concentrate": "ext"}
    preds, skills, leak = {}, {}, {}
    print(f"Test assays: {len(test)} ({test.date.min()} to {test.date.max()}); "
          f"block = {BLOCK} assays, {N_BOOT} resamples\n")
    for t in TARGETS:
        other = [x for x in TARGETS if x != t][0]
        ar_basic = [f"{t}_lag{k}" for k in (1, 2, 3)] + [f"{t}_mom12", f"{t}_mom13"]
        ar_ext = ar_basic + [f"{t}_rm6", f"{t}_rm24", f"{t}_ewm", f"{other}_lag1"]
        cols = ar_ext if selected[t] == "ext" else ar_basic
        y_tr, y_te = train[t].values, test[t].values
        base = test[f"{t}_lag1"].values

        p_ridge = ridge().fit(train[cols], y_tr).predict(test[cols])
        p_hyb = hybrid_xgb().fit(train[ar_ext + sensor_mean], y_tr).predict(test[ar_ext + sensor_mean])
        p_sens = hybrid_xgb().fit(train[sensor_mean], y_tr).predict(test[sensor_mean])
        preds[t] = (y_te, base, p_ridge)

        print(f"=== {t} ===")
        skills[t] = {}
        for name, p in [("Sensor-only XGBoost", p_sens), ("Hybrid XGBoost", p_hyb),
                        ("Ridge (CV-selected)", p_ridge)]:
            s = 1 - mean_squared_error(y_te, p) / mean_squared_error(y_te, base)
            lo, hi = block_bootstrap_skill(y_te, p, base, rng)
            skills[t][name] = (s, lo, hi)
            print(f"  {name:22s} R2 = {r2_score(y_te, p):+.3f}  skill = {s:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

        # y(t) - y(t-1) contains the target itself, so this refit measures how far one leaky
        # feature inflates R2.
        leaky = test[t].values - test[f"{t}_lag1"].values
        leaky_tr = train[t].values - train[f"{t}_lag1"].values
        Xl_tr = np.column_stack([train[ar_basic].values, leaky_tr])
        Xl_te = np.column_stack([test[ar_basic].values, leaky])
        r2_leak = r2_score(y_te, hybrid_xgb().fit(Xl_tr, y_tr).predict(Xl_te))
        r2_fix = r2_score(y_te, hybrid_xgb().fit(train[ar_basic], y_tr).predict(test[ar_basic]))
        leak[t] = (r2_leak, r2_fix)
        print(f"  Leakage demo (XGBoost, AR lags): with y_t - y_t-1 R2 = {r2_leak:.4f}; "
              f"past-only R2 = {r2_fix:.4f}\n")

    # The full 34-day test period is too dense to see individual assays.
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    for ax, t in zip(axes, TARGETS):
        y, base, p = preds[t]
        d = test.date.values
        keep = d <= d[0] + np.timedelta64(10, "D")
        ax.plot(d[keep], y[keep], "k.-", lw=0.8, ms=3, label="Laboratory assay")
        ax.plot(d[keep], base[keep], color="0.6", lw=1, ls="--", label="Persistence")
        ax.plot(d[keep], p[keep], color="tab:blue", lw=1.2, label="Ridge (CV-selected)")
        ax.set_ylabel(f"{SHORT[t]} in concentrate (%)")
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "fig1_test_forecast.png", dpi=300)

    names = ["Sensor-only XGBoost", "Hybrid XGBoost", "Ridge (CV-selected)"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    w = 0.38
    for j, t in enumerate(TARGETS):
        s = np.array([skills[t][n][0] for n in names])
        lo = np.array([skills[t][n][1] for n in names])
        hi = np.array([skills[t][n][2] for n in names])
        x = np.arange(len(names)) + (j - 0.5) * w
        ax.bar(x, s, w, label=SHORT[t], color=["tab:orange", "tab:blue"][j])
        ax.errorbar(x, s, yerr=[s - lo, hi - s], fmt="none", ecolor="k", capsize=3, lw=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel("Skill vs persistence")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_skill.png", dpi=300)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(TARGETS))
    ax.bar(x - 0.2, [leak[t][0] for t in TARGETS], 0.4, color="tab:red", label="With y(t) - y(t-1) (leaky)")
    ax.bar(x + 0.2, [leak[t][1] for t in TARGETS], 0.4, color="tab:blue", label="Past assays only")
    ax.set_xticks(x, [SHORT[t] for t in TARGETS])
    ax.set_ylabel("Test R2")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_leakage.png", dpi=300)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    lags = np.arange(1, 25)
    for t, c in zip(TARGETS, ["tab:orange", "tab:blue"]):
        s = a[t]
        ax.plot(lags, [s.autocorr(k) for k in lags], "o-", ms=3, color=c, label=SHORT[t])
    ax.set_xlabel("Lag (assays, about 1 h each)")
    ax.set_ylabel("Autocorrelation")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_acf.png", dpi=300)
    print(f"Figures written to {FIG.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
