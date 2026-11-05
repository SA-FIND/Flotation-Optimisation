"""Guards against target leakage in autoregressive features.

A feature row for assay t may only depend on assays before t. The test changes
the CURRENT target value and checks that no feature in that row moves. The old
diff()/pct_change() features failed this test.
"""
import numpy as np
import pandas as pd

from src.build_assay_dataset import add_autoregressive_features


def _frame(last_fe):
    fe = [60.0, 61.0, 62.5, 61.8, 63.0, last_fe]
    si = [3.0, 2.8, 2.5, 2.6, 2.2, 2.0]
    return pd.DataFrame({
        'pct_Iron_Concentrate': fe,
        'pct_Silica_Concentrate': si,
        'minutes_since_prev_assay': [np.nan] + [60.0] * 5,
    })


def test_features_do_not_depend_on_current_target():
    targets = ['pct_Iron_Concentrate', 'pct_Silica_Concentrate']
    a = add_autoregressive_features(_frame(64.0), targets)
    b = add_autoregressive_features(_frame(99.0), targets)  # only y_t changed
    feature_cols = [c for c in a.columns if c not in targets]
    last_a = a[feature_cols].iloc[-1]
    last_b = b[feature_cols].iloc[-1]
    changed = [c for c in feature_cols if not np.isclose(last_a[c], last_b[c], equal_nan=True)]
    assert changed == [], f"Features leak the current target: {changed}"


def test_interpolated_hours_are_not_assays():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from improve_case_study_2 import interpolated_rows

    hours = pd.to_datetime(["2017-04-01 10:00"] * 3 + ["2017-04-01 11:00"] * 3)
    df = pd.DataFrame({
        "date": hours,
        "% Iron Concentrate": [65.0, 65.0, 65.0, 65.1, 65.2, 65.3],  # second hour is a ramp
        "% Silica Concentrate": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    })
    assert interpolated_rows(df).tolist() == [False] * 3 + [True] * 3
