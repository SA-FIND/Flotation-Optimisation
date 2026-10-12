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
