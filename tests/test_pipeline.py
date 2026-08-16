import unittest
import pandas as pd
import numpy as np
from src.build_assay_dataset import extract_assay_events, build_features, add_autoregressive_features

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Create synthetic historian dataset with 10 rows and 2 real assay updates
        # Assay 1 at row 0 (% Fe = 65.0)
        # Assay 2 at row 5 (% Fe changes to 67.0)
        dates = pd.date_range(start='2026-01-01 08:00', periods=10, freq='5min')
        fe_target = [65.0]*5 + [67.0]*5
        si_target = [2.0]*5 + [1.5]*5
        sensor_air = np.linspace(200, 250, 10)
        sensor_pulp = np.linspace(7.0, 7.5, 10)

        self.df = pd.DataFrame({
            'date': dates,
            '% Iron Concentrate': fe_target,
            '% Silica Concentrate': si_target,
            'Air Flow': sensor_air,
            'Ore Pulp pH': sensor_pulp
        })
        self.target_cols = ['% Iron Concentrate', '% Silica Concentrate']
        self.sensor_cols = ['Air Flow', 'Ore Pulp pH']

    def test_extract_assay_events(self):
        df_real, real_indices = extract_assay_events(self.df, self.target_cols)
        # Should identify row 0 (initial) and row 5 (value changed from 65.0 -> 67.0)
        self.assertEqual(len(df_real), 2)
        self.assertEqual(real_indices, [0, 5])
        self.assertEqual(df_real.iloc[0]['% Iron Concentrate'], 65.0)
        self.assertEqual(df_real.iloc[1]['% Iron Concentrate'], 67.0)

    def test_build_features_window_stats(self):
        df_real, real_indices = extract_assay_events(self.df, self.target_cols)
        feat_df = build_features(self.df, df_real, real_indices, self.sensor_cols, self.target_cols)
        self.assertEqual(len(feat_df), 2)
        self.assertIn('Air_Flow_mean', feat_df.columns)
        self.assertIn('Air_Flow_min', feat_df.columns)
        self.assertIn('Air_Flow_max', feat_df.columns)
        self.assertIn('Air_Flow_at_assay', feat_df.columns)

    def test_add_autoregressive_features_zero_leakage(self):
        # Create a 6-row assay dataframe
        data = {
            'pct_Iron_Concentrate': [60.0, 61.0, 62.0, 63.0, 64.0, 65.0],
            'pct_Silica_Concentrate': [3.0, 2.8, 2.5, 2.2, 2.0, 1.8],
            'minutes_since_prev_assay': [np.nan, 60.0, 60.0, 60.0, 60.0, 60.0],
            'datetime': pd.date_range('2026-01-01', periods=6, freq='1h')
        }
        df_assay = pd.DataFrame(data)
        target_cols = ['pct_Iron_Concentrate', 'pct_Silica_Concentrate']
        ml_df = add_autoregressive_features(df_assay, target_cols)

        # Ensure lag1 at row t is exactly value at t-1 (strictly backward looking)
        self.assertIn('pct_Iron_Concentrate_lag1', ml_df.columns)
        for i in range(len(ml_df)):
            current_target = ml_df.loc[i, 'pct_Iron_Concentrate']
            lag1_target = ml_df.loc[i, 'pct_Iron_Concentrate_lag1']
            # Target values are strictly increasing by 1.0, so current - lag1 must be exactly 1.0
            self.assertAlmostEqual(current_target - lag1_target, 1.0)

if __name__ == '__main__':
    unittest.main()
