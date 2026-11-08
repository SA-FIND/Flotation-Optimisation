import unittest
import numpy as np
import pandas as pd
from src.utils.correlation import compute_spearman, compute_pearson, normality_check, rank_features_by_spearman

class TestCorrelation(unittest.TestCase):
    def setUp(self):
        # Monotonic non-linear relationship: y = x^3
        x = np.linspace(1, 10, 50)
        y = x ** 3
        z = np.random.RandomState(42).randn(50)
        self.df = pd.DataFrame({'x': x, 'y': y, 'noise': z})

    def test_spearman_monotonic_capture(self):
        rho, pvals = compute_spearman(self.df)
        # Perfect monotonic relationship between x and y -> Spearman rho should be exactly 1.0
        self.assertAlmostEqual(rho.loc['x', 'y'], 1.0, places=4)
        self.assertLess(pvals.loc['x', 'y'], 0.001)

    def test_pearson_vs_spearman(self):
        pr, _ = compute_pearson(self.df)
        sr, _ = compute_spearman(self.df)
        # Pearson will be less than 1.0 due to non-linearity, while Spearman is 1.0
        self.assertLess(pr.loc['x', 'y'], 1.0)
        self.assertAlmostEqual(sr.loc['x', 'y'], 1.0, places=4)

    def test_normality_check(self):
        # Uniform or cubic data will fail normality
        results = normality_check(self.df)
        self.assertEqual(len(results), 3)
        self.assertIn('variable', results.columns)
        self.assertIn('is_normal', results.columns)

    def test_rank_features_by_spearman(self):
        rho, _ = compute_spearman(self.df)
        ranked = rank_features_by_spearman(rho, targets=['y'])
        self.assertIn('feature', ranked.columns)
        self.assertEqual(ranked.iloc[0]['feature'], 'x')

if __name__ == '__main__':
    unittest.main()
