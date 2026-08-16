import unittest
import numpy as np
from src.models.linear_models import get_ridge_model

class TestModels(unittest.TestCase):
    def test_ridge_initialization_and_fit(self):
        model = get_ridge_model(alpha=2.0)
        self.assertEqual(model.alpha, 2.0)

        # Fit with dummy linear data
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = np.array([3.0, 5.0, 7.0, 9.0])
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(preds.shape, (4,))

    def test_xgboost_initialization(self):
        try:
            import xgboost as xgb
            from src.models.tree_models import get_xgboost_model
            model = get_xgboost_model(max_depth=4)
            self.assertEqual(model.max_depth, 4)
            self.assertEqual(model.learning_rate, 0.05)

            # Fit with dummy data
            X = np.random.randn(20, 5)
            y = np.random.randn(20)
            model.fit(X, y)
            preds = model.predict(X)
            self.assertEqual(preds.shape, (20,))
        except ImportError:
            self.skipTest("xgboost not installed in current environment")

    def test_lstm_models(self):
        try:
            import torch
            from src.models.lstm import SimpleLSTM, FloatLSTM, AssayLSTM, get_assay_lstm

            # Test SimpleLSTM
            simple = SimpleLSTM(n_features=10, hidden_size=16)
            inp = torch.randn(2, 1, 10)
            out = simple(inp)
            self.assertEqual(out.shape, (2, 1))

            # Test FloatLSTM
            flt = FloatLSTM(input_size=20, hidden_size=64, num_layers=2, output_size=2)
            inp_flt = torch.randn(4, 15, 20)
            out_flt = flt(inp_flt)
            self.assertEqual(out_flt.shape, (4, 2))

            # Test AssayLSTM factory
            assay = get_assay_lstm(n_features=15, output_size=1)
            inp_assay = torch.randn(3, 20, 15)
            out_assay = assay(inp_assay)
            self.assertEqual(out_assay.shape, (3, 1))
        except ImportError:
            self.skipTest("torch not installed in current environment")

if __name__ == '__main__':
    unittest.main()
