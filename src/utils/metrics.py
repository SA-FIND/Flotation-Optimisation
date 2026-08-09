"""
Evaluation metrics for Flotation Analysis.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def print_evaluation_metrics(y_true, y_pred, target_names):
    """
    Computes and prints RMSE, MAE, and R2 for a list of targets.
    """
    for i, name in enumerate(target_names):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"  {name:<30}  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
