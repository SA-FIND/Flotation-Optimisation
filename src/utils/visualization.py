"""
Visualization utilities for Flotation Analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def plot_actual_vs_predicted(y_true, y_pred, target_names, title="Actual vs Predicted"):
    """
    Plots scatter plots of actual vs predicted values.
    """
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(8*n_targets, 5))
    if n_targets == 1:
        axes = [axes]
        
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20, color='steelblue')
        mn = min(y_true[:, i].min(), y_pred[:, i].min())
        mx = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect')
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        ax.set_xlabel(f'Actual {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}  (R² = {r2:.4f})')
        ax.legend()
        
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_training_curve(train_losses, val_losses, title="Training Curve"):
    """
    Plots the training and validation loss curves.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label='Train loss')
    ax.plot(val_losses,   label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
