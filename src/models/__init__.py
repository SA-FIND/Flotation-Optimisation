"""Models package for Flotation Analysis."""
from .tree_models import get_xgboost_model
from .linear_models import get_ridge_model
from .lstm import TORCH_AVAILABLE

# LSTM classes exist only when PyTorch is installed; the tree and linear
# pipelines must still import cleanly without it.
if TORCH_AVAILABLE:
    from .lstm import FloatLSTM, AssayLSTM, get_assay_lstm, train_lstm_model

__all__ = [
    'get_xgboost_model',
    'get_ridge_model',
    'FloatLSTM',
    'AssayLSTM',
    'get_assay_lstm',
    'train_lstm_model',
]
