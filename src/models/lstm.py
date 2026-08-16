"""
PyTorch LSTM models for Flotation Analysis.
"""
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    Dataset = object
    DataLoader = None

from src.utils.config import get_section_config

if TORCH_AVAILABLE:
    class AssayDataset(Dataset):
        """Generic dataset for sequence-to-target pairs."""
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            
        def __len__(self):
            return len(self.X)
            
        def __getitem__(self, i):
            return self.X[i], self.y[i]

    class SimpleLSTM(nn.Module):
        """Simple 1-layer LSTM for LOOCV on small datasets (e.g., UG2)."""
        def __init__(self, n_features: int, hidden_size: int = 16):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    class FloatLSTM(nn.Module):
        """2-layer LSTM for processing continuous sequence data."""
        def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, output_size: int = 2, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, output_size)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.dropout(out)
            return self.fc(out)

    class AssayLSTM(nn.Module):
        """LSTM model for assay-level time-series prediction."""
        def __init__(self, n_features: int, hidden: int = 128, layers: int = 2, dropout: float = 0.3, output_size: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(
                n_features,
                hidden,
                layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0
            )
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Sequential(
                nn.Linear(hidden, 64), 
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, output_size)
            )
            
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :]))


def get_assay_lstm(n_features: int, output_size: int = 1, **kwargs: Any):
    """Factory helper to build an AssayLSTM with config defaults."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to build AssayLSTM.")
    cfg = get_section_config('lstm')
    hidden = kwargs.get('hidden_size', cfg.get('hidden_size', 128))
    layers = kwargs.get('num_layers', cfg.get('num_layers', 2))
    dropout = kwargs.get('dropout', cfg.get('dropout', 0.3))
    return AssayLSTM(n_features=n_features, hidden=hidden, layers=layers, dropout=dropout, output_size=output_size)


def run_lstm_loocv(X: np.ndarray, y: np.ndarray, epochs: int = 150, device: str = 'cpu') -> np.ndarray:
    """
    Leave-One-Out Cross-Validation for SimpleLSTM on small datasets.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for run_lstm_loocv.")
    n_samples, n_features = X.shape
    y_pred = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Leave one out
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i+1]
        
        # Format for LSTM: (batch, seq_len=1, features)
        X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        X_test_t  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
        
        model = SimpleLSTM(n_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_train_t)
            loss = criterion(out, y_train_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            pred = model(X_test_t).item()
        y_pred[i] = pred
        
    return y_pred


def train_lstm_model(model, train_loader, test_loader=None, epochs: Optional[int] = None, 
                     lr: Optional[float] = None, patience: Optional[int] = None, 
                     device: str = 'cpu', verbose: bool = True):
    """
    Standard training loop for PyTorch LSTM models with early stopping.
    Pulls default epochs, lr, patience from config.yaml if not explicitly provided.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for train_lstm_model.")
    cfg = get_section_config('lstm')
    epochs = epochs if epochs is not None else cfg.get('epochs', 150)
    lr = lr if lr is not None else cfg.get('learning_rate', 1e-3)
    patience = patience if patience is not None else cfg.get('patience', 20)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_count = 0
    train_losses, val_losses = [], []

    if verbose:
        print(f"Training for up to {epochs} epochs (early stop patience={patience})...")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * len(Xb)
        train_loss = running / len(train_loader.dataset)
        train_losses.append(train_loss)

        if test_loader is not None:
            model.eval()
            running_val = 0.0
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    pred = model(Xb)
                    running_val += criterion(pred, yb).item() * len(Xb)
            val_loss = running_val / len(test_loader.dataset)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    if verbose:
                        print(f"Early stop at epoch {epoch} (best val_loss={best_val_loss:.5f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses
