import pytest
import torch
from src.models.lstm import SimpleLSTM, FloatLSTM, AssayLSTM

def test_simple_lstm_initialization():
    model = SimpleLSTM(n_features=10, hidden_size=16)
    assert isinstance(model, torch.nn.Module)
    assert model.lstm.input_size == 10
    assert model.lstm.hidden_size == 16

def test_float_lstm_initialization():
    model = FloatLSTM(input_size=20, hidden_size=128, num_layers=2, output_size=2)
    assert isinstance(model, torch.nn.Module)
    assert model.lstm.num_layers == 2
    
def test_assay_lstm_initialization():
    model = AssayLSTM(n_features=15, hidden=128, layers=2)
    assert isinstance(model, torch.nn.Module)

def test_simple_lstm_forward_shape():
    model = SimpleLSTM(n_features=5, hidden_size=16)
    # Batch size 2, seq_len 1, features 5
    dummy_input = torch.randn(2, 1, 5)
    output = model(dummy_input)
    assert output.shape == (2, 1)  # output should be (batch_size, 1)
