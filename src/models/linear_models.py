"""
Linear models for Flotation Analysis.
"""
from typing import Any
from sklearn.linear_model import Ridge
from src.utils.config import get_section_config

def get_ridge_model(**kwargs: Any) -> Ridge:
    """
    Returns a Ridge regressor initialized with default hyperparameters from config.yaml,
    with any passed keyword arguments overriding the defaults.
    """
    params = get_section_config('ridge')
    params.update(kwargs)
    return Ridge(**params)
