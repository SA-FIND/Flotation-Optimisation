"""
Tree-based models for Flotation Analysis.
"""
from typing import Any
import xgboost as xgb
from src.utils.config import get_section_config

def get_xgboost_model(**kwargs: Any) -> xgb.XGBRegressor:
    """
    Returns an XGBRegressor loaded with parameters from config.yaml,
    with any passed keyword arguments overriding the defaults.
    """
    params = get_section_config('xgboost')
    params.update(kwargs)
    return xgb.XGBRegressor(**params)
