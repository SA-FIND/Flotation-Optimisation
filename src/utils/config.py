"""
Configuration loader for Flotation ML pipeline.
"""
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG = {
    'xgboost': {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    },
    'ridge': {
        'alpha': 1.0
    },
    'lstm': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 150,
        'patience': 20
    }
}

def get_project_root() -> Path:
    """Returns the root directory of the Flotation project."""
    # Since config.py is in src/utils/, root is 2 levels up
    return Path(__file__).resolve().parent.parent.parent

def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file, merging with default fallbacks.
    """
    if config_path is None:
        config_path = get_project_root() / 'config.yaml'
    else:
        config_path = Path(config_path)

    config = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}

    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    for section, values in loaded.items():
                        if isinstance(values, dict) and section in config:
                            config[section].update(values)
                        elif isinstance(values, dict):
                            config[section] = values
        except ImportError:
            pass  # Fallback to DEFAULT_CONFIG if pyyaml is not installed
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")

    return config

def get_section_config(section: str) -> Dict[str, Any]:
    """Retrieves a specific configuration section."""
    config = load_config()
    return config.get(section, DEFAULT_CONFIG.get(section, {}))
