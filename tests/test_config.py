import unittest
from pathlib import Path
from src.utils.config import load_config, get_section_config, get_project_root, DEFAULT_CONFIG

class TestConfig(unittest.TestCase):
    def test_project_root_exists(self):
        root = get_project_root()
        self.assertTrue(root.exists(), f"Project root {root} does not exist")
        self.assertTrue((root / 'src').exists(), "src/ directory not found in project root")

    def test_load_default_config(self):
        config = load_config()
        self.assertIn('xgboost', config)
        self.assertIn('ridge', config)
        self.assertIn('lstm', config)
        self.assertEqual(config['xgboost']['max_depth'], 5)
        self.assertEqual(config['ridge']['alpha'], 1.0)

    def test_get_section_config(self):
        xgb_cfg = get_section_config('xgboost')
        self.assertIsInstance(xgb_cfg, dict)
        self.assertEqual(xgb_cfg['learning_rate'], 0.05)

        ridge_cfg = get_section_config('ridge')
        self.assertEqual(ridge_cfg['alpha'], 1.0)

        lstm_cfg = get_section_config('lstm')
        self.assertEqual(lstm_cfg['hidden_size'], 128)

if __name__ == '__main__':
    unittest.main()
