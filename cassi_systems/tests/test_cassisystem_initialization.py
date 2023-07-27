import unittest
import glob
from ..functions_general_purpose import load_yaml_config
from ..CassiSystem import CassiSystem

class TestCassiSystemInitialization(unittest.TestCase):

    def test_config_system_loading(self):

        config_files = glob.glob('./cassi_systems/configs/cassi_system*.yml')

        for config_file in config_files:
            config_system = load_yaml_config(config_file)
            self.assertIsInstance(config_system, dict)

    def test_dmd_grid_generation(self):

        config_files = glob.glob('./cassi_systems/configs/cassi_system*.yml')

        for config_file in config_files:
            config_system = load_yaml_config(config_file)

            cassi_system = CassiSystem(system_config=config_system)

            self.assertEqual(cassi_system.X_dmd_coordinates_grid.shape, (config_system["SLM"]["sampling across Y"], config_system["SLM"]["sampling across X"]))
            self.assertEqual(cassi_system.Y_dmd_coordinates_grid.shape, (config_system["SLM"]["sampling across Y"], config_system["SLM"]["sampling across X"]))


if __name__ == '__main__':
    unittest.main()

