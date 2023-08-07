import unittest
import numpy as np
import random
import glob
from ..functions_general_purpose import load_yaml_config
from ..CassiSystem import CassiSystem
import os

class TestCassiSystemInitialization(unittest.TestCase):

    def setUp(self):
        self.config_system = load_yaml_config('./cassi_systems/tests/test_configs/cassi_system.yml')
        self.config_dataset = load_yaml_config('./cassi_systems/tests/test_configs/dataset.yml')
    def test_config_system_loading(self):

        config_files = glob.glob('./cassi_systems/tests/test_configs/*.yml')

        for config_file in config_files:
            config_system = load_yaml_config(config_file)
            self.assertIsInstance(config_system, dict)

        self.config_system = load_yaml_config('./cassi_systems/tests/test_configs/cassi_system.yml')
        self.config_dataset = load_yaml_config('./cassi_systems/tests/test_configs/dataset.yml')
    def test_coded_aper_grid_generation(self):

        config_files = glob.glob('./cassi_systems/tests/test_configs/cassi_system*.yml')

        for config_file in config_files:
            config_system = load_yaml_config(config_file)

            cassi_system = CassiSystem(system_config=config_system)

            self.assertEqual(cassi_system.X_coded_aper_coordinates.shape, (config_system["coded aperture"]["number of pixels along Y"], config_system["coded aperture"]["number of pixels along X"]))
            self.assertEqual(cassi_system.Y_coded_aper_coordinates.shape, (config_system["coded aperture"]["number of pixels along Y"], config_system["coded aperture"]["number of pixels along X"]))

    def test_loading_dataset(self):

        # load all datasets in the datasets directory

        for dataset in glob.glob('./datasets/*'):

            dataset_name = os.path.basename(os.path.normpath(dataset))
            # dataset_name = dataset.split('\\')[-1].split('\\')[0]


            cassi_system = CassiSystem(system_config=self.config_system)

            cassi_system.load_dataset(dataset_name, self.config_dataset["datasets directory"])

            self.assertIsInstance(cassi_system.dataset, np.ndarray)
            self.assertEqual(cassi_system.dataset.ndim, 3)

            self.assertIsInstance(cassi_system.dataset_wavelengths, np.ndarray)
            self.assertIs(cassi_system.dataset_wavelengths.shape[0] > 0, True)
            self.assertEqual(cassi_system.dataset_wavelengths.shape[0], cassi_system.dataset.shape[2])

            if cassi_system.dataset_labels is not None :
                self.assertIsInstance(cassi_system.dataset_labels,np.ndarray)
                self.assertEqual(cassi_system.dataset_labels.ndim, 2)
                self.assertEqual(cassi_system.dataset_labels.shape[0], cassi_system.dataset.shape[0])
                self.assertEqual(cassi_system.dataset_labels.shape[1], cassi_system.dataset.shape[1])

                self.assertIsInstance(cassi_system.dataset_label_names, list)
                self.assertIs(len(cassi_system.dataset_label_names),len(list(np.unique(cassi_system.dataset_labels))))

                # ignored labels not tested

                self.assertIsInstance(cassi_system.dataset_palette, dict)

    def test_generate_pattern(self):

        cassi_system = CassiSystem(system_config=self.config_system)

        config_files = glob.glob('./cassi_systems/tests/test_configs/filtering_simple*.yml')

        for config_file in config_files:

            config_pattern = load_yaml_config(config_file)
            pattern = cassi_system.generate_2D_pattern(config_pattern)

            self.assertIsInstance(pattern, np.ndarray)
            self.assertEqual(pattern.shape, (self.config_system["coded aperture"]["number of pixels along Y"], self.config_system["coded aperture"]["number of pixels along X"]))

    import random

    def test_generate_filtering_cube(self, num_tests=5):

        config_system_files = glob.glob('./cassi_systems/tests/test_configs/cassi_system*.yml')
        config_patterns_files = glob.glob('./cassi_systems/tests/test_configs/filtering_simple*.yml')

        # Create all possible combinations of system and coded_aperture configs
        all_combinations = [(system, pattern) for system in config_system_files for pattern in config_patterns_files]

        # Randomly choose 'num_tests' combinations to test
        selected_combinations = random.sample(all_combinations, num_tests)

        for config_system_file, config_patterns_file in selected_combinations:
            config_system = load_yaml_config(config_system_file)
            config_patterns = load_yaml_config(config_patterns_file)

            cassi_system = CassiSystem(system_config=config_system)
            cassi_system.generate_2D_pattern(config_patterns)
            cassi_system.propagate_coded_aperture_grid()
            filtering_cube = cassi_system.generate_filtering_cube()

            self.assertIsInstance(filtering_cube, np.ndarray)
            self.assertEqual(filtering_cube.ndim, 3)
            self.assertEqual(filtering_cube.shape[0], cassi_system.Y_detector_coordinates_grid.shape[0])
            self.assertEqual(filtering_cube.shape[1], cassi_system.X_detector_coordinates_grid.shape[1])
            self.assertEqual(filtering_cube.shape[2], len(cassi_system.optical_model.system_wavelengths))

    def test_generate_multiple_patterns(self,num_tests=4,number_of_pattern=3):

        config_system_files = glob.glob('./cassi_systems/tests/test_configs/cassi_system*.yml')
        config_patterns_files = glob.glob('./cassi_systems/tests/test_configs/filtering_multiple*.yml')

        # Create all possible combinations of system and pattern configs
        all_combinations = [(system, coded_aperture) for system in config_system_files for coded_aperture in config_patterns_files]

        # Randomly choose 'num_tests' combinations to test
        selected_combinations = random.sample(all_combinations, num_tests)

        for config_system_file, config_patterns_file in selected_combinations:
            config_system = load_yaml_config(config_system_file)
            config_patterns = load_yaml_config(config_patterns_file)

            cassi_system = CassiSystem(system_config=config_system)
            cassi_system.generate_multiple_patterns(config_patterns,number_of_patterns=number_of_pattern)
            cassi_system.propagate_coded_aperture_grid()
            list_of_filtering_cubes = cassi_system.generate_multiple_filtering_cubes(number_of_patterns=number_of_pattern)

            self.assertIsInstance(list_of_filtering_cubes,list)
            self.assertEqual(len(list_of_filtering_cubes),number_of_pattern)
            self.assertEqual(list_of_filtering_cubes[0].ndim, 3)
            self.assertEqual(list_of_filtering_cubes[0].shape[0], cassi_system.Y_detector_coordinates_grid.shape[0])
            self.assertEqual(list_of_filtering_cubes[0].shape[1], cassi_system.X_detector_coordinates_grid.shape[1])
            self.assertEqual(list_of_filtering_cubes[0].shape[2], len(cassi_system.optical_model.system_wavelengths))

            self.assertEqual(list_of_filtering_cubes[-1].ndim, 3)
            self.assertEqual(list_of_filtering_cubes[-1].shape[0], cassi_system.Y_detector_coordinates_grid.shape[0])
            self.assertEqual(list_of_filtering_cubes[-1].shape[1], cassi_system.X_detector_coordinates_grid.shape[1])
            self.assertEqual(list_of_filtering_cubes[-1].shape[2], len(cassi_system.optical_model.system_wavelengths))



if __name__ == '__main__':
    unittest.main()


