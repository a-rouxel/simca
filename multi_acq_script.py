import matplotlib.pyplot as plt

from cassi_systems import CassiSystem
from cassi_systems.functions_general_purpose import *
import os

config_dataset = load_yaml_config("cassi_systems/configs/dataset.yml")
config_system = load_yaml_config("cassi_systems/configs/cassi_system.yml")
config_patterns = load_yaml_config("cassi_systems/configs/pattern.yml")
config_acquisition = load_yaml_config("cassi_systems/configs/acquisition.yml")

dataset_name = "indian_pines"
results_directory = "./data/results/lego_test_1"
nb_of_acq = 10

if __name__ == '__main__':

    # Initialize the CASSI system
    cassi_system = CassiSystem(system_config=config_system)

    # DATASET : Load the hyperspectral dataset
    cassi_system.load_dataset(dataset_name, config_dataset["datasets directory"])


    cassi_system.generate_multiple_patterns(config_patterns,nb_of_acq)

    # PROPAGATION : Propagate the pattern grid to the detector plane
    cassi_system.propagate_coded_aperture_grid()

    for i in range(nb_of_acq):
        plt.imshow(cassi_system.list_of_patterns[i])
        plt.show()

    # FILTERING CUBE : Generate the filtering cubes
    cassi_system.generate_multiple_filtering_cubes(nb_of_acq)

    # ACQUISITION : Simulate the acquisition with psf (use_psf is optional)
    cassi_system.multiple_image_acquisitions(use_psf=False,nb_of_filtering_cubes=nb_of_acq,chunck_size=50)

    # Save the acquisition
    cassi_system.result_directory =results_directory
    os.makedirs(results_directory, exist_ok=True)

    save_config_file("config_system", cassi_system.system_config, cassi_system.result_directory)
    save_config_file("config_pattern", config_patterns,cassi_system.result_directory)
    save_config_file("config_acquisition", config_acquisition, cassi_system.result_directory)

    save_data_in_hdf5("interpolated_scene",cassi_system.interpolated_scene, cassi_system.result_directory)
    save_data_in_hdf5("panchro",cassi_system.panchro,cassi_system.result_directory)

    save_data_in_hdf5("wavelengths",cassi_system.optical_model.system_wavelengths,cassi_system.result_directory)
    save_data_in_hdf5("list_of_compressed_measurements",cassi_system.list_of_measurements,cassi_system.result_directory)
    save_data_in_hdf5("list_of_filtering_cubes",cassi_system.list_of_filtering_cubes,cassi_system.result_directory)
    save_data_in_hdf5("list_of_patterns",cassi_system.list_of_patterns,cassi_system.result_directory)











