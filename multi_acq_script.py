import matplotlib.pyplot as plt

from cassi_systems import CassiSystem
from cassi_systems.functions_general_purpose import *
from utils.helpers import load_yaml_config
import os

config_dataset = load_yaml_config("cassi_systems/configs/dataset.yml")
config_system = load_yaml_config("cassi_systems/configs/cassi_system.yml")
config_masks = load_yaml_config("cassi_systems/configs/filtering.yml")
config_acquisition = load_yaml_config("cassi_systems/configs/acquisition.yml")

dataset_name = "F_fluocompact"
results_directory = "./data/results/slit_scanning_high_res"
nb_of_acq = 1

if __name__ == '__main__':

    # Initialize the CASSI system
    cassi_system = CassiSystem(system_config=config_system)

    # DATASET : Load the hyperspectral dataset
    cassi_system.load_dataset(dataset_name, config_dataset["datasets directory"])

    for i in range (nb_of_acq):
        # MASK : Generate the dmd mask
        new_slit_position = config_masks["mask"]["slit position"] + 1
        config_masks["mask"]["slit position"] = new_slit_position
        cassi_system.generate_2D_masks(config_masks)

        # PROPAGATION : Propagate the mask grid to the detector plane
        cassi_system.propagate_mask_grid()

        # FILTERING CUBE : Generate the filtering cube
        cassi_system.generate_filtering_cubes()

        # ACQUISITION : Simulate the acquisition with psf (use_psf is optional)
        cassi_system.image_acquisition(use_psf=False,chunck_size=50)

        # Save the acquisition
        cassi_system.result_directory =results_directory
        os.makedirs(results_directory, exist_ok=True)

        if i == 0:
            save_config_system("config_system", cassi_system.system_config, cassi_system.result_directory)
            save_config_mask_and_filtering("config_mask_and_filtering", config_masks,cassi_system.result_directory)

            save_interpolated_scene("interpolated_scene", cassi_system.interpolated_scene, cassi_system.result_directory)
            save_panchromatic_image("panchro", cassi_system.panchro, cassi_system.result_directory)
            save_wavelengths("wavelengths", cassi_system.system_wavelengths, cassi_system.result_directory)

        save_measurement(f"measurement_{i}",cassi_system.measurement,cassi_system.result_directory)
        save_filtering_cube(f"filtering_cube_{i}",cassi_system.filtering_cube,cassi_system.result_directory)
        save_mask(f"mask_{i}",cassi_system.mask,cassi_system.result_directory)



