import matplotlib.pyplot as plt

from CassiSystem import CassiSystem
from utils.helpers import load_yaml_config
import os

config_dataset = load_yaml_config("config/dataset.yml")
config_system = load_yaml_config("config/cassi_system.yml")
config_masks = load_yaml_config("config/filtering.yml")
config_acquisition = load_yaml_config("config/acquisition.yml")

dataset_name = "F_fluocompact"
results_directory = "./data/results/slit_scanning_high_res"
nb_of_acq = 240

if __name__ == '__main__':

    # Initialize the CASSI system
    cassi_system = CassiSystem(system_config=config_system)

    # DATASET : Load the hyperspectral dataset
    cassi_system.load_dataset(dataset_name, config_dataset["datasets directory"])

    for i in range (nb_of_acq):
        # MASK : Generate the dmd mask
        new_slit_position = config_masks["mask"]["slit position"] + 1
        config_masks["mask"]["slit position"] = new_slit_position
        cassi_system.generate_2D_mask(config_masks)

        # PROPAGATION : Propagate the mask grid to the detector plane
        cassi_system.propagate_mask_grid()

        # FILTERING CUBE : Generate the filtering cube
        cassi_system.generate_filtering_cube()

        # ACQUISITION : Simulate the acquisition with psf (use_psf is optional)
        cassi_system.image_acquisition(use_psf=False,chunck_size=50)

        # Save the acquisition
        cassi_system.result_directory =results_directory
        os.makedirs(results_directory, exist_ok=True)

        if i == 0:
            cassi_system.save_config_system("config_system")
            cassi_system.save_config_mask_and_filtering(config_masks,"config_mask_and_filtering")

            cassi_system.save_interpolated_scene("interpolated_scene")
            cassi_system.save_panchromatic_image("panchro")
            cassi_system.save_wavelengths("wavelengths")

        cassi_system.save_measurement(f"measurement_{i}")
        cassi_system.save_filtering_cube(f"filtering_cube_{i}")
        cassi_system.save_mask(f"mask_{i}")

