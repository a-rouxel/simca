from cassi_systems import CassiSystem
from utils.helpers import load_yaml_config

config_dataset = load_yaml_config("config/dataset.yml")
config_system = load_yaml_config("config/cassi_system.yml")
config_masks = load_yaml_config("config/filtering.yml")
config_acquisition = load_yaml_config("config/acquisition.yml")

dataset_name = "PaviaU"

if __name__ == '__main__':

    # Initialize the CASSI system
    cassi_system = CassiSystem(system_config=config_system)

    # DATASET : Load the hyperspectral dataset
    cassi_system.load_dataset(dataset_name, config_dataset["datasets directory"])

    # MASK : Generate the dmd mask
    cassi_system.generate_2D_mask(config_masks)

    # PROPAGATION : Propagate the mask grid to the detector plane
    cassi_system.propagate_mask_grid()

    # FILTERING CUBE : Generate the filtering cube
    cassi_system.generate_filtering_cube()

    # PSF (optional) : Generate the psf
    cassi_system.generate_psf(type="Gaussian",radius=100)

    # ACQUISITION : Simulate the acquisition with psf (use_psf is optional)
    cassi_system.image_acquisition(use_psf=True,chunck_size=50)

    # Save the acquisition
    cassi_system.save_acquisition(config_masks, config_acquisition)
