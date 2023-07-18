from CassiSystem import CassiSystem
from utils import *
import matplotlib.pyplot as plt

config_system = load_yaml_config("config/cassi_system.yml")
config_masks = load_yaml_config("config/filtering.yml")
config_acquisition = load_yaml_config("config/acquisition.yml")

scene_directory = "./datasets/"
scene_name = "PaviaU"
results_directory = "experiment_results"

if __name__ == '__main__':

    # Initialize the CASSI system
    cassi_system = CassiSystem(system_config_path="config/cassi_system.yml")

    # SCENE : Load the hyperspectral scene
    cassi_system.load_dataset(scene_name, scene_directory)

    # MASK : Generate the dmd mask
    cassi_system.generate_2D_mask(config_masks)

    # PROPAGATION : Propagate the mask grid to the detector plane
    cassi_system.propagate_mask_grid()

    # FILTERING CUBE : Generate the filtering cube
    cassi_system.generate_filtering_cube()

    # PSF (optional) : Generate the psf
    psf = cassi_system.generate_psf("Gaussian",100)

    # ACQUISITION : Simulate the acquisition with psf (optional)
    cassi_system.image_acquisition(use_psf=True,chunck_size=50)

    plt.imshow(cassi_system.last_measurement_3D[:,:,0],cmap='gray')
    plt.show()

    # Save the acquisition
    cassi_system.save_acquisition(config_masks, config_acquisition)