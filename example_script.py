import matplotlib.pyplot as plt
from CassiSystem import CassiSystem
from utils import *


system_config = load_yaml_config("config/cassi_system.yml")
simulation_config = load_yaml_config("config/filtering.yml")

scene_directory = "./datasets/"
scene_name = "PaviaU"
results_directory = "experiment_results"

if __name__ == '__main__':

    # Initialize the CASSI system object
    cassi_system = CassiSystem(system_config_path="config/cassi_system.yml")


    # SCENE : Load the scene
    cassi_system.load_scene(scene_name, scene_directory)


    # FILTERING CUBE : Generate the dmd mask
    cassi_system.generate_dmd_grid()
    cassi_system.generate_2D_mask(simulation_config["mask"]["type"],simulation_config["mask"]["slit position"],simulation_config["mask"]["slit width"])
    # Propagate the mask grid to the detector plane
    cassi_system.propagate_mask_grid([system_config["spectral range"]["wavelength min"],system_config["spectral range"]["wavelength max"]], system_config["spectral range"]["number of spectral samples"])
    cassi_system.generate_filtering_cube()

    # ACQUISITION : Simulate the acquisition
    # Spectral interpolation of the scene to match filtering cube values
    scene = cassi_system.interpolate_scene(cassi_system.list_wavelengths, chunk_size=50)
    # crop if scene is too big
    interpolated_scene = match_scene_to_instrument(cassi_system.scene, cassi_system.filtering_cube)
    # Apply the filtering cube to the scene
    measurement_in_3D = get_measurement_in_3D(interpolated_scene, cassi_system.filtering_cube, chunk_size = 50)

    plt.imshow(np.sum(measurement_in_3D[:,:],axis=2))
    plt.show()
