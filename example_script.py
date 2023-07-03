from CassiSystem import CassiSystem
from utils import *


system_config = load_yaml_config("config/cassi_system.yml")
simulation_config = load_yaml_config("config/filtering.yml")

scene_directory = "./datasets/"
scene_name = "PaviaU"
results_directory = "experiment_results"



cassi_system = CassiSystem(system_config_path="config/cassi_system.yml")


cassi_system.load_scene(scene_name, scene_directory)

cassi_system.create_dmd_mask()
cassi_system.generate_2D_mask(simulation_config["mask"]["type"],simulation_config["mask"]["slit position"],simulation_config["mask"]["slit width"])

cassi_system.propagate_mask_grid([system_config["spectral range"]["wavelength min"],
                                                        system_config["spectral range"]["wavelength max"]],
                                                        system_config["spectral range"]["number of spectral samples"])
# cassi_system.generate_filtering_cube()
#
# scene = match_scene_to_instrument(cassi_system, filtering_cube)
#
# self.finished_interpolated_scene.emit(scene)
#
# # Define chunk size
# chunk_size = 50  # Adjust this value based on your system's memory
#
# t_0 = time.time()
# measurement_in_3D = get_measurement_in_3D(scene, filtering_cube, chunk_size)
# print("Acquisition time: ", time.time() - t_0)
#
# self.last_measurement_3D = measurement_in_3D
# self.interpolated_scene = scene
#
# self.finished_acquire_measure.emit(self.last_measurement_3D)  # Emit a tuple of arrays