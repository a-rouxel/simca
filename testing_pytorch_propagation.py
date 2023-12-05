from simca import CassiSystem, load_yaml_config
import matplotlib.pyplot as plt

config_dataset = load_yaml_config("simca/configs/dataset.yml")
config_system = load_yaml_config("simca/configs/cassi_system.yml")
config_patterns = load_yaml_config("simca/configs/pattern.yml")
config_acquisition = load_yaml_config("simca/configs/acquisition.yml")

if __name__ == '__main__':

    # Initialize the CASSI system
    cassi_system = CassiSystem(system_config=config_system)


    # pattern : Generate the coded aperture pattern
    cassi_system.generate_2D_pattern(config_patterns)

    # PROPAGATION : Propagate the pattern grid to the detector plane
    # X_coordinates_propagated_coded_aperture, Y_coordinates_propagated_coded_aperture, system_wavelengths = cassi_system.propagate_coded_aperture_grid(use_torch=False)

    # print(system_wavelengths)
    # plt.scatter(X_coordinates_propagated_coded_aperture[:,:,0], Y_coordinates_propagated_coded_aperture[:,:,0],color='blue')
    # plt.scatter(X_coordinates_propagated_coded_aperture[:,:,45], Y_coordinates_propagated_coded_aperture[:,:,45],color='green')
    # plt.scatter(X_coordinates_propagated_coded_aperture[:,:,90], Y_coordinates_propagated_coded_aperture[:,:,90],color='red')
    # plt.show()

    X_coordinates_propagated_coded_aperture, Y_coordinates_propagated_coded_aperture, system_wavelengths = cassi_system.propagate_coded_aperture_grid(use_torch=True)


