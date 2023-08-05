from cassi_systems import CassiSystem, load_yaml_config

config_dataset = load_yaml_config("cassi_systems/configs/dataset.yml")
config_system = load_yaml_config("cassi_systems/configs/cassi_system.yml")
config_patterns = load_yaml_config("cassi_systems/configs/pattern.yml")
config_acquisition = load_yaml_config("cassi_systems/configs/acquisition.yml")

dataset_name = "indian_pines"

if __name__ == '__main__':

    # Initialize the CASSI system
    cassi_system = CassiSystem(system_config=config_system)

    # DATASET : Load the hyperspectral dataset
    cassi_system.load_dataset(dataset_name, config_dataset["datasets directory"])

    # pattern : Generate the coded aperture pattern
    cassi_system.generate_2D_pattern(config_patterns)

    # PROPAGATION : Propagate the pattern grid to the detector plane
    cassi_system.propagate_coded_aperture_grid()

    # FILTERING CUBE : Generate the filtering cube
    cassi_system.generate_filtering_cube()

    # PSF (optional) : Generate the psf
    cassi_system.optical_model.generate_psf(type="Gaussian",radius=100)

    # ACQUISITION : Simulate the acquisition with psf (use_psf is optional)
    cassi_system.image_acquisition(use_psf=True,chunck_size=50)

    # Save the acquisition
    cassi_system.save_acquisition(config_patterns, config_acquisition)
