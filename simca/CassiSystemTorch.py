from simca.OpticalModel import OpticalModelTorch
from simca.functions_acquisition import *
from simca.functions_patterns_generation import *
from simca.functions_scenes import *
from simca.functions_general_purpose import *
from CassiSystem import CassiSystem
from functions_acquisition_torch import *

class CassiSystemTorch(CassiSystem):
    """Class that contains the cassi system main attributes and methods"""

    def __init__(self, system_config=None, system_config_path=None):

        """

        Args:
            system_config_path (str): path to the configs file
            system_config (dict): system configuration

        """
        super().__init__(system_config=system_config, system_config_path=system_config_path)
        self.set_up_system(system_config_path=system_config_path, system_config=system_config)

    def set_up_system(self, system_config_path=None, system_config=None):
        """
        Loading system config & initializing the grids coordinates for the coded aperture and the detector

        Args:
            system_config_path (str): path to the configs file
            system_config (dict): system configuration

        """

        if system_config_path is not None:
            self.system_config = load_yaml_config(system_config_path)
        elif system_config is not None:
            self.system_config = system_config

        self.optical_model = OpticalModelTorch(self.system_config)

        self.X_coded_aper_coordinates, self.Y_coded_aper_coordinates = self.create_coordinates_grid(
            self.system_config["coded aperture"]["number of pixels along X"],
            self.system_config["coded aperture"]["number of pixels along Y"],
            self.system_config["coded aperture"]["pixel size along X"],
            self.system_config["coded aperture"]["pixel size along Y"])

        self.X_detector_coordinates_grid, self.Y_detector_coordinates_grid = self.create_coordinates_grid(
            self.system_config["detector"]["number of pixels along X"],
            self.system_config["detector"]["number of pixels along Y"],
            self.system_config["detector"]["pixel size along X"],
            self.system_config["detector"]["pixel size along Y"])
    
    def update_config(self, system_config_path=None, system_config=None):

        """
        Update the system configuration file and re-initialize the grids for the coded aperture and the detector

        Args:
            system_config_path (str): path to the configs file
            system_config (dict): system configuration
        Returns:
            dict: updated system configuration

        """

        self.set_up_system(system_config_path=system_config_path, system_config=system_config)

        return self.system_config
        
    def generate_filtering_cube(self):
        """
        Generate filtering cube : each slice of the cube is a propagated pattern interpolated on the detector grid

        Returns:
        numpy.ndarray: filtering cube generated according to the optical system & the pattern configuration (R x C x W)

        """

        self.filtering_cube = interpolate_data_on_grid_positions_torch(data=self.pattern,
                                                                 X_init=self.X_coordinates_propagated_coded_aperture,
                                                                 Y_init=self.Y_coordinates_propagated_coded_aperture,
                                                                 X_target=self.X_detector_coordinates_grid,
                                                                 Y_target=self.Y_detector_coordinates_grid)


        return self.filtering_cube
    
    def generate_multiple_filtering_cubes(self, number_of_patterns):
        """
        Generate multiple filtering cubes, each cube corresponds to a pattern, and for each pattern, each slice is a propagated coded apertureinterpolated on the detector grid

        Args:
            number_of_patterns (int): number of patterns to generate
        Returns:
            list: filtering cubes generated according to the current optical system and the pattern configuration

        """
        self.list_of_filtering_cubes = []

        for idx in range(number_of_patterns):

            self.filtering_cube = interpolate_data_on_grid_positions_torch(data=self.list_of_patterns[idx],
                                                                     X_init=self.X_coordinates_propagated_coded_aperture,
                                                                     Y_init=self.Y_coordinates_propagated_coded_aperture,
                                                                     X_target=self.X_detector_coordinates_grid,
                                                                     Y_target=self.Y_detector_coordinates_grid)

            self.list_of_filtering_cubes.append(self.filtering_cube)

        return self.list_of_filtering_cubes
    
    def image_acquisition(self, use_psf=False, chunck_size=50):
        """
        Run the acquisition/measurement process depending on the cassi system type

        Args:
            chunck_size (int): default block size for the interpolation

        Returns:
            numpy.ndarray: compressed measurement (R x C)
        """

        dataset = self.interpolate_dataset_along_wavelengths_torch(self.optical_model.system_wavelengths, chunck_size)

        if dataset is None:
            return None
        dataset_labels = self.dataset_labels

        if self.system_config["system architecture"]["system type"] == "DD-CASSI":

            try:
                self.filtering_cube
            except:
                return print("Please generate filtering cube first")

            scene = torch.from_numpy(match_dataset_to_instrument(dataset, self.filtering_cube))

            measurement_in_3D = generate_dd_measurement_torch(scene, self.filtering_cube, chunck_size)

            self.last_filtered_interpolated_scene = measurement_in_3D
            self.interpolated_scene = scene

            if dataset_labels is not None:
                scene_labels = torch.from_numpy(match_dataset_labels_to_instrument(dataset_labels, self.filtering_cube))
                self.scene_labels = scene_labels


        elif self.system_config["system architecture"]["system type"] == "SD-CASSI":

            X_coded_aper_coordinates_crop = crop_center(self.X_coded_aper_coordinates,dataset.shape[1], dataset.shape[0])
            Y_coded_aper_coordinates_crop = crop_center(self.Y_coded_aper_coordinates,dataset.shape[1], dataset.shape[0])

            scene = torch.from_numpy(match_dataset_to_instrument(dataset, X_coded_aper_coordinates_crop))

            pattern_crop = crop_center(self.pattern, scene.shape[1], scene.shape[0])

            filtered_scene = scene * pattern_crop[..., None].repeat((1, 1, scene.shape[2]))

            self.propagate_coded_aperture_grid(X_input_grid=X_coded_aper_coordinates_crop, Y_input_grid=Y_coded_aper_coordinates_crop, use_torch = True)

            sd_measurement = interpolate_data_on_grid_positions_torch(filtered_scene,
                                                                self.X_coordinates_propagated_coded_aperture,
                                                                self.Y_coordinates_propagated_coded_aperture,
                                                                self.X_detector_coordinates_grid,
                                                                self.Y_detector_coordinates_grid)

            self.last_filtered_interpolated_scene = sd_measurement
            self.interpolated_scene = scene

            if dataset_labels is not None:
                scene_labels = torch.from_numpy(match_dataset_labels_to_instrument(dataset_labels, self.last_filtered_interpolated_scene))
                self.scene_labels = scene_labels

        self.panchro = torch.sum(self.interpolated_scene, dim=2)

        if use_psf:
            self.apply_psf_torch()
        else:
            print("No PSF was applied")

        # Calculate the other two arrays
        self.measurement = torch.sum(self.last_filtered_interpolated_scene, dim=2)

        return self.measurement

    def multiple_image_acquisitions(self, use_psf=False, nb_of_filtering_cubes=1,chunck_size=50):
        """
        Run the acquisition process depending on the cassi system type

        Args:
            chunck_size (int): default block size for the dataset

        Returns:
             list: list of compressed measurements (list of numpy.ndarray of size R x C)
        """

        dataset = self.interpolate_dataset_along_wavelengths_torch(self.optical_model.system_wavelengths, chunck_size)
        if dataset is None:
            return None
        dataset_labels = self.dataset_labels

        self.list_of_filtered_scenes = []

        if self.system_config["system architecture"]["system type"] == "DD-CASSI":
            try:
                self.list_of_filtering_cubes
            except:
                return print("Please generate list of filtering cubes first")

            scene = torch.from_numpy(match_dataset_to_instrument(dataset, self.list_of_filtering_cubes[0]))

            if dataset_labels is not None:
                scene_labels = torch.from_numpy(match_dataset_labels_to_instrument(dataset_labels, self.filtering_cube))
                self.scene_labels = scene_labels

            self.interpolated_scene = scene

            for i in range(nb_of_filtering_cubes):

                filtered_scene = generate_dd_measurement_torch(scene, self.list_of_filtering_cubes[i], chunck_size)
                self.list_of_filtered_scenes.append(filtered_scene)

        elif self.system_config["system architecture"]["system type"] == "SD-CASSI":

            X_coded_aper_coordinates_crop = crop_center(self.X_coded_aper_coordinates,dataset.shape[1], dataset.shape[0])
            Y_coded_aper_coordinates_crop = crop_center(self.Y_coded_aper_coordinates,dataset.shape[1], dataset.shape[0])

            scene = torch.from_numpy(match_dataset_to_instrument(dataset, X_coded_aper_coordinates_crop))

            if dataset_labels is not None:
                scene_labels = torch.from_numpy(match_dataset_labels_to_instrument(dataset_labels, self.filtering_cube))
                self.scene_labels = scene_labels

            self.interpolated_scene = scene

            for i in range(nb_of_filtering_cubes):

                mask_crop = crop_center(self.list_of_patterns[i], scene.shape[1], scene.shape[0])

                filtered_scene = scene * mask_crop[..., None].repeat((1, 1, scene.shape[2]))

                self.propagate_coded_aperture_grid(X_input_grid=X_coded_aper_coordinates_crop, Y_input_grid=Y_coded_aper_coordinates_crop, use_torch = True)

                sd_measurement_cube = interpolate_data_on_grid_positions_torch(filtered_scene,
                                                                    self.X_coordinates_propagated_coded_aperture,
                                                                    self.Y_coordinates_propagated_coded_aperture,
                                                                    self.X_detector_coordinates_grid,
                                                                    self.Y_detector_coordinates_grid)
                self.list_of_filtered_scenes.append(sd_measurement_cube)

        self.panchro = torch.sum(self.interpolated_scene, dim=2)

        if use_psf:
            self.apply_psf_torch()
        else:
            print("No PSF was applied")

        # Calculate the other two arrays
        self.list_of_measurements = []
        for i in range(nb_of_filtering_cubes):
            self.list_of_measurements.append(torch.sum(self.list_of_filtered_scenes[i], dim=2))

        return self.list_of_measurements
    
    def apply_psf(self):
        """
        Apply the PSF to the last measurement

        Returns:
            numpy.ndarray: last measurement cube convolved with by PSF (shape= R x C x W). Each slice of the 3D filtered scene is convolved with the PSF
        """
        if (self.optical_model.psf is not None) and (self.last_filtered_interpolated_scene is not None):
            # Expand the dimensions of the 2D matrix to match the 3D matrix
            psf_3D = self.optical_model.psf[..., None]

            # Perform the convolution using convolve
            result = torch.nn.functional.conv3d(self.last_filtered_interpolated_scene[None, None, ...], torch.flip(psf_3D, (0,1,2))[None, None, ...], padding = tuple((np.array(psf_3D.shape)-1)//2)).squeeze(0,1)
            result_panchro = torch.nn.functional.conv2d(self.panchro[None, None, ...], torch.flip(self.optical_model.psf, (0,1))[None, None, ...], padding = tuple((np.array(self.optical_model.psf.shape)-1)//2)).squeeze(0,1)

        else:
            print("No PSF or last measurement to apply PSF")
            result = self.last_filtered_interpolated_scene
            result_panchro = self.panchro

        self.last_filtered_interpolated_scene = result
        self.panchro = result_panchro

        return self.last_filtered_interpolated_scene