from simca.OpticalModel import OpticalModel
from simca.functions_acquisition import *
from simca.functions_patterns_generation import *
from simca.functions_scenes import *
from simca.functions_general_purpose import *
from scipy.signal import convolve



class CassiSystem():
    """Class that contains the cassi system main attributes and methods"""

    def __init__(self, system_config=None, system_config_path=None):

        """

        Args:
            system_config_path (str): path to the configs file
            system_config (dict): system configuration

        """

        self.set_up_system(system_config=system_config, system_config_path=system_config_path)

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

        self.optical_model = OpticalModel(self.system_config)

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

    def load_dataset(self, directory, dataset_name):
        """
        Loading the dataset and related attributes

        Args:
            directory (str): name of the directory containing the dataset
            dataset_name (str): dataset name

        Returns:
            list: a list containing the dataset (shape= R_dts x C_dts x W_dts), the corresponding wavelengths (shape= W_dts), the labeled dataset, the label names and the ignored labels
        """

        dataset, wavelengths_vec, dataset_labels, label_names, ignored_labels = get_dataset(directory, dataset_name)

        self.dataset = dataset
        self.dataset_labels = dataset_labels
        self.dataset_wavelengths = wavelengths_vec

        # additional attributes
        self.dataset_label_names = label_names
        self.dataset_ignored_labels = ignored_labels
        self.dataset_palette = palette_init(label_names)


        list_dataset_data = [self.dataset, self.dataset_labels, self.dataset_wavelengths, self.dataset_label_names,
                             self.dataset_ignored_labels, self.dataset_palette]

        return list_dataset_data

    def interpolate_dataset_along_wavelengths(self, new_wavelengths_sampling, chunk_size):
        """
        Interpolate the dataset cube along the wavelength axis to match the system sampling

        Args:
            new_wavelengths_sampling (numpy.ndarray): new wavelengths on which to interpolate the dataset (shape = W)
            chunk_size (int): chunk size for the multiprocessing

        Returns:
            numpy.ndarray : interpolated dataset cube along the wavelength axis (shape = R_dts x C_dts x W)

        """
        try:
            self.dataset
        except :
            raise ValueError("The dataset must be loaded first")

        if self.dataset_wavelengths[0] <= new_wavelengths_sampling[0] and self.dataset_wavelengths[-1] >= new_wavelengths_sampling[-1]:

            self.dataset_interpolated = interpolate_data_along_wavelength(self.dataset,self.dataset_wavelengths,new_wavelengths_sampling, chunk_size)
            return self.dataset_interpolated
        else:
            raise ValueError("The new wavelengths sampling must be inside the dataset wavelengths range")


    def generate_2D_pattern(self, config_pattern):
        """
        Generate the coded aperture 2D pattern based on the "pattern" configuration file

        Args:
            config_pattern (dict): coded-aperture pattern configuration

        Returns:
            numpy.ndarray: coded-aperture 2D pattern based on the configuration file (shape = H x L)
        """

        pattern_type = config_pattern['pattern']['type']

        if pattern_type == "random":
            pattern= generate_random_pattern((self.system_config["coded aperture"]["number of pixels along Y"],self.system_config["coded aperture"]["number of pixels along X"]),
                                        config_pattern['pattern']['ROM'])

        elif pattern_type == "slit":
            pattern= generate_slit_pattern((self.system_config["coded aperture"]["number of pixels along Y"],self.system_config["coded aperture"]["number of pixels along X"]),
                                      config_pattern['pattern']['slit position'],
                                      config_pattern['pattern']['slit width'])

        elif pattern_type == "blue-noise type 1":
            pattern= generate_blue_noise_type_1_pattern((self.system_config["coded aperture"]["number of pixels along Y"], self.system_config["coded aperture"]["number of pixels along X"]))

        elif pattern_type == "blue-noise type 2":
            pattern= generate_blue_noise_type_2_pattern((self.system_config["coded aperture"]["number of pixels along Y"], self.system_config["coded aperture"]["number of pixels along X"]))

        elif pattern_type == "custom h5 pattern":
            pattern= load_custom_pattern((self.system_config["coded aperture"]["number of pixels along Y"], self.system_config["coded aperture"]["number of pixels along X"]),
                                    config_pattern['pattern']['file path'])
        else:
            raise ValueError("patterntype is not supported for single patterngeneration, change it in the 'pattern.yml' config file")

        self.pattern= pattern

        return pattern

    def generate_multiple_patterns(self, config_pattern, number_of_patterns):
        """
        Generate a list of coded aperture patterns based on the "pattern" configuration file

        Args:
            config_pattern (dict): pattern configuration
            number_of_patterns (int): number of patterns to generate

        Returns:
            list: coded aperture patterns (numpy.ndarray) generated according to the configuration file
        """
        list_of_patterns = list()
        pattern_type = config_pattern['pattern']['type']

        if pattern_type == "random":
            for i in range(number_of_patterns):
                pattern= generate_random_pattern((self.system_config["coded aperture"]["number of pixels along Y"], self.system_config["coded aperture"]["number of pixels along X"]),config_pattern['pattern']['ROM'])
                list_of_patterns.append(pattern)

        elif pattern_type == "slit":
            # mmmmh you are weird, why would you want to do that ?
            for i in range(number_of_patterns):
                pattern= generate_slit_pattern((self.system_config["coded aperture"]["number of pixels along Y"], self.system_config["coded aperture"]["number of pixels along X"]),
                    config_pattern['pattern']['slit position'],
                    config_pattern['pattern']['slit width'])
                list_of_patterns.append(pattern)

        elif pattern_type == "LN-random":
            list_of_patterns = generate_ln_orthogonal_pattern(size=(self.system_config["coded aperture"]["number of pixels along Y"],
                                                                    self.system_config["coded aperture"]["number of pixels along X"]),
                                                            W=self.system_config["spectral range"]["number of spectral samples"],
                                                            N=number_of_patterns)

        elif pattern_type == "blue-noise type 1":

            for i in range(number_of_patterns):
                pattern= generate_blue_noise_type_1_pattern((self.system_config["coded aperture"]["number of pixels along Y"], self.system_config["coded aperture"]["number of pixels along X"]))
                list_of_patterns.append(pattern)

        elif pattern_type == "blue-noise type 2":

            for i in range(number_of_patterns):
                pattern= generate_blue_noise_type_2_pattern((self.system_config["coded aperture"]["number of pixels along Y"], self.system_config["coded aperture"]["number of pixels along X"]))
                list_of_patterns.append(pattern)

        elif pattern_type == "custom h5":
            list_of_patterns = load_custom_pattern((self.system_config["coded aperture"]["number of pixels along Y"], self.system_config["coded aperture"]["number of pixels along X"]),
                                                  config_pattern['pattern']['file path'])

        else:
            print("pattern type is not supported")
            list_of_patterns = None

        self.list_of_patterns = list_of_patterns

        return self.list_of_patterns



    def generate_filtering_cube(self):
        """
        Generate filtering cube : each slice of the cube is a propagated pattern interpolated on the detector grid

        Returns:
           numpy.ndarray: filtering cube generated according to the optical system & the pattern configuration (R x C x W)

        """

        self.filtering_cube = interpolate_data_on_grid_positions(data=self.pattern,
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

            self.filtering_cube = interpolate_data_on_grid_positions(data=self.list_of_patterns[idx],
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

        dataset = self.interpolate_dataset_along_wavelengths(self.optical_model.system_wavelengths, chunck_size)

        if dataset is None:
            return None
        dataset_labels = self.dataset_labels

        if self.system_config["system architecture"]["system type"] == "DD-CASSI":

            try:
                self.filtering_cube
            except:
                return print("Please generate filtering cube first")

            scene = match_dataset_to_instrument(dataset, self.filtering_cube)

            measurement_in_3D = generate_dd_measurement(scene, self.filtering_cube, chunck_size)

            self.last_filtered_interpolated_scene = measurement_in_3D
            self.interpolated_scene = scene

            if dataset_labels is not None:
                scene_labels = match_dataset_labels_to_instrument(dataset_labels, self.filtering_cube)
                self.scene_labels = scene_labels


        elif self.system_config["system architecture"]["system type"] == "SD-CASSI":

            X_coded_aper_coordinates_crop = crop_center(self.X_coded_aper_coordinates,dataset.shape[1], dataset.shape[0])
            Y_coded_aper_coordinates_crop = crop_center(self.Y_coded_aper_coordinates,dataset.shape[1], dataset.shape[0])

            scene = match_dataset_to_instrument(dataset, X_coded_aper_coordinates_crop)

            pattern_crop = crop_center(self.pattern, scene.shape[1], scene.shape[0])

            filtered_scene = scene * np.tile(pattern_crop[..., np.newaxis], (1, 1, scene.shape[2]))

            self.propagate_coded_aperture_grid(X_input_grid=X_coded_aper_coordinates_crop, Y_input_grid=Y_coded_aper_coordinates_crop)

            sd_measurement = interpolate_data_on_grid_positions(filtered_scene,
                                                                self.X_coordinates_propagated_coded_aperture,
                                                                self.Y_coordinates_propagated_coded_aperture,
                                                                self.X_detector_coordinates_grid,
                                                                self.Y_detector_coordinates_grid)

            self.last_filtered_interpolated_scene = sd_measurement
            self.interpolated_scene = scene

            if dataset_labels is not None:
                scene_labels = match_dataset_labels_to_instrument(dataset_labels, self.last_filtered_interpolated_scene)
                self.scene_labels = scene_labels

        self.panchro = np.sum(self.interpolated_scene, axis=2)

        if use_psf:
            self.apply_psf()
        else:
            print("No PSF was applied")

        # Calculate the other two arrays
        self.measurement = np.sum(self.last_filtered_interpolated_scene, axis=2)


        return self.measurement

    def multiple_image_acquisitions(self, use_psf=False, nb_of_filtering_cubes=1,chunck_size=50):
        """
        Run the acquisition process depending on the cassi system type

        Args:
            chunck_size (int): default block size for the dataset

        Returns:
             list: list of compressed measurements (list of numpy.ndarray of size R x C)
        """

        dataset = self.interpolate_dataset_along_wavelengths(self.optical_model.system_wavelengths, chunck_size)
        if dataset is None:
            return None
        dataset_labels = self.dataset_labels

        self.list_of_filtered_scenes = []

        if self.system_config["system architecture"]["system type"] == "DD-CASSI":
            try:
                self.list_of_filtering_cubes
            except:
                return print("Please generate list of filtering cubes first")

            scene = match_dataset_to_instrument(dataset, self.list_of_filtering_cubes[0])

            if dataset_labels is not None:
                scene_labels = match_dataset_labels_to_instrument(dataset_labels, self.filtering_cube)
                self.scene_labels = scene_labels

            self.interpolated_scene = scene

            for i in range(nb_of_filtering_cubes):

                filtered_scene = generate_dd_measurement(scene, self.list_of_filtering_cubes[i], chunck_size)
                self.list_of_filtered_scenes.append(filtered_scene)


        elif self.system_config["system architecture"]["system type"] == "SD-CASSI":

            X_coded_aper_coordinates_crop = crop_center(self.X_coded_aper_coordinates,dataset.shape[1], dataset.shape[0])
            Y_coded_aper_coordinates_crop = crop_center(self.Y_coded_aper_coordinates,dataset.shape[1], dataset.shape[0])


            scene = match_dataset_to_instrument(dataset, X_coded_aper_coordinates_crop)

            if dataset_labels is not None:
                scene_labels = match_dataset_labels_to_instrument(dataset_labels, self.filtering_cube)
                self.scene_labels = scene_labels

            self.interpolated_scene = scene

            for i in range(nb_of_filtering_cubes):

                mask_crop = crop_center(self.list_of_patterns[i], scene.shape[1], scene.shape[0])

                filtered_scene = scene * np.tile(mask_crop[..., np.newaxis], (1, 1, scene.shape[2]))

                self.propagate_coded_aperture_grid(X_input_grid=X_coded_aper_coordinates_crop, Y_input_grid=Y_coded_aper_coordinates_crop)

                sd_measurement_cube = interpolate_data_on_grid_positions(filtered_scene,
                                                                    self.X_coordinates_propagated_coded_aperture,
                                                                    self.Y_coordinates_propagated_coded_aperture,
                                                                    self.X_detector_coordinates_grid,
                                                                    self.Y_detector_coordinates_grid)
                self.list_of_filtered_scenes.append(sd_measurement_cube)

        self.panchro = np.sum(self.interpolated_scene, axis=2)

        if use_psf:
            self.apply_psf()
        else:
            print("No PSF was applied")

        # Calculate the other two arrays
        self.list_of_measurements = []
        for i in range(nb_of_filtering_cubes):
            self.list_of_measurements.append(np.sum(self.list_of_filtered_scenes[i], axis=2))

        return self.list_of_measurements



    def create_coordinates_grid(self, nb_of_pixels_along_x, nb_of_pixels_along_y, delta_x, delta_y):
        """
        Create a coordinates grid for a given number of samples along X and Y axis and a given pixel size

        Args:
            nb_of_pixels_along_x (int): number of samples along X axis
            nb_of_pixels_along_y (int): number of samples along Y axis
            delta_x (float): pixel size along X axis
            delta_y (float): pixel size along Y axis

        Returns:
            tuple: X coordinates grid (numpy.ndarray) and Y coordinates grid (numpy.ndarray)
        """
        x = np.arange(-(nb_of_pixels_along_x-1) * delta_x / 2, (nb_of_pixels_along_x+1) * delta_x / 2,delta_x)
        y = np.arange(-(nb_of_pixels_along_y-1) * delta_y / 2, (nb_of_pixels_along_y+1) * delta_y / 2, delta_y)


        # Create a two-dimensional grid of coordinates
        X_input_grid, Y_input_grid = np.meshgrid(x, y)

        return X_input_grid, Y_input_grid

    def propagate_coded_aperture_grid(self, X_input_grid=None, Y_input_grid=None,use_torch=False):
        """
        Propagate the coded_aperture pattern through one CASSI system

        Args:
            X_input_grid (numpy.ndarray): x coordinates grid
            Y_input_grid (numpy.ndarray): y coordinates grid

        Returns:
            tuple: propagated coded aperture x coordinates grid in the detector plane (3D numpy.ndarray), propagated coded aperture y coordinates grid in the detector plane (3D numpy.ndarray), 1D array of propagated coded aperture x coordinates (numpy.ndarray), 1D array of system wavelengths (numpy.ndarray)
        """

        if X_input_grid is None:
            X_input_grid = self.X_coded_aper_coordinates
        if Y_input_grid is None:
            Y_input_grid = self.Y_coded_aper_coordinates

        propagation_type = self.system_config["system architecture"]["propagation type"]

        if propagation_type == "simca" and use_torch == False:
            self.X_coordinates_propagated_coded_aperture, self.Y_coordinates_propagated_coded_aperture = self.optical_model.propagation_with_distorsions(X_input_grid, Y_input_grid)

        if propagation_type == "simca" and use_torch == True:
            self.X_coordinates_propagated_coded_aperture, self.Y_coordinates_propagated_coded_aperture = self.optical_model.propagation_with_distorsions_torch(X_input_grid, Y_input_grid)

        if propagation_type == "higher-order":
            self.X_coordinates_propagated_coded_aperture, self.Y_coordinates_propagated_coded_aperture = self.optical_model.propagation_with_no_distorsions(X_input_grid, Y_input_grid)

        self.optical_model.check_if_sampling_is_sufficiant()

        self.X_coordinates_propagated_coded_aperture = np.nan_to_num(self.X_coordinates_propagated_coded_aperture)
        self.Y_coordinates_propagated_coded_aperture = np.nan_to_num(self.Y_coordinates_propagated_coded_aperture)

        return self.X_coordinates_propagated_coded_aperture, self.Y_coordinates_propagated_coded_aperture, self.optical_model.system_wavelengths


    def apply_psf(self):
        """
        Apply the PSF to the last measurement

        Returns:
            numpy.ndarray: last measurement cube convolved with by PSF (shape= R x C x W). Each slice of the 3D filtered scene is convolved with the PSF
        """
        if (self.optical_model.psf is not None) and (self.last_filtered_interpolated_scene is not None):
            # Expand the dimensions of the 2D matrix to match the 3D matrix
            psf_3D = np.expand_dims(self.optical_model.psf, axis=-1)

            # Perform the convolution using convolve
            result = convolve(self.last_filtered_interpolated_scene, psf_3D, mode='same')
            result_panchro = convolve(self.panchro, self.optical_model.psf, mode='same')

        else:
            print("No PSF or last measurement to apply PSF")
            result = self.last_filtered_interpolated_scene
            result_panchro = self.panchro

        self.last_filtered_interpolated_scene = result
        self.panchro = result_panchro

        return self.last_filtered_interpolated_scene



    def save_acquisition(self, config_pattern, config_acquisition):
        """
        Save the all data related to an acquisition

        Args:
            config_pattern (dict): configuration dictionary related to pattern generation
            config_acquisition (dict): configuration dictionary related to acquisition parameters

        """

        self.result_directory = initialize_acquisitions_directory(config_acquisition)

        save_config_file("config_system",self.system_config,self.result_directory)
        save_config_file("config_pattern",config_pattern,self.result_directory)
        save_config_file("config_acquisition",config_acquisition,self.result_directory)
        save_data_in_hdf5("interpolated_scene",self.interpolated_scene, self.result_directory)
        try:
            save_data_in_hdf5("scene_labels",self.scene_labels,self.result_directory)
        except :
            pass
        save_data_in_hdf5("filtered_interpolated_scene",self.last_filtered_interpolated_scene, self.result_directory)
        save_data_in_hdf5("measurement",self.measurement,self.result_directory)
        save_data_in_hdf5("panchro",self.panchro,self.result_directory)
        save_data_in_hdf5("filtering_cube",self.filtering_cube,self.result_directory)
        save_data_in_hdf5("pattern",self.pattern,self.result_directory)
        save_data_in_hdf5("wavelengths",self.optical_model.system_wavelengths,self.result_directory)

        print("Acquisition saved in " + self.result_directory)
