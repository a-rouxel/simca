from cassi_systems.OpticalModel import OpticalModel
from cassi_systems.functions_acquisition import *
from cassi_systems.functions_masks_generation import *
from cassi_systems.functions_scenes import *
from cassi_systems.functions_general_purpose import *
from scipy.interpolate import griddata

import multiprocessing as mp
from multiprocessing import Pool
from scipy.signal import convolve



class CassiSystem():
    """Class that contains the cassi system main attributes and methods"""

    def __init__(self, system_config=None, system_config_path=None):
        """
        Load the system configuration file and initialize the grids for the DMD and the detector

        Args:
            system_config_path (str): path to the configs file
            system_config (dict): system configuration

        Initial Attributes:
            system_config (dict): system configuration
            X_dmd_coordinates_grid (numpy array): X grid coordinates of the center of the DMD pixels
            Y_dmd_coordinates_grid (numpy array): Y grid coordinates of the center of the DMD pixels
            X_detector_coordinates_grid (numpy array): X grid coordinates of the center of the detector pixels
            Y_detector_coordinates_grid (numpy array): Y grid coordinates of the center of the detector pixels
        """

        if system_config_path is not None:
            self.system_config = load_yaml_config(system_config_path)
        elif system_config is not None:
            self.system_config = system_config

        self.optical_model = OpticalModel(self.system_config)

        self.X_dmd_coordinates_grid, self.Y_dmd_coordinates_grid = self.create_coordinates_grid(
            self.system_config["SLM"]["number of pixels along X"],
            self.system_config["SLM"]["number of pixels along Y"],
            self.system_config["SLM"]["pixel size along X"],
            self.system_config["SLM"]["pixel size along Y"])

        self.X_detector_coordinates_grid, self.Y_detector_coordinates_grid = self.create_coordinates_grid(
            self.system_config["detector"]["number of pixels along X"],
            self.system_config["detector"]["number of pixels along Y"],
            self.system_config["detector"]["pixel size along X"],
            self.system_config["detector"]["pixel size along Y"])




    def load_dataset(self, directory, dataset_name):
        """Loading the dataset and related attributes

        Args:
            directory (str): name of the directory containing the dataset
            dataset_name (str): dataset name

        Returns:
            list_dataset_data (list) : a list containing the dataset, the ground truth, the list of wavelengths, the label values, the ignored labels, the rgb bands, the palette and the delta lambda
        """

        dataset, list_wavelengths, dataset_labels, label_names, ignored_labels = get_dataset(
            directory, dataset_name)

        self.dataset = dataset
        self.dataset_labels = dataset_labels
        self.dataset_wavelengths = list_wavelengths

        # additional attributes
        self.dataset_label_names = label_names
        self.dataset_ignored_labels = ignored_labels
        self.dataset_palette = palette_init(label_names)


        list_dataset_data = [self.dataset, self.dataset_labels, self.dataset_wavelengths, self.dataset_label_names,
                             self.dataset_ignored_labels, self.dataset_palette]

        return list_dataset_data

    def update_config(self, new_config):
        """
        Update the system configuration and recalculate the DMD and detector grids coordinates

        Args:
            new_config (dict): new system configuration

        Returns:
            configs (dict) : updated system configuration
        """

        self.system_config = new_config

        self.X_dmd_coordinates_grid, self.Y_dmd_coordinates_grid = self.create_coordinates_grid(
            self.system_config["SLM"]["number of pixels along X"],
            self.system_config["SLM"]["number of pixels along Y"],
            self.system_config["SLM"]["pixel size along X"],
            self.system_config["SLM"]["pixel size along Y"])

        self.X_detector_coordinates_grid, self.Y_detector_coordinates_grid = self.create_coordinates_grid(
            self.system_config["detector"]["number of pixels along X"],
            self.system_config["detector"]["number of pixels along Y"],
            self.system_config["detector"]["pixel size along X"],
            self.system_config["detector"]["pixel size along Y"])

        self.optical_model.update_config(self.system_config)

        return self.system_config

    def interpolate_dataset_along_wavelengths(self, new_wavelengths_sampling, chunk_size):
        """
        Interpolate the dataset cube along the wavelength axis to match the system sampling

        Args:
            new_wavelengths_sampling (numpy array): system wavelengths sampling
            chunk_size (int): chunk size for the multiprocessing

        Returns:
            dataset_interpolated (numpy array): interpolated dataset cube along the wavelength axis

        """
        try:
            self.dataset
        except :
            raise ValueError("The dataset must be loaded first")

        if self.dataset_wavelengths[0] <= new_wavelengths_sampling[0] and self.dataset_wavelengths[-1] >= new_wavelengths_sampling[-1]:

            self.dataset_interpolated = interpolate_dataset_cube_along_wavelength(self.dataset,
                                                                                  self.dataset_wavelengths,
                                                                                  new_wavelengths_sampling, chunk_size)
            return self.dataset_interpolated
        else:
            raise ValueError("The new wavelengths sampling must be inside the dataset wavelengths range")


    def generate_2D_mask(self, config_mask_and_filtering):
        """
        Generate the 2D DMD mask based on the "filtering" configuration file

        Args:
            config_mask_and_filtering (dict): masks and filtering configuration

        Returns:
            mask (numpy array): 2D DMD mask based on the configuration file
        """

        mask_type = config_mask_and_filtering['mask']['type']

        if mask_type == "random":
            mask = generate_random_mask((self.system_config["SLM"]["number of pixels along Y"],self.system_config["SLM"]["number of pixels along X"]),
                                        config_mask_and_filtering['mask']['ROM'])

        elif mask_type == "slit":
            mask = generate_slit_mask((self.system_config["SLM"]["number of pixels along Y"],self.system_config["SLM"]["number of pixels along X"]),
                                      config_mask_and_filtering['mask']['slit position'],
                                      config_mask_and_filtering['mask']['slit width'])

        elif mask_type == "blue-noise type 1":
            mask = generate_blue_noise_type_1_mask((self.system_config["SLM"]["number of pixels along Y"], self.system_config["SLM"]["number of pixels along X"]))

        elif mask_type == "blue-noise type 2":
            mask = generate_blue_noise_type_2_mask((self.system_config["SLM"]["number of pixels along Y"], self.system_config["SLM"]["number of pixels along X"]))

        elif mask_type == "custom h5 mask":
            mask = load_custom_mask((self.system_config["SLM"]["number of pixels along Y"], self.system_config["SLM"]["number of pixels along X"]),
                                    config_mask_and_filtering['mask']['file path'])
        else:
            print("Mask type is not supported")
            mask = None

        self.mask = mask
        return mask

    def generate_multiple_SLM_masks(self, config_mask_and_filtering, number_of_masks):
        """
        Generate a list of SLM masks based on the "filtering" configuration file

        Args:
            config_mask_and_filtering (dict): masks and filtering configuration
            number_of_masks (int): number of masks to generate

        Returns:
            list_of_SLM_masks (list): list of SLM mask (numpy arrays) generated according to the configuration file
        """
        list_of_SLM_masks = list()
        mask_type = config_mask_and_filtering['mask']['type']

        if mask_type == "random":
            for i in range(number_of_masks):
                mask = generate_random_mask((self.system_config["SLM"]["number of pixels along Y"], self.system_config["SLM"]["number of pixels along X"]),config_mask_and_filtering['mask']['ROM'])
                list_of_SLM_masks.append(mask)

        elif mask_type == "slit":
            # mmmmh you are weird, why would you want to do that ?
            for i in range(number_of_masks):
                mask = generate_slit_mask((self.system_config["SLM"]["number of pixels along Y"], self.system_config["SLM"]["number of pixels along X"]),
                    config_mask_and_filtering['mask']['slit position'],
                    config_mask_and_filtering['mask']['slit width'])
                list_of_SLM_masks.append(mask)

        elif mask_type == "LN-random":
            list_of_SLM_masks = generate_ln_orthogonal_mask(size=(self.system_config["SLM"]["number of pixels along Y"],self.system_config["SLM"]["number of pixels along X"]),
                                                            W=self.system_config["spectral range"]["number of spectral samples"],
                                                            N=number_of_masks)

        elif mask_type == "blue-noise type 1":

            for i in range(number_of_masks):
                mask = generate_blue_noise_type_1_mask((self.system_config["SLM"]["number of pixels along Y"], self.system_config["SLM"]["number of pixels along X"]))
                list_of_SLM_masks.append(mask)

        elif mask_type == "blue-noise type 2":

            for i in range(number_of_masks):
                mask = generate_blue_noise_type_2_mask((self.system_config["SLM"]["number of pixels along Y"], self.system_config["SLM"]["number of pixels along X"]))
                list_of_SLM_masks.append(mask)

        elif mask_type == "custom h5":
            list_of_SLM_masks = load_custom_mask((self.system_config["SLM"]["number of pixels along Y"], self.system_config["SLM"]["number of pixels along X"]),
                                                  config_mask_and_filtering['mask']['file path'])

        else:
            print("Mask type is not supported")
            list_of_SLM_masks = None

        self.list_of_SLM_masks = list_of_SLM_masks

        return self.list_of_SLM_masks

    def generate_filtering_cube(self):
        """
        Generate filtering cube, each slice is a propagated mask interpolated on the detector grid

        Returns:
            filtering_cube (numpy array): 3D filtering cube generated according to the filtering configuration

        """

        self.filtering_cube = np.zeros((self.system_config["detector"]["number of pixels along Y"],
                                        self.system_config["detector"]["number of pixels along X"],
                                        self.system_config["spectral range"]["number of spectral samples"]))

        with Pool(mp.cpu_count()) as p:

            if self.system_config["system architecture"]["propagation type"] == "simca":
                worker = worker_unstructured
            elif self.system_config["system architecture"]["propagation type"] == "higher-order":
                worker = worker_regulargrid

            tasks = [(self.list_X_propagated_mask, self.list_Y_propagated_mask, self.mask,
                      self.X_detector_coordinates_grid, self.Y_detector_coordinates_grid, i)
                     for i in range(len(self.optical_model.system_wavelengths))]
            for index, zi in tqdm(enumerate(p.imap(worker, tasks)), total=len(self.optical_model.system_wavelengths),
                                  desc='Processing tasks'):
                self.filtering_cube[:, :, index] = zi

        self.filtering_cube = np.nan_to_num(self.filtering_cube)

        return self.filtering_cube

    def generate_multiple_filtering_cubes(self, number_of_masks):
        """
        Generate multiple filtering cubes, each cube corresponds to a mask, and for each mask, each slice is a propagated mask interpolated on the detector grid

        Args:
            number_of_masks (int): number of masks to generate
        Returns:
            list_of_filtering_cubes (list): list of 3D filtering cubes generated according to the filtering configuration and

        """
        self.list_of_filtering_cubes = []

        for idx in range(number_of_masks):


            self.filtering_cube = np.zeros((self.system_config["detector"]["number of pixels along Y"],
                                            self.system_config["detector"]["number of pixels along X"],
                                            self.system_config["spectral range"]["number of spectral samples"]))

            with Pool(mp.cpu_count()) as p:

                if self.system_config["system architecture"]["propagation type"] == "simca":
                    worker = worker_unstructured
                elif self.system_config["system architecture"]["propagation type"] == "higher-order":
                    worker = worker_regulargrid


                tasks = [(self.list_X_propagated_mask, self.list_Y_propagated_mask, self.list_of_SLM_masks[idx],
                          self.X_detector_coordinates_grid, self.Y_detector_coordinates_grid, i)
                         for i in range(len(self.optical_model.system_wavelengths))]
                for index, zi in tqdm(enumerate(p.imap(worker, tasks)), total=len(self.optical_model.system_wavelengths),
                                      desc='Processing tasks'):
                    self.filtering_cube[:, :, index] = zi

                self.filtering_cube = np.nan_to_num(self.filtering_cube)

                self.list_of_filtering_cubes.append(self.filtering_cube)

        return self.list_of_filtering_cubes

    def image_acquisition(self, use_psf=False, chunck_size=50):
        """
        Run the acquisition process depending on the cassi system type

        Args:
            chunck_size (int): default block size for the dataset

        Returns:
            last_filtered_interpolated_scene (numpy array): filtered scene cube
            interpolated_scene (numpy array): interpolated scene cube
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

            scene = match_scene_to_instrument(dataset, self.filtering_cube)

            measurement_in_3D = generate_dd_measurement(scene, self.filtering_cube, chunck_size)

            self.last_filtered_interpolated_scene = measurement_in_3D
            self.interpolated_scene = scene

            if dataset_labels is not None:
                scene_labels = match_scene_labels_to_instrument(dataset_labels, self.filtering_cube)
                self.scene_labels = scene_labels


        elif self.system_config["system architecture"]["system type"] == "SD-CASSI":

            X_dmd_coordinates_grid_crop = crop_center(self.X_dmd_coordinates_grid,dataset.shape[1], dataset.shape[0])
            Y_dmd_coordinates_grid_crop = crop_center(self.Y_dmd_coordinates_grid,dataset.shape[1], dataset.shape[0])

            scene = match_scene_to_instrument(dataset, X_dmd_coordinates_grid_crop)

            mask_crop = crop_center(self.mask, scene.shape[1], scene.shape[0])

            filtered_scene = scene * np.tile(mask_crop[..., np.newaxis], (1, 1, scene.shape[2]))

            self.propagate_mask_grid(X_input_grid=X_dmd_coordinates_grid_crop, Y_input_grid=Y_dmd_coordinates_grid_crop)

            sd_measurement = self.generate_sd_measurement_cube(filtered_scene)

            self.last_filtered_interpolated_scene = sd_measurement
            self.interpolated_scene = scene

            if dataset_labels is not None:
                scene_labels = match_scene_labels_to_instrument(dataset_labels, self.last_filtered_interpolated_scene)
                self.scene_labels = scene_labels

        if use_psf:
            self.apply_psf()
        else:
            print("No PSF was applied")

        # Calculate the other two arrays
        self.measurement = np.sum(self.last_filtered_interpolated_scene, axis=2)
        self.panchro = np.sum(self.interpolated_scene, axis=2)

        return self.last_filtered_interpolated_scene, self.interpolated_scene

    def multiple_image_acquisitions(self, use_psf=False, nb_of_filtering_cubes=1,chunck_size=50):
        """
        Run the acquisition process depending on the cassi system type

        Args:
            chunck_size (int): default block size for the dataset

        Returns:
            last_filtered_interpolated_scene (numpy array): filtered scene cube
            interpolated_scene (numpy array): interpolated scene cube
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

            scene = match_scene_to_instrument(dataset, self.list_of_filtering_cubes[0])

            if dataset_labels is not None:
                scene_labels = match_scene_labels_to_instrument(dataset_labels, self.filtering_cube)
                self.scene_labels = scene_labels

            self.interpolated_scene = scene

            for i in range(nb_of_filtering_cubes):

                filtered_scene = generate_dd_measurement(scene, self.list_of_filtering_cubes[i], chunck_size)
                self.list_of_filtered_scenes.append(filtered_scene)


        elif self.system_config["system architecture"]["system type"] == "SD-CASSI":

            X_dmd_coordinates_grid_crop = crop_center(self.X_dmd_coordinates_grid,dataset.shape[1], dataset.shape[0])
            Y_dmd_coordinates_grid_crop = crop_center(self.Y_dmd_coordinates_grid,dataset.shape[1], dataset.shape[0])


            scene = match_scene_to_instrument(dataset, X_dmd_coordinates_grid_crop)

            if dataset_labels is not None:
                scene_labels = match_scene_labels_to_instrument(dataset_labels, self.filtering_cube)
                self.scene_labels = scene_labels

            self.interpolated_scene = scene

            for i in range(nb_of_filtering_cubes):

                mask_crop = crop_center(self.list_of_SLM_masks[i], scene.shape[1], scene.shape[0])

                filtered_scene = scene * np.tile(mask_crop[..., np.newaxis], (1, 1, scene.shape[2]))

                self.propagate_mask_grid(X_input_grid=X_dmd_coordinates_grid_crop, Y_input_grid=Y_dmd_coordinates_grid_crop)

                filtered_and_propagated_scene = self.generate_sd_measurement_cube(filtered_scene)
                self.list_of_filtered_scenes.append(filtered_and_propagated_scene)


        if use_psf:
            self.apply_psf()
        else:
            print("No PSF was applied")

        # Calculate the other two arrays
        self.list_of_measurements = []
        for i in range(nb_of_filtering_cubes):
            self.list_of_measurements.append(np.sum(self.list_of_filtered_scenes[i], axis=2))

        self.panchro = np.sum(self.interpolated_scene, axis=2)

        return self.list_of_filtered_scenes, self.interpolated_scene

    def generate_sd_measurement_cube(self, scene):
        """
        Generate SD measurement cube from the mask and the scene.
        For Single Disperser CASSI systems, the scene is filtered then propagated in the detector plane.

        Args:
            scene (numpy array): scene cube

        Returns:
            measurement_sd (numpy array): SD measurement cube
        """

        X_detector_coordinates_grid = self.X_detector_coordinates_grid
        Y_detector_coordinates_grid = self.Y_detector_coordinates_grid
        list_X_propagated_masks = self.list_X_propagated_mask
        list_Y_propagated_masks = self.list_Y_propagated_mask
        scene = scene

        print("--- Generating SD measurement cube ---- ")

        self.measurement_sd = np.zeros((self.system_config["detector"]["number of pixels along Y"],
                                        self.system_config["detector"]["number of pixels along X"],
                                        self.system_config["spectral range"]["number of spectral samples"]))

        if self.system_config["system architecture"]["propagation type"] == "simca":
            worker = worker_unstructured
        elif self.system_config["system architecture"]["propagation type"] == "higher-order":
            worker = worker_regulargrid
        else:
            return None

        with Pool(mp.cpu_count()) as p:
            tasks = [(list_X_propagated_masks, list_Y_propagated_masks, scene[:, :, i], X_detector_coordinates_grid,
                      Y_detector_coordinates_grid, i)
                     for i in range(len(self.optical_model.system_wavelengths))]
            for index, zi in tqdm(enumerate(p.imap(worker, tasks)), total=len(self.optical_model.system_wavelengths),
                                  desc='Processing tasks'):
                self.measurement_sd[:, :, index] = zi

        self.measurement_sd = np.nan_to_num(self.measurement_sd)
        return self.measurement_sd

    def create_coordinates_grid(self, nb_of_samples_along_x, nb_of_samples_along_y, delta_x, delta_y):
        """
        Create a coordinates grid for a given number of samples along x and y axis and a given pixel size

        Args:
            nb_of_samples_along_x (int): number of samples along x axis
            nb_of_samples_along_y (int): number of samples along y axis
            delta_x (float): pixel size along x axis
            delta_y (float): pixel size along y axis

        Returns:
            X_input_grid (numpy array): x coordinates grid
            Y_input_grid (numpy array): y coordinates grid
        """
        x = np.arange(-(nb_of_samples_along_x-1) * delta_x / 2, (nb_of_samples_along_x+1) * delta_x / 2,delta_x)
        y = np.arange(-(nb_of_samples_along_y-1) * delta_y / 2, (nb_of_samples_along_y+1) * delta_y / 2, delta_y)


        # Create a two-dimensional grid of coordinates
        X_input_grid, Y_input_grid = np.meshgrid(x, y)

        return X_input_grid, Y_input_grid

    def propagate_mask_grid(self, X_input_grid=None, Y_input_grid=None):
        """
        Propagate the SLM mask through one CASSI system

        Args:
            X_input_grid (numpy array): x coordinates grid
            Y_input_grid (numpy array): y coordinates grid

        Returns:
            list_X_propagated_mask (list): list of the X coordinates of the propagated masks
            list_Y_propagated_mask (list): list of the Y coordinates of the propagated masks
        """

        if X_input_grid is None:
            X_input_grid = self.X_dmd_coordinates_grid
        if Y_input_grid is None:
            Y_input_grid = self.Y_dmd_coordinates_grid

        propagation_type = self.system_config["system architecture"]["propagation type"]

        if propagation_type == "simca":
            self.list_X_propagated_mask, self.list_Y_propagated_mask = self.optical_model.propagation_with_distorsions(X_input_grid, Y_input_grid)

        if propagation_type == "higher-order":
            self.list_X_propagated_mask, self.list_Y_propagated_mask = self.optical_model.propagation_with_no_distorsions(X_input_grid, Y_input_grid)

        self.optical_model.check_if_sampling_is_sufficiant()

        return self.list_X_propagated_mask, self.list_Y_propagated_mask, self.optical_model.system_wavelengths


    def apply_psf(self):
        """
        Apply the PSF to the last measurement

        Returns:
            last_filtered_interpolated_scene (numpy array): last measurement convolved with by PSF. Each slice of the 3D filtered scene is convolved with the PSF
        """
        if (self.optical_model.psf is not None) and (self.last_filtered_interpolated_scene is not None):
            # Expand the dimensions of the 2D matrix to match the 3D matrix
            psf_3D = np.expand_dims(self.optical_model.psf, axis=-1)

            # Perform the convolution using convolve
            result = convolve(self.last_filtered_interpolated_scene, psf_3D, mode='same')

        else:
            print("No PSF or last measurement to apply PSF")
            result = self.last_filtered_interpolated_scene

        self.last_filtered_interpolated_scene = result

        return self.last_filtered_interpolated_scene



    def save_acquisition(self, config_mask_and_filtering, config_acquisition):
        """
        Save the all data related to an acquisition

        Args:
            config_mask_and_filtering (dict): configuration dictionary for the mask and filtering parameters
            config_acquisition (dict): configuration dictionary for the acquisition parameters

        Returns:
        """

        self.result_directory = initialize_acquisitions_directory(config_acquisition)

        save_config_file("config_system",self.system_config,self.result_directory)
        save_config_file("config_mask_and_filtering",config_mask_and_filtering,self.result_directory)
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
        save_data_in_hdf5("mask",self.mask,self.result_directory)
        save_data_in_hdf5("wavelengths",self.optical_model.system_wavelengths,self.result_directory)

        print("Acquisition saved in " + self.result_directory)


def worker_unstructured(args):
    """
    Process to parallellize the unstructured griddata interpolation between the propagated grid (mask and the detector grid

    Args:
        list_X_propagated_masks (list): list of arrays, each element is a 2D array of X coordinates corresponding to a system wavelength
        list_Y_propagated_masks (list): list of arrays, each element is a 2D array of Y coordinates corresponding to a system wavelength
        mask (numpy array): 2D array of the mask values to be interpolated
        X_detector_coordinates_grid (numpy array): 2D array of the X coordinates of the detector grid
        Y_detector_coordinates_grid (numpy array): 2D array of the Y coordinates of the detector grid
        wavelength_index (int): index of the system wavelength to be processed

    Returns:
        interpolated_mask (numpy array): 2D array of the interpolated mask
    """
    list_X_propagated_masks, list_Y_propagated_masks, mask, X_detector_coordinates_grid, Y_detector_coordinates_grid, wavelength_index = args

    list_X_propagated_masks = np.nan_to_num(list_X_propagated_masks)
    interpolated_mask = griddata((list_X_propagated_masks[wavelength_index][:, :].flatten(),
                                  list_Y_propagated_masks[wavelength_index][:, :].flatten()),
                                 mask.flatten(),
                                 (X_detector_coordinates_grid, Y_detector_coordinates_grid),
                                 method='linear')
    return interpolated_mask

# Currently,  regulargrid method == unstructured method, However we could do faster for regular grid interpolation
def worker_regulargrid(args):
    """
    Process to parallellize the structured griddata interpolation between the propagated grid (mask and the detector grid
    For now it is identical to the unstructured method but it could be faster ...

    Args:
        list_X_propagated_masks (list): list of arrays, each element is a 2D array of X coordinates corresponding to a system wavelength
        list_Y_propagated_masks (list): list of arrays, each element is a 2D array of Y coordinates corresponding to a system wavelength
        mask (numpy array): 2D array of the mask values to be interpolated
        X_detector_coordinates_grid (numpy array): 2D array of the X coordinates of the detector grid
        Y_detector_coordinates_grid (numpy array): 2D array of the Y coordinates of the detector grid
        wavelength_index (int): index of the system wavelength to be processed

    Returns:
        interpolated_mask (numpy array): 2D array of the interpolated mask
    """
    list_X_propagated_masks, list_Y_propagated_masks, mask, X_detector_coordinates_grid, Y_detector_coordinates_grid, wavelength_index = args

    list_X_propagated_masks = np.nan_to_num(list_X_propagated_masks)

    interpolated_mask = griddata((list_X_propagated_masks[wavelength_index][:, :].flatten(),
                                  list_Y_propagated_masks[wavelength_index][:, :].flatten()),
                                 mask.flatten(),
                                 (X_detector_coordinates_grid, Y_detector_coordinates_grid),
                                 method='linear')
    return interpolated_mask