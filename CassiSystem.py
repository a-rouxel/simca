from utils.functions_retropropagating import *
from scipy.interpolate import griddata
import multiprocessing as mp
from multiprocessing import Pool
from scipy import fftpack
from utils.scenes_helper import *
from utils.functions_acquisition import *
from scipy.signal import convolve

class CassiSystem():
    """Class that contains the optical system main attributes and methods"""

    def __init__(self,system_config_path):
        """
        Load the system configuration file and initialize the grids for the DMD and the detector

        Args:
            system_config_path:

        Initial Attributes:
            system_config (dict): system configuration
            X_dmd_coordinates_grid (numpy array): X grid coordinates of the center of the DMD pixels
            Y_dmd_coordinates_grid (numpy array): Y grid coordinates of the center of the DMD pixels
            X_detector_coordinates_grid (numpy array): X grid coordinates of the center of the detector pixels
            Y_detector_coordinates_grid (numpy array): Y grid coordinates of the center of the detector pixels
        """

        self.system_config = load_yaml_config(system_config_path)


        self.X_dmd_coordinates_grid, self.Y_dmd_coordinates_grid = self.create_coordinates_grid(self.system_config["SLM"]["sampling across X"],
                                                                        self.system_config["SLM"]["sampling across Y"],
                                                                        self.system_config["SLM"]["delta X"],
                                                                        self.system_config["SLM"]["delta Y"])


        self.X_detector_coordinates_grid, self.Y_detector_coordinates_grid = self.create_coordinates_grid(self.system_config["detector"]["sampling across X"],
                                                                        self.system_config["detector"]["sampling across Y"],
                                                                        self.system_config["detector"]["delta X"],
                                                                        self.system_config["detector"]["delta Y"])

    def load_dataset(self,directory,dataset_name):
        """Loading the dataset

        Args:
            directory (str): name of the directory containing the dataset
            dataset_name (str): dataset name
        
        Returns:
            list: a list containing the dataset, the ground truth, the list of wavelengths, the label values, the ignored labels, the rgb bands, the palette and the delta lambda
        """


        img, gt, list_wavelengths, label_values, ignored_labels, rgb_bands, palette, delta_lambda = get_dataset(directory,dataset_name)
        self.dataset = img
        self.dataset_gt = gt
        self.list_dataset_wavelengths = list_wavelengths
        self.dataset_label_values = label_values
        self.dataset_ignored_labels = ignored_labels
        self.dataset_rgb_bands = rgb_bands
        self.dataset_palette = palette
        self.dataset_delta_lambda = delta_lambda

        self.dataset_palette = palette_init(label_values, palette)

        return [self.dataset, self.dataset_gt, self.list_dataset_wavelengths, self.dataset_label_values, self.dataset_ignored_labels, self.dataset_rgb_bands, self.dataset_palette, self.dataset_delta_lambda,self.dataset_palette]

    def update_config(self,new_config):
        """
        Update the system configuration and recalculate the DMD and detector grids coordinates
        Args:
            new_config (dict): new system configuration

        Returns:

        """

        self.system_config = new_config

        self.X_dmd_coordinates_grid, self.Y_dmd_coordinates_grid = self.create_coordinates_grid(self.system_config["SLM"]["sampling across X"],
                                                                        self.system_config["SLM"]["sampling across Y"],
                                                                        self.system_config["SLM"]["delta X"],
                                                                        self.system_config["SLM"]["delta Y"])


        self.X_detector_coordinates_grid, self.Y_detector_coordinates_grid = self.create_coordinates_grid(self.system_config["detector"]["sampling across X"],
                                                                        self.system_config["detector"]["sampling across Y"],
                                                                        self.system_config["detector"]["delta X"],
                                                                        self.system_config["detector"]["delta Y"])

    def interpolate_dataset(self,new_sampling,chunk_size):
        """
        Interpolate the dataset to a new sampling
        Args:
            new_sampling:
            chunk_size:

        Returns:
            dataset_interpolated (numpy array): interpolated dataset

        """
        self.dataset_interpolated = interpolate_dataset_cube_along_wavelength(self.dataset, self.list_dataset_wavelengths, new_sampling,chunk_size)
        return self.dataset_interpolated


    def generate_2D_mask(self,config_filtering):
        """

        Args:
            config_filtering:

        Returns:
            mask (numpy array): 2D DMD mask based on the configuration file
        """

        mask_type = config_filtering['mask']['type']

        if self.system_config["SLM"]["sampling across Y"] % 2 == 0:
            self.system_config["SLM"]["sampling across Y"] += 1
        if self.system_config["SLM"]["sampling across X"] % 2 == 0:
            self.system_config["SLM"]["sampling across X"] += 1

        if mask_type == "random":
            self.mask = np.random.randint(0, 2, (self.system_config["SLM"]["sampling across Y"],
                                                 self.system_config["SLM"]["sampling across X"]))
        elif mask_type == "slit":
            slit_position = config_filtering['mask']['slit position']
            slit_width = config_filtering['mask']['slit width']

            self.mask = np.zeros((self.system_config["SLM"]["sampling across Y"],
                                  self.system_config["SLM"]["sampling across X"]))

            slit_position = int(self.system_config["SLM"]["sampling across X"] / 2) + slit_position

            self.mask[:,slit_position-slit_width//2:slit_position+slit_width] = 1

        elif mask_type == "blue":
            size = (self.system_config["SLM"]["sampling across Y"], self.system_config["SLM"]["sampling across X"])
            self.mask = self.generate_blue_noise(size)

        elif mask_type == "custom h5 mask":
            mask_path = config_filtering['mask']['file path']
            if mask_path is None:
                raise ValueError("Please provide h5 file path for custom mask.")
            else:
                with h5py.File(mask_path, 'r') as f:
                    mask = f['mask'][:]

                slm_sampling_y = self.system_config["SLM"]["sampling across Y"]
                slm_sampling_x = self.system_config["SLM"]["sampling across X"]

                if mask.shape[0] != slm_sampling_y or mask.shape[1] != slm_sampling_x:
                    # Find center point of the mask
                    center_y, center_x = mask.shape[0] // 2, mask.shape[1] // 2

                    # Determine starting and ending indices for the crop
                    start_y = center_y - slm_sampling_y // 2
                    end_y = start_y + slm_sampling_y
                    start_x = center_x - slm_sampling_x // 2
                    end_x = start_x + slm_sampling_x

                    # Crop the mask
                    mask = mask[start_y:end_y, start_x:end_x]

                    # Confirm the mask is the correct shape
                    if mask.shape[0] != slm_sampling_y or mask.shape[1] != slm_sampling_x:
                        raise ValueError("Error cropping the mask, its shape does not match the SLM sampling.")

                self.mask = mask

        return self.mask

    @staticmethod
    def generate_blue_noise(size):
        """
        Generate blue noise (high frequency pseudo-random) type mask
        Args:
            size:

        Returns:
            binary mask (numpy array): binary blue noise type mask

        """
        shape = (size[0], size[1])
        N = shape[0] * shape[1]
        rng = np.random.default_rng()
        noise = rng.standard_normal(N)
        noise = np.reshape(noise, shape)

        f_x = fftpack.fftfreq(shape[1])
        f_y = fftpack.fftfreq(shape[0])
        f_x_shift = fftpack.fftshift(f_x)
        f_y_shift = fftpack.fftshift(f_y)
        f_matrix = np.sqrt(f_x_shift[None, :] ** 2 + f_y_shift[:, None] ** 2)

        spectrum = fftpack.fftshift(fftpack.fft2(noise))
        filtered_spectrum = spectrum * f_matrix
        filtered_noise = fftpack.ifft2(fftpack.ifftshift(filtered_spectrum)).real

        # Make the mask binary
        threshold = np.median(filtered_noise)
        binary_mask = np.where(filtered_noise > threshold, 1, 0)

        return binary_mask

    def generate_filtering_cube(self):
        """
        Generate filtering cube, each slice is a propagated mask interpolated on the detector grid

        Returns:
            filtering_cube (numpy array): 3D filtering cube

        """

        if self.system_config["detector"]["sampling across Y"] %2 ==0:
            self.system_config["detector"]["sampling across Y"] +=1
        if self.system_config["detector"]["sampling across X"] %2 ==0:
            self.system_config["detector"]["sampling across X"] +=1

        self.filtering_cube = np.zeros((self.system_config["detector"]["sampling across Y"],
                                        self.system_config["detector"]["sampling across X"],
                                        self.system_config["spectral range"]["number of spectral samples"]))

        with Pool(mp.cpu_count()) as p:
            tasks = [(self.list_X_propagated_mask, self.list_Y_propagated_mask, self.mask, self.X_detector_coordinates_grid, self.Y_detector_coordinates_grid, i)
                     for i in range(len(self.list_wavelengths))]
            for index, zi in tqdm(enumerate(p.imap(worker, tasks)), total=len(self.list_wavelengths), desc='Processing tasks'):
                self.filtering_cube[:, :, index] = zi


        return self.filtering_cube


    def image_acquisition(self,use_psf=False,chunck_size=50):
        """
        Contains the acquisition process depending on the cassi system type
        Args:
            chunck_size (int): default block size for the dataset

        Returns:

        """


        dataset = self.interpolate_dataset(self.list_wavelengths, chunck_size)

        if self.system_config["system architecture"]["system type"] == "DD-CASSI":

            try:
                self.filtering_cube
            except:
                return "Please generate filtering cube first"

            scene = match_scene_to_instrument(dataset, self.filtering_cube)
            measurement_in_3D = generate_dd_measurement(scene, self.filtering_cube, chunck_size)

            self.last_measurement_3D = measurement_in_3D
            self.interpolated_scene = scene


        elif self.system_config["system architecture"]["system type"] == "SD-CASSI":

            X_dmd_coordinates_grid_crop, Y_dmd_coordinates_grid_crop = crop_center(self.X_dmd_coordinates_grid, self.Y_dmd_coordinates_grid,
                                                           dataset.shape[1], dataset.shape[0])

            scene = match_scene_to_instrument(dataset, X_dmd_coordinates_grid_crop)

            mask_crop, mask_crop = crop_center(self.mask, self.mask, scene.shape[1], scene.shape[0])

            filtered_scene = scene * np.tile(mask_crop[..., np.newaxis], (1, 1, scene.shape[2]))

            self.propagate_mask_grid(X_input_grid=X_dmd_coordinates_grid_crop,Y_input_grid=Y_dmd_coordinates_grid_crop)

            sd_measurement = self.generate_sd_measurement_cube(filtered_scene)

            self.last_measurement_3D = sd_measurement
            self.interpolated_scene = scene

        if use_psf:
            self.apply_psf()
        else:
            print("No PSF was applied")

        return self.last_measurement_3D, self.interpolated_scene

    def generate_sd_measurement_cube(self,scene):


        X_detector_coordinates_grid = self.X_detector_coordinates_grid
        Y_detector_coordinates_grid = self.Y_detector_coordinates_grid
        list_X_propagated_masks = self.list_X_propagated_mask
        list_Y_propagated_masks = self.list_Y_propagated_mask
        scene =  scene



        print("--- Generating SD measurement cube ---- ")
        if self.system_config["detector"]["sampling across Y"] %2 ==0:
            self.system_config["detector"]["sampling across Y"] +=1
        if self.system_config["detector"]["sampling across X"] %2 ==0:
            self.system_config["detector"]["sampling across X"] +=1

        self.measurement_sd = np.zeros((self.system_config["detector"]["sampling across Y"],
                                        self.system_config["detector"]["sampling across X"],
                                        self.system_config["spectral range"]["number of spectral samples"]))

        wavelengths = np.linspace(self.system_config["spectral range"]["wavelength min"],
                                  self.system_config["spectral range"]["wavelength max"],
                                  self.system_config['spectral range']["number of spectral samples"])
        with Pool(mp.cpu_count()) as p:
            tasks = [(list_X_propagated_masks, list_Y_propagated_masks, scene[:,:,i], X_detector_coordinates_grid, Y_detector_coordinates_grid, i)
                     for i in range(len(wavelengths))]
            for index, zi in tqdm(enumerate(p.imap(worker, tasks)), total=len(wavelengths), desc='Processing tasks'):
                self.measurement_sd[:, :, index] = zi

        self.list_wavelengths = wavelengths

        return self.measurement_sd



    def create_coordinates_grid(self,nb_of_samples_along_x, nb_of_samples_along_y, delta_x, delta_y):

            if nb_of_samples_along_x % 2 == 0:
                nb_of_samples_along_x += 1
                logging.warning("Number of grid samples along X is even. It has been increased by 1 to be odd.")
            if nb_of_samples_along_y % 2 == 0:
                nb_of_samples_along_y += 1
                logging.warning("Number of grid samples along Y is even. It has been increased by 1 to be odd.")

            # Generate one-dimensional arrays for x and y coordinates
            x = np.linspace(-nb_of_samples_along_x * delta_x/2, nb_of_samples_along_x * delta_x/2, nb_of_samples_along_x)
            y = np.linspace(-nb_of_samples_along_y * delta_y/2, nb_of_samples_along_y * delta_y/2, nb_of_samples_along_y)

            # Create a two-dimensional grid of coordinates
            X_input_grid, Y_input_grid = np.meshgrid(x, y)

            return X_input_grid, Y_input_grid



    def calculate_alpha_c(self):



        self.Dm = D_m(sellmeier(self.system_config["system architecture"]["dispersive element"]["wavelength center"]),
                      np.radians(self.system_config["system architecture"]["dispersive element"]["A"]))

        if self.system_config["system architecture"]["dispersive element"]["type"] == "grating":
            self.alpha_c = 0
            self.alpha_c_transmis = simplified_grating_in_out(self.alpha_c,
                                                              self.system_config["system architecture"]["dispersive element"]["wavelength center"],
                                                              self.system_config["system architecture"]["dispersive element"]["m"],
                                                              self.system_config["system architecture"]["dispersive element"]["G"])
        elif self.system_config["system architecture"]["dispersive element"]["type"] == "prism":
            self.alpha_c = alpha_c(np.radians(self.system_config["system architecture"]["dispersive element"]["A"]),
                               self.Dm)
            self.alpha_c_transmis = self.alpha_c

        return self.alpha_c
    def propagate_mask_grid(self,X_input_grid=None,Y_input_grid=None):

        wavelength_min = self.system_config["spectral range"]["wavelength min"]
        wavelength_max = self.system_config["spectral range"]["wavelength max"]
        spectral_samples = self.system_config["spectral range"]["number of spectral samples"]

        if X_input_grid is None:
            X_input_grid = self.X_dmd_coordinates_grid
        if Y_input_grid is None:
            Y_input_grid = self.Y_dmd_coordinates_grid


        wavelengths = np.linspace(wavelength_min,
                                  wavelength_max,
                                  spectral_samples)
        self.list_wavelengths = wavelengths

        self.n_array_center = np.full(X_input_grid.shape,
                                      sellmeier(self.system_config["system architecture"]["dispersive element"]["wavelength center"]))
        # self.lba_array_center = np.full(X_input_grid.shape,
        #                             self.system_config["system architecture"]["dispersive element"]["wavelength center"])

        X_input_grid_flatten = X_input_grid.flatten()
        Y_input_grid_flatten = Y_input_grid.flatten()


        self.list_X_propagated_mask = list()
        self.list_Y_propagated_mask = list()

        self.alpha_c = self.calculate_alpha_c()


        for lba in np.linspace(wavelength_min,wavelength_max,spectral_samples):

            n_array_flatten = np.full(X_input_grid_flatten.shape, sellmeier(lba))
            lba_array_flatten = np.full(X_input_grid_flatten.shape, lba)

            X_propagated_mask, Y_propagated_mask = propagate_through_arm_vector(
                                                        dispersive_element_type=self.system_config["system architecture"]["dispersive element"]["type"],
                                                        X_mask= X_input_grid_flatten ,
                                                        Y_mask= Y_input_grid_flatten,
                                                        n = n_array_flatten,
                                                        lba = lba_array_flatten,
                                                        A =np.radians(self.system_config["system architecture"]["dispersive element"]["A"]),
                                                        G = self.system_config["system architecture"]["dispersive element"]["G"],
                                                        m = self.system_config["system architecture"]["dispersive element"]["m"],
                                                        F = self.system_config["system architecture"]["focal lens"],
                                                        alpha_c = self.alpha_c,
                                                        alpha_c_transmis = self.alpha_c_transmis,
                                                        delta_alpha_c = np.radians(self.system_config["system architecture"]["dispersive element"]["delta alpha c"]),
                                                        delta_beta_c= np.radians(self.system_config["system architecture"]["dispersive element"]["delta beta c"])
                                                        )
            self.list_X_propagated_mask.append(X_propagated_mask.reshape(X_input_grid.shape))
            self.list_Y_propagated_mask.append(Y_propagated_mask.reshape(Y_input_grid.shape))




        return self.list_X_propagated_mask, self.list_Y_propagated_mask, self.list_wavelengths

    def generate_psf(self,type,radius):

        if type =="Gaussian":

            X, Y, PSF = generate_2D_gaussian(radius,self.system_config["detector"]["delta X"],self.system_config["detector"]["delta Y"], 10)
            self.psf = PSF

        return self.psf

    def apply_psf(self):

        if (self.psf is not None) and (self.last_measurement_3D is not None):
            # Expand the dimensions of the 2D matrix to match the 3D matrix
            psf_3D = np.expand_dims(self.psf, axis=-1)

            # Perform the convolution using convolve
            result = convolve(self.last_measurement_3D, psf_3D, mode='same')

        self.last_measurement_3D = result

        return self.last_measurement_3D



    def save_acquisition(self, config_filtering,config_acquisition):

        self.result_directory = initialize_directory(config_acquisition)

        with open(self.result_directory + "/config_system.yml", 'w') as file:
            yaml.safe_dump(self.system_config, file)

        with open(self.result_directory + "/config_filtering.yml", 'w') as file:
            yaml.safe_dump(config_filtering, file)

        with open(self.result_directory + "/config_acquisition.yml", 'w') as file:
            yaml.safe_dump(config_acquisition, file)


            # Calculate the other two arrays
            sum_last_measurement = np.sum(self.last_measurement_3D, axis=2)
            sum_scene_interpolated = np.sum(self.interpolated_scene, axis=2)

            # Save the arrays in an H5 file
            with h5py.File(self.result_directory + '/filtered_image.h5', 'w') as f:
                f.create_dataset('filtered_image', data=self.last_measurement_3D)
            with h5py.File(self.result_directory + '/image.h5', 'w') as f:
                f.create_dataset('image', data=sum_last_measurement)
            with h5py.File(self.result_directory + '/panchro.h5', 'w') as f:
                f.create_dataset('panchro', data=sum_scene_interpolated)
            with h5py.File(self.result_directory + '/filtering_cube.h5', 'w') as f:
                f.create_dataset('filtering_cube', data=self.filtering_cube)
            with h5py.File(self.result_directory + '/wavelengths.h5', 'w') as f:
                f.create_dataset('wavelengths', data=self.list_wavelengths)

        print("Acquisition saved in " + self.result_directory)

def worker(args):
    """
    Process to parallellize
    :param args:
    :return:
    """
    list_X_propagated_masks, list_Y_propagated_masks, mask, X_detector_coordinates_grid, Y_detector_coordinates_grid, wavelength_index = args

    list_X_propagated_masks = np.nan_to_num(list_X_propagated_masks)
    interpolated_mask = griddata((list_X_propagated_masks[wavelength_index][:, :].flatten(),
                                  list_Y_propagated_masks[wavelength_index][:, :].flatten()),
                                 mask.flatten(),
                                 (X_detector_coordinates_grid, Y_detector_coordinates_grid),
                                 method='linear')
    return interpolated_mask