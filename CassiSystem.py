from utils.functions_retropropagating import *
from scipy.interpolate import griddata
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from scipy import fftpack
import numpy as np
from utils.scenes_helper import *


def worker(args):
    """
    Process to parallellize
    :param args:
    :return:
    """
    list_X_propagated_masks, list_Y_propagated_masks, mask, X_detector_grid, Y_detector_grid, wavelength_index = args

    list_X_propagated_masks = np.nan_to_num(list_X_propagated_masks)
    interpolated_mask = griddata((list_X_propagated_masks[wavelength_index][:, :].flatten(),
                                  list_Y_propagated_masks[wavelength_index][:, :].flatten()),
                                 mask.flatten(),
                                 (X_detector_grid, Y_detector_grid),
                                 method='linear')
    return interpolated_mask



class CassiSystem():

    def __init__(self,system_config_path):

        self.system_config = load_yaml_config(system_config_path)


        self.X_dmd_grid, self.Y_dmd_grid = self.create_grid(self.system_config["SLM"]["sampling across X"],
                                                                        self.system_config["SLM"]["sampling across Y"],
                                                                        self.system_config["SLM"]["delta X"],
                                                                        self.system_config["SLM"]["delta Y"])


        self.X_detector_grid, self.Y_detector_grid = self.create_grid(self.system_config["detector"]["sampling across X"],
                                                                        self.system_config["detector"]["sampling across Y"],
                                                                        self.system_config["detector"]["delta X"],
                                                                        self.system_config["detector"]["delta Y"])

    def load_scene(self,directory,scene_name):


            img, gt, list_wavelengths, label_values, ignored_labels, rgb_bands, palette, delta_lambda = get_dataset(directory,scene_name)
            self.scene = img
            self.scene_gt = gt
            self.list_scene_wavelengths = list_wavelengths
            self.scene_label_values = label_values
            self.scene_ignored_labels = ignored_labels
            self.scene_rgb_bands = rgb_bands
            self.scene_palette = palette
            self.scene_delta_lambda = delta_lambda

            self.scene_palette = palette_init(label_values, palette)
            #
            # print("Error: scene not found")
            # self.scene_loaded.emit(self.scene.shape[1],self.scene.shape[0],self.scene.shape[2],list_wavelengths[0],list_wavelengths[-1])

    def update_config(self,new_config):

        self.system_config = new_config

        self.X_dmd_grid, self.Y_dmd_grid = self.create_grid(self.system_config["SLM"]["sampling across X"],
                                                                        self.system_config["SLM"]["sampling across Y"],
                                                                        self.system_config["SLM"]["delta X"],
                                                                        self.system_config["SLM"]["delta Y"])


        self.X_detector_grid, self.Y_detector_grid = self.create_grid(self.system_config["detector"]["sampling across X"],
                                                                        self.system_config["detector"]["sampling across Y"],
                                                                        self.system_config["detector"]["delta X"],
                                                                        self.system_config["detector"]["delta Y"])

    def interpolate_scene(self,new_sampling,chunk_size):
        self.scene_interpolated = interpolate_scene_cube_along_wavelength(self.scene, self.list_scene_wavelengths, new_sampling,chunk_size)
        return self.scene_interpolated

    def generate_dmd_grid(self):

        self.X_dmd_mask, self.Y_dmd_mask = self.create_grid(
            self.system_config["SLM"]["sampling across X"],
            self.system_config["SLM"]["sampling across Y"],
            self.system_config["SLM"]["delta X"],
            self.system_config["SLM"]["delta Y"])

        return self.X_dmd_mask, self.Y_dmd_mask

    def generate_2D_mask(self, mask_type,slit_position=None,slit_width=None,mask_path=None):


        if self.system_config["SLM"]["sampling across Y"] % 2 == 0:
            self.system_config["SLM"]["sampling across Y"] += 1
        if self.system_config["SLM"]["sampling across X"] % 2 == 0:
            self.system_config["SLM"]["sampling across X"] += 1

        if mask_type == "random":
            self.mask = np.random.randint(0, 2, (self.system_config["SLM"]["sampling across Y"],
                                                 self.system_config["SLM"]["sampling across X"]))
        elif mask_type == "slit":
            self.mask = np.zeros((self.system_config["SLM"]["sampling across Y"],
                                  self.system_config["SLM"]["sampling across X"]))

            if slit_position is not None:
                slit_position = int(self.system_config["SLM"]["sampling across X"] / 2) + slit_position
            else :
                slit_position = int(self.system_config["SLM"]["sampling across X"] / 2)
            if slit_width is None:
                slit_width = 1

            self.mask[:,slit_position-slit_width//2:slit_position+slit_width] = 1

        elif mask_type == "blue":
            size = (self.system_config["SLM"]["sampling across Y"], self.system_config["SLM"]["sampling across X"])
            self.mask = self.generate_blue_noise(size)

        elif mask_type == "custom h5 mask":
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

        print("--- Generating filtering cube ---- ")

        X_detector_grid = self.X_detector_grid
        Y_detector_grid = self.Y_detector_grid
        list_X_propagated_masks = self.list_X_propagated_mask
        list_Y_propagated_masks = self.list_Y_propagated_mask
        mask =  self.mask

        if self.system_config["detector"]["sampling across Y"] %2 ==0:
            self.system_config["detector"]["sampling across Y"] +=1
        if self.system_config["detector"]["sampling across X"] %2 ==0:
            self.system_config["detector"]["sampling across X"] +=1

        self.filtering_cube = np.zeros((self.system_config["detector"]["sampling across Y"],
                                        self.system_config["detector"]["sampling across X"],
                                        self.system_config["spectral range"]["number of spectral samples"]))

        wavelengths = np.linspace(self.system_config["spectral range"]["wavelength min"],
                                  self.system_config["spectral range"]["wavelength max"],
                                  self.system_config['spectral range']["number of spectral samples"])
        with Pool(mp.cpu_count()) as p:
            tasks = [(list_X_propagated_masks, list_Y_propagated_masks, mask, X_detector_grid, Y_detector_grid, i)
                     for i in range(len(wavelengths))]
            for index, zi in tqdm(enumerate(p.imap(worker, tasks)), total=len(wavelengths), desc='Processing tasks'):
                self.filtering_cube[:, :, index] = zi

        self.list_wavelengths = wavelengths

        return self.filtering_cube

    def generate_sd_measurement_cube(self,scene):


        X_detector_grid = self.X_detector_grid
        Y_detector_grid = self.Y_detector_grid
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
            tasks = [(list_X_propagated_masks, list_Y_propagated_masks, scene[:,:,i], X_detector_grid, Y_detector_grid, i)
                     for i in range(len(wavelengths))]
            for index, zi in tqdm(enumerate(p.imap(worker, tasks)), total=len(wavelengths), desc='Processing tasks'):
                self.measurement_sd[:, :, index] = zi

        self.list_wavelengths = wavelengths

        return self.measurement_sd



    def create_grid(self,nb_of_samples_along_x, nb_of_samples_along_y, delta_x, delta_y):

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
    def propagate_mask_grid(self,spectral_range,spectral_samples,X_input_grid=None,Y_input_grid=None):

        if X_input_grid is None:
            X_input_grid = self.X_dmd_grid
        if Y_input_grid is None:
            Y_input_grid = self.Y_dmd_grid

        wavelength_min = spectral_range[0]
        wavelength_max = spectral_range[1]

        self.n_array_center = np.full(X_input_grid.shape,
                                      sellmeier(self.system_config["system architecture"]["dispersive element"]["wavelength center"]))
        # self.lba_array_center = np.full(X_input_grid.shape,
        #                             self.system_config["system architecture"]["dispersive element"]["wavelength center"])

        X_input_grid_flatten = X_input_grid.flatten()
        Y_input_grid_flatten = Y_input_grid.flatten()


        self.list_wavelengths= list()
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
            self.list_wavelengths.append(np.full(X_input_grid.shape, lba).reshape(Y_input_grid.shape))




        return self.list_X_propagated_mask, self.list_Y_propagated_mask, self.list_wavelengths

