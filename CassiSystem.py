from utils.functions_retropropagating import *
import numpy as np
from scipy.interpolate import griddata
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from scipy import fftpack
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.interpolate import interpn
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

    def __init__(self,system_config):

        self.system_config = system_config
        self.result_directory = initialize_directory(self.system_config)
        self.alpha_c = self.calculate_alpha_c()

        self.X_dmd_grid, self.Y_dmd_grid = self.create_grid(self.system_config["SLM"]["sampling across X"],
                                                                        self.system_config["SLM"]["sampling across Y"],
                                                                        self.system_config["SLM"]["delta X"],
                                                                        self.system_config["SLM"]["delta Y"])


        self.X_detector_grid, self.Y_detector_grid = self.create_grid(self.system_config["detector"]["sampling across X"],
                                                                        self.system_config["detector"]["sampling across Y"],
                                                                        self.system_config["detector"]["delta X"],
                                                                        self.system_config["detector"]["delta Y"])



    def create_dmd_mask(self):

        self.X_dmd_mask, self.Y_dmd_mask = self.create_grid(
            self.system_config["SLM"]["sampling across X"],
            self.system_config["SLM"]["sampling across Y"],
            self.system_config["SLM"]["delta X"],
            self.system_config["SLM"]["delta Y"])

        return self.X_dmd_mask, self.Y_dmd_mask

    def generate_2D_mask(self, mask_type):

        print(self.system_config["SLM"]["sampling across Y"])
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
            self.mask[:, int(self.system_config["SLM"]["sampling across X"] / 2)] = 1
        elif mask_type == "blue":
            size = (self.system_config["SLM"]["sampling across Y"], self.system_config["SLM"]["sampling across X"])
            self.mask = self.generate_blue_noise(size)

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



    def generate_filtering_cube(self, X_detector_grid, Y_detector_grid, list_X_propagated_masks,
                                list_Y_propagated_masks, mask):

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

    def interpolate_filtering_cube_along_wavelength(self,nb_of_wav_samples):

        list_wavelength = np.array(self.list_wavelengths)

        print(list_wavelength.shape)

        # Generate the coordinates for the original grid

        # Create new coordinates for interpolation
        new_z = np.linspace(list_wavelength[0,0,0], list_wavelength[-1,0,0], nb_of_wav_samples)

        print(new_z.shape)
        new_coordinates = np.meshgrid(self.Y_detector_grid[:,0], self.X_detector_grid[0,:], new_z, indexing='ij')

        print(new_coordinates[0].shape)

        print(self.filtering_cube.shape)

        # Perform the interpolation
        new_grid = interpn((self.Y_detector_grid[:,0],self.X_detector_grid[0,:], list_wavelength[:,0,0]), self.filtering_cube, tuple(new_coordinates))

        self.filtering_cube = new_grid
        self.list_wavelengths = new_z

        print(new_grid.shape)
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
        self.alpha_c = alpha_c(np.radians(self.system_config["system architecture"]["dispersive element"]["A"]),
                               self.Dm)

        return self.alpha_c
    def propagate_mask_grid(self,X_input_grid,Y_input_grid,spectral_range,spectral_samples):

        wavelength_min = spectral_range[0]
        wavelength_max = spectral_range[1]

        self.n_array_center = np.full(X_input_grid.shape,
                                      sellmeier(self.system_config["system architecture"]["dispersive element"]["wavelength center"]))

        X_input_grid_flatten = X_input_grid.flatten()
        Y_input_grid_flatten = Y_input_grid.flatten()


        self.list_wavelengths= list()
        self.list_X_propagated_mask = list()
        self.list_Y_propagated_mask = list()

        for lba in np.linspace(wavelength_min,wavelength_max,spectral_samples):

            n_array_flatten = np.full(X_input_grid_flatten.shape, sellmeier(lba))

            X_propagated_mask, Y_propagated_mask = propagate_through_arm_vector(X_mask= X_input_grid_flatten ,
                                                        Y_mask= Y_input_grid_flatten,
                                                        n = n_array_flatten,
                                                        A =np.radians(self.system_config["system architecture"]["dispersive element"]["A"]),
                                                        F = self.system_config["system architecture"]["focal lens"],
                                                        alpha_c = self.alpha_c,
                                                        delta_alpha_c = np.radians(self.system_config["system architecture"]["dispersive element"]["delta alpha c"]),
                                                        delta_beta_c= np.radians(self.system_config["system architecture"]["dispersive element"]["delta beta c"])
                                                        )
            self.list_X_propagated_mask.append(X_propagated_mask.reshape(X_input_grid.shape))
            self.list_Y_propagated_mask.append(Y_propagated_mask.reshape(Y_input_grid.shape))
            self.list_wavelengths.append(np.full(X_input_grid.shape, lba).reshape(Y_input_grid.shape))

        return self.list_X_propagated_mask, self.list_Y_propagated_mask, self.list_wavelengths

