from utils.functions_retropropagating import *
class CassiSystem():

    def __init__(self,system_config,simulation_config):

        self.system_config = system_config
        self.simulation_config = simulation_config
        self.result_directory = initialize_directory(self.simulation_config)
        self.alpha_c = self.calculate_alpha_c()
        self.X_mask_grid, self.Y_mask_grid = self.create_mask_grid(self.simulation_config["input grid sampling"]["sampling across X"],
                                                                      self.simulation_config["input grid sampling"]["sampling across Y"],
                                                                      self.simulation_config["input grid sampling"]["delta X"],
                                                                      self.simulation_config["input grid sampling"]["delta Y"])

    def create_mask_grid(self,nb_of_samples_along_x, nb_of_samples_along_y, delta_x, delta_y):

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
            X_mask_grid, Y_mask_grid = np.meshgrid(x, y)

            return X_mask_grid, Y_mask_grid

    def calculate_alpha_c(self):



        self.Dm = D_m(sellmeier(self.system_config["system architecture"]["dispersive element 1"]["wavelength center"]),
                      np.radians(self.system_config["system architecture"]["dispersive element 1"]["A"]))
        self.alpha_c = alpha_c(np.radians(self.system_config["system architecture"]["dispersive element 1"]["A"]),
                               self.Dm)

        return self.alpha_c
    def propagate_mask_grid(self,spectral_range,spectral_samples):

        wavelength_min = spectral_range[0]
        wavelength_max = spectral_range[1]

        self.n_array_center = np.full(self.X_mask_grid.shape,
                                      sellmeier(self.system_config["system architecture"]["dispersive element 1"]["wavelength center"]))

        self.X_mask_grid_flatten = self.X_mask_grid.flatten()
        self.Y_mask_grid_flatten = self.Y_mask_grid.flatten()


        self.list_wavelengths= list()
        self.list_X_detector = list()
        self.list_Y_detector = list()

        for lba in np.linspace(wavelength_min,wavelength_max,spectral_samples):

            n_array_flatten = np.full(self.X_mask_grid_flatten.shape, sellmeier(lba))

            X_detector, Y_detector = propagate_through_arm_vector(X_mask= self.X_mask_grid_flatten ,
                                                        Y_mask= self.Y_mask_grid_flatten,
                                                        n = n_array_flatten,
                                                        A =np.radians(self.system_config["system architecture"]["dispersive element 1"]["A"]),
                                                        F = self.system_config["system architecture"]["focal lens 1"],
                                                        alpha_c = self.alpha_c,
                                                        delta_alpha_c = np.radians(self.system_config["system architecture"]["dispersive element 1"]["delta alpha c"]),
                                                        delta_beta_c= np.radians(self.system_config["system architecture"]["dispersive element 1"]["delta beta c"])
                                                        )
            self.list_X_detector.append(X_detector.reshape(self.X_mask_grid.shape))
            self.list_Y_detector.append(Y_detector.reshape(self.Y_mask_grid.shape))
            self.list_wavelengths.append(np.full(self.X_mask_grid.shape, lba).reshape(self.Y_mask_grid.shape))

        return self.list_X_detector, self.list_Y_detector, self.list_wavelengths

