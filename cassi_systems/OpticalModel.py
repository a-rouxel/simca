from cassi_systems.functions_general_purpose import *

class OpticalModel:
    """
    Class that contains the optical model caracteristics and propagation model
    """
    def __init__(self, system_config):
        self.set_optical_config(system_config)

    def update_config(self, new_config):
        self.set_optical_config(new_config)

    def set_optical_config(self, config):

        self.system_config = config

        self.dispersive_element_type = config["system architecture"]["dispersive element"]["type"]
        self.A = math.radians(config["system architecture"]["dispersive element"]["A"])
        self.G = config["system architecture"]["dispersive element"]["G"]
        self.lba_c = config["system architecture"]["dispersive element"]["wavelength center"]
        self.m = config["system architecture"]["dispersive element"]["m"]
        self.F = config["system architecture"]["focal lens"]
        self.delta_alpha_c = math.radians(config["system architecture"]["dispersive element"]["delta alpha c"])
        self.delta_beta_c = math.radians(config["system architecture"]["dispersive element"]["delta beta c"])

        self.set_wavelengths(config["spectral range"]["wavelength min"],
                             config["spectral range"]["wavelength max"],
                             config["spectral range"]["number of spectral samples"])

    def propagation_with_distorsions(self, X_input_grid, Y_input_grid):
        """
        Propagate the SLM mask through one CASSI system

        Args:
            X_input_grid (numpy array): x coordinates grid
            Y_input_grid (numpy array): y coordinates grid

        Returns:
            list_X_propagated_mask (list): list of the X coordinates of the propagated masks
            list_Y_propagated_mask (list): list of the Y coordinates of the propagated masks
        """

        self.calculate_central_dispersion()

        list_X_propagated_mask = list()
        list_Y_propagated_mask = list()


        X_input_grid_flatten = X_input_grid.flatten()
        Y_input_grid_flatten = Y_input_grid.flatten()


        for lba in np.linspace(self.system_wavelengths[0],
                               self.system_wavelengths[-1],
                               self.system_wavelengths.shape[0]):

            n_array_flatten = np.full(X_input_grid_flatten.shape, self.sellmeier(lba))
            lba_array_flatten = np.full(X_input_grid_flatten.shape, lba)

            X_propagated_mask, Y_propagated_mask = self.propagate_through_arm(X_input_grid_flatten,Y_input_grid_flatten,n=n_array_flatten,lba=lba_array_flatten)

            list_X_propagated_mask.append(X_propagated_mask.reshape(X_input_grid.shape))
            list_Y_propagated_mask.append(Y_propagated_mask.reshape(Y_input_grid.shape))

        return list_X_propagated_mask, list_Y_propagated_mask

    def propagation_with_no_distorsions(self, X_input_grid=None, Y_input_grid=None):
        """
        Vanilla Propagation model used in most cassi acquisitions simulation.

        Args:
            X_input_grid (numpy array): x coordinates of the grid to be propagated
            Y_input_grid (numpy array): y coordinates of the grid to be propagated

        Returns:
            list_X_propagated_mask (list): list of the X coordinates grids of the propagated masks
            list_Y_propagated_mask (list): list of the Y coordinates grids of the propagated masks
        """

        self.calculate_central_dispersion()

        list_X_propagated_mask = list()
        list_Y_propagated_mask = list()

        for idx, wav in enumerate(self.system_wavelengths):

            X_ref = -1 * X_input_grid + self.X0_propagated[idx]
            Y_ref = -1 * Y_input_grid + self.Y0_propagated[idx]

            list_X_propagated_mask.append(X_ref)
            list_Y_propagated_mask.append(Y_ref)

        return list_X_propagated_mask, list_Y_propagated_mask

    def set_wavelengths(self, wavelength_min, wavelength_max, nb_of_spectral_samples):
        """
        Set the wavelengths range of the optical system

        Args:
            wavelength_min (float): minimum wavelength of the system
            wavelength_max (float): maximum wavelength of the system
            nb_of_spectral_samples (int): number of spectral samples of the system
        Returns:

        """
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.nb_of_spectral_samples = nb_of_spectral_samples

        self.system_wavelengths = np.linspace(self.wavelength_min,self.wavelength_max,self.nb_of_spectral_samples)

    def calculate_central_dispersion(self):
        """
        Calculate the dispersion related to the central pixel of the SLM

        Returns:
            X0_propagated (numpy array): x coordinates of the central pixel of the SLM, each coordinate is associated to a system wavelength
            Y0_propagated (numpy array): y coordinates of the central pixel of the SLM, each coordinate is associated to a system wavelength

        """

        self.alpha_c = self.calculate_alpha_c()

        X0_coordinates_array_flatten = np.zeros(self.system_wavelengths.shape[0])
        Y0_coordinates_array_flatten = np.zeros(self.system_wavelengths.shape[0])
        lba_array_flatten = self.system_wavelengths

        n_array_flatten = np.full(lba_array_flatten.shape, self.sellmeier(lba_array_flatten))

        X0_propagated, Y0_propagated = self.propagate_through_arm(X_vec_in=X0_coordinates_array_flatten,Y_vec_in=Y0_coordinates_array_flatten,n=n_array_flatten,lba=lba_array_flatten)

        self.X0_propagated, self.Y0_propagated = X0_propagated, Y0_propagated

        self.central_distorsion_in_X = np.abs(self.X0_propagated[-1] - self.X0_propagated[0])

        return X0_propagated, Y0_propagated

    def propagate_through_arm(self, X_vec_in, Y_vec_in, n, lba):

        """
        Propagate the light through one system arm : (lens + dispersive element + lens)

        Args:
            X_vec_in (numpy array) : 1D array of the x coordinates of the SLM pixels
            Y_vec_in (numpy array) : 1D array of the y coordinates of the SLM pixels
            n (numpy array) : 1D array of the refractive index of the system (at the corresponding wavelength)
            lba (numpy array) : 1D array of the considered wavelength

        Returns:
            X_vec_out (numpy array) : 1D array of the x coordinates after propagation through the system
            Y_vec_out (numpy array) : 1D array of the y coordinates after propagation through the system
        """

        dispersive_element_type = self.dispersive_element_type
        A = self.A
        G = self.G
        m = self.m
        F = self.F
        delta_alpha_c = self.delta_alpha_c
        delta_beta_c = self.delta_beta_c
        alpha_c = self.alpha_c
        alpha_c_transmis = self.alpha_c_transmis


        if dispersive_element_type == "prism":

            angle_with_P1 = alpha_c - A / 2 + delta_alpha_c
            angle_with_P2 = alpha_c_transmis - A / 2 - delta_alpha_c

            k = self.model_Lens_pos_to_angle(X_vec_in, Y_vec_in, F)
            # Rotation in relation to P1 around the Y axis

            k_1 = rotation_y(angle_with_P1) @ k[:, 0, :]
            # Rotation in relation to P1 around the X axis
            k_2 = rotation_x(delta_beta_c) @ k_1
            # Rotation of P1 in relation to frame_in along the new Y axis
            k_3 = rotation_y(A / 2) @ k_2

            norm_k = np.sqrt(k_3[0] ** 2 + k_3[1] ** 2 + k_3[2] ** 2)
            k_3 /= norm_k

            k_out_p = self.model_Prism_angle_to_angle(k_3, n, A)
            k_out_p = k_out_p * norm_k

            k_3_bis = np.dot(rotation_y(A / 2), k_out_p)

            # Rotation in relation to P2 around the X axis
            k_2_bis = np.dot(rotation_x(-delta_beta_c), k_3_bis)
            # Rotation in relation to P2 around the Y axis
            k_1_bis = np.dot(rotation_y(angle_with_P2), k_2_bis)

            theta_out = np.arctan(k_1_bis[0] / k_1_bis[2])
            phi_out = np.arctan(k_1_bis[1] / k_1_bis[2])

            [X_vec_out, Y_vec_out] = self.model_Lens_angle_to_position(theta_out, phi_out, F)


        elif dispersive_element_type == "grating":

            angle_with_P1 = alpha_c - delta_alpha_c
            angle_with_P2 = alpha_c_transmis + delta_alpha_c

            k = self.model_Lens_pos_to_angle(X_vec_in, Y_vec_in, F)
            # Rotation in relation to P1 around the Y axis

            k_1 = rotation_y(angle_with_P1) @ k[:, 0, :]
            # Rotation in relation to P1 around the X axis
            k_2 = rotation_x(delta_beta_c) @ k_1

            k_3 = rotation_y(0) @ k_2
            norm_k = np.sqrt(k_3[0] ** 2 + k_3[1] ** 2 + k_3[2] ** 2)
            k_3 /= norm_k

            k_out_p = self.model_Grating_angle_to_angle(k_3, lba, m, G)
            k_out_p = k_out_p * norm_k

            k_3_bis = np.dot(rotation_y(0), k_out_p)

            # Rotation in relation to P2 around the X axis
            k_2_bis = np.dot(rotation_x(-delta_beta_c), k_3_bis)
            # Rotation in relation to P2 around the Y axis
            k_1_bis = np.dot(rotation_y(angle_with_P2), k_2_bis)

            theta_out = np.arctan(k_1_bis[0] / k_1_bis[2])
            phi_out = np.arctan(k_1_bis[1] / k_1_bis[2])

            [X_vec_out, Y_vec_out] = self.model_Lens_angle_to_position(theta_out, phi_out, F)

        else:
            raise Exception("dispersive_element_type should be prism or grating")

        return X_vec_out, Y_vec_out

    def model_Grating_angle_to_angle(self,k_in, lba, m, G):

        alpha_in = np.arctan(k_in[0]) * np.sqrt(1 + np.tan(k_in[0])**2 + np.tan(k_in[1])**2)
        beta_in = np.arctan(k_in[1]) * np.sqrt(1 + np.tan(k_in[0])**2 + np.tan(k_in[1])**2)

        alpha_out = -1*np.arcsin(m * lba*10**-9  * G * 10**3 - np.sin(alpha_in))
        beta_out = beta_in


        k_out = [np.sin(alpha_out) * np.cos(beta_out),
                 np.sin(beta_out)*np.cos(alpha_out),
                 np.cos(alpha_out) * np.cos(beta_out)]

        return k_out

    def simplified_grating_in_out(self, alpha,lba,m,G):

        alpha_out = np.arcsin(m * lba * 10 ** -9 * G * 10 ** 3 - np.sin(alpha))

        return alpha_out

    def model_Lens_angle_to_position(self,alpha, beta,F):

        x = F * np.tan(alpha)
        y = F * np.tan(beta)

        return [x, y]

    def model_Prism_angle_to_angle(self,k0, n,A):
        """
        Ray tracing through the prism

        Parameters
        ----------
        k0 : numpy array -- no units
            input vector
        n : float -- no units
            refractive index of the considered wavelength
        A : float -- degrees
            apex angle of the prism
        Returns
        -------
        kout : numpy array -- no units
            output vector

        """

        kp = np.array([k0[0], k0[1], np.sqrt(n ** 2 - k0[0] ** 2 - k0[1] ** 2)])

        kp_r = np.matmul(rotation_y(-A), kp)

        kout = [kp_r[0], kp_r[1], np.sqrt(1 - kp_r[0] ** 2 - kp_r[1] ** 2)]

        return kout


    def model_Lens_pos_to_angle(self,x_obj, y_obj, F):

        alpha = -1*np.arctan(x_obj / F)
        beta  = -1*np.arctan(y_obj / F)

        k0 = np.array([[np.sin(alpha) * np.cos(beta)],
                       [np.sin(beta)*np.cos(alpha)],
                       [np.cos(alpha) * np.cos(beta)]])

        return k0

    def calculate_alpha_c(self):
        """
        Calculate the relative angle of incidence between the lenses and the dispersive element

        Args:

        Returns:
            alpha_c (float): angle of incidence
        """
        if self.dispersive_element_type == "prism":
            self.Dm = self.calculate_minimum_deviation(self.sellmeier(self.lba_c), self.A)
            self.alpha_c = self.get_incident_angle_min_dev(self.A,self.Dm)
            self.alpha_c_transmis = self.alpha_c

        if self.dispersive_element_type == "grating":
            self.alpha_c = 0
            self.alpha_c_transmis = self.simplified_grating_in_out(self.alpha_c,self.lba_c,self.m,self.G)

        return self.alpha_c


    def calculate_minimum_deviation(self,n, A):
        """
        minimum deviation angle of a prism of index n and apex angle A

        Args:
            n (float or numpy array): index of the prism -- no units
            A (float): apex angle of the prism -- in radians

        Returns:
            D_m (float or numpy array): minimum deviation angle -- in radians
        """
        return 2 * np.arcsin(n * np.sin(A / 2)) - A

    def get_incident_angle_min_dev(self,A, D_m):
        """
        Calculate the angle of incidence corresponding to the minimum deviation angle

        Args:
            A (float): apex angle of the prism -- in radians
            D_m (float): minimum deviation angle -- in radians

        """
        return (A + D_m) / 2


    def sellmeier(self,lambda_, glass_type="BK7"):
        """
        Evaluating the refractive index value of a prism for a given lambda based on Sellmeier equation

        Args:
            lambda_ (numpy array of float) : wavelength in nm

        Returns:
            n (numpy array of float): index value corresponding to the input wavelength

        """

        if glass_type == "BK7":
            B1 = 1.03961212
            B2 = 0.231792344
            B3 = 1.01046945
            C1 = 6.00069867 * (10 ** -3)
            C2 = 2.00179144 * (10 ** -2)
            C3 = 1.03560653 * (10 ** 2)

        else :
            raise Exception("glass_type is Unknown")

        lambda_in_mm = lambda_ / 1000

        n = np.sqrt(1 + B1 * lambda_in_mm ** 2 / (lambda_in_mm ** 2 - C1) + B2 * lambda_in_mm ** 2 / (
                    lambda_in_mm ** 2 - C2) + B3 * lambda_in_mm ** 2 / (lambda_in_mm ** 2 - C3))

        return n

    def generate_2D_gaussian(self,radius, sample_size_x, sample_size_y, nb_of_samples):
        """
        Generate a 2D Gaussian of a given radius

        Args:
            radius (float): radius of the Gaussian
            sample_size_x (float): size of each sample along the x axis
            sample_size_y (float): size of each sample along the y axis
            nb_of_samples (int): number of samples along each axis

        Returns:
            X (numpy array): 2D array of the x coordinates of the grid
            Y (numpy array): 2D array of the y coordinates of the grid
            gaussian_2d (numpy array): array of the 2D Gaussian
        """

        # Define the grid
        grid_size_x = sample_size_x * (nb_of_samples - 1)
        grid_size_y = sample_size_y * (nb_of_samples - 1)
        x = np.linspace(-grid_size_x / 2, grid_size_x / 2, nb_of_samples)
        y = np.linspace(-grid_size_y / 2, grid_size_y / 2, nb_of_samples)
        X, Y = np.meshgrid(x, y)

        # Compute the 2D Gaussian function
        gaussian_2d = np.exp(-(X ** 2 + Y ** 2) / (2 * radius ** 2))

        return X, Y, gaussian_2d
    def generate_psf(self, type, radius):
        """
        Generate a PSF

        Args:
            type (str): type of PSF to generate
            radius (float): radius of the PSF

        Returns:
            PSF (numpy array): PSF generated
        """

        if type == "Gaussian":
            X, Y, PSF = self.generate_2D_gaussian(radius, self.system_config["detector"]["pixel size along X"],
                                             self.system_config["detector"]["pixel size along Y"], 10)
            self.psf = PSF

        return self.psf

    def check_if_sampling_is_sufficiant(self):
        """
        Check if the sampling is sufficiant to avoid aliasing.

        Returns:
            nb_of_sample_points_per_pix (float): number of sample points per pixel

        """

        pix_size = self.system_config["detector"]["pixel size along X"]
        nb_of_system_wavelengths = self.system_wavelengths.shape[0]

        nb_of_sample_points_per_pix =  pix_size / (self.central_distorsion_in_X / nb_of_system_wavelengths )
        print("number of spectral sample points per pixel =",nb_of_sample_points_per_pix)

        if nb_of_sample_points_per_pix < 2:

            print("The 'number of spectral samples' (cf. system config) is not sufficiant to avoid weird sampling effect( aliasing ?). RAISE IT !")

        return nb_of_sample_points_per_pix
