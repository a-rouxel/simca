import torch
import snoop
import opticalglass as og
from simca.helper import *
# import snoop
import torch
from opticalglass.glassfactory import create_glass
import pytorch_lightning as pl
from torch import nn
from opticalglass.glassfactory import get_glass_catalog


class OpticalModel(pl.LightningModule):
    """
    Class that contains the optical model caracteristics and propagation models
    """
    def __init__(self, config,index_estimation_method="cauchy",device="cpu"):
        super().__init__()

        self.to(device)
        self.system_config = config
        self.index_estimation_method = index_estimation_method

        self.lba_c = nn.Parameter(torch.tensor(config["system architecture"]["dispersive element"]["wavelength center"], dtype=torch.float,device=self.device,requires_grad=True))
        self.define_lenses_carac()
        self.define_dispersive_element_carac()


    def define_lenses_carac(self):

        config = self.system_config
        self.F = torch.tensor(config["system architecture"]["focal lens"],
                              dtype=torch.float,device=self.device)  # If F needs to be optimized, wrap with nn.Parameter as well
        
        return self.F


    def define_dispersive_element_carac(self):

        config = self.system_config 
        self.dispersive_element_type = config["system architecture"]["dispersive element"]["type"]

        if self.dispersive_element_type == "grating":
            self.G = torch.tensor(config["system architecture"]["dispersive element"]["G"], dtype=torch.float,device=self.device)
            self.m = torch.tensor(config["system architecture"]["dispersive element"]["m"], dtype=torch.float,device=self.device)
            if self.index_estimation_method == "cauchy":
                self.nd1, self.vd1 = (nn.Parameter(torch.tensor(1, dtype=torch.float,device=self.device)),nn.Parameter(torch.tensor(1, dtype=torch.float,device=self.device)))
                self.nd2, self.vd2 = (nn.Parameter(torch.tensor(1, dtype=torch.float,device=self.device)),nn.Parameter(torch.tensor(1, dtype=torch.float,device=self.device)))
                self.nd3, self.vd3 = (nn.Parameter(torch.tensor(1, dtype=torch.float,device=self.device)),nn.Parameter(torch.tensor(1, dtype=torch.float,device=self.device)))
            self.delta_beta_c = nn.Parameter(torch.tensor(math.radians(config["system architecture"]["dispersive element"]["delta beta c"]), dtype=torch.float,device=self.device))


        elif self.dispersive_element_type in ["prism", "doubleprism", "amici", "tripleprism"]:
            self.catalog = config["system architecture"]["dispersive element"]["catalog"]
            self.glass1 = config["system architecture"]["dispersive element"]["glass1"]
            self.A1 = nn.Parameter(torch.tensor(math.radians(config["system architecture"]["dispersive element"]["A1"]), dtype=torch.float,device=self.device))
            self.A2 = nn.Parameter(torch.tensor(0, dtype=torch.float,device=self.device))
            self.A3 = nn.Parameter(torch.tensor(0, dtype=torch.float,device=self.device))
            if self.index_estimation_method == "cauchy":
                self.nd1, self.vd1 = self.get_corresponding_nd_vd(self.glass1,self.catalog)
                self.nd2, self.vd2 = (nn.Parameter(torch.tensor(1, dtype=torch.float,device=self.device)),nn.Parameter(torch.tensor(1, dtype=torch.float,device=self.device)))
                self.nd3, self.vd3 = (nn.Parameter(torch.tensor(1, dtype=torch.float,device=self.device)),nn.Parameter(torch.tensor(1, dtype=torch.float,device=self.device)))
            # only used for prisms
            self.delta_beta_c = nn.Parameter(torch.tensor(math.radians(config["system architecture"]["dispersive element"]["delta beta c"]), dtype=torch.float,device=self.device))
        
        if self.dispersive_element_type in ["doubleprism", "amici", "tripleprism"]:
        
            self.glass2 = config["system architecture"]["dispersive element"]["glass2"]
            self.A2 = nn.Parameter(torch.tensor(math.radians(config["system architecture"]["dispersive element"]["A2"]), dtype=torch.float,device=self.device))
            if self.index_estimation_method == "cauchy":
                self.nd2, self.vd2 = self.get_corresponding_nd_vd(self.glass2,self.catalog)
                self.nd3, self.vd3 = (nn.Parameter(torch.tensor(1, dtype=torch.float,device=self.device)),nn.Parameter(torch.tensor(1, dtype=torch.float,device=self.device)))

        
        if self.dispersive_element_type in ["amici", "tripleprism"]:
            
            self.glass3 = config["system architecture"]["dispersive element"]["glass3"]
            self.A3 = nn.Parameter(torch.tensor(math.radians(config["system architecture"]["dispersive element"]["A3"]), dtype=torch.float,device=self.device))
            if self.index_estimation_method == "cauchy":
                self.nd3, self.vd3 = self.get_corresponding_nd_vd(self.glass3,self.catalog)
        
        if self.dispersive_element_type =="amici":
            self.A3 = self.A1
            self.glass3 = self.glass1
            if self.index_estimation_method == "cauchy":
                self.nd3 = self.nd1
                self.vd3 = self.vd1

        self.alpha_c = nn.Parameter(
            torch.tensor(math.radians(config["system architecture"]["dispersive element"]["alpha_c"]),
                         dtype=torch.float,device=self.device))
        self.delta_alpha_c = nn.Parameter(torch.tensor(
            math.radians(config["system architecture"]["dispersive element"]["delta alpha c"]), dtype=torch.float,device=self.device))
        
        # Calculation based on the above parameters
        alpha_c_transmis = -1 * self.propagate_central_microm_through_disperser(self.lba_c)
        self.alpha_c_transmis = alpha_c_transmis[0,0,0]

    
    def get_corresponding_nd_vd(self,glass_name,catalog):
        glass_pd = get_glass_catalog(catalog)
        list_of_glasses = glass_pd.df.iloc[:, 0].index.tolist()

        idx_glass = list_of_glasses.index(glass_name)
        tuple_glass_data = glass_pd.glass_map_data("d")

        nd = tuple_glass_data[0][idx_glass]
        vd = tuple_glass_data[1][idx_glass]

        nd = nn.Parameter(torch.tensor(nd, dtype=torch.float,device=self.device))
        vd = nn.Parameter(torch.tensor(vd, dtype=torch.float,device=self.device))

        # print(f"nd = {nd}, vd = {vd}")

        return nd, vd

    def rerun_central_dispersion(self):
        alpha_c_transmis = -1 * self.propagate_central_microm_through_disperser(self.lba_c)
        self.alpha_c_transmis = alpha_c_transmis
        return alpha_c_transmis
    

    def propagate(self,X,Y,lba,n1,n2,n3):

        k = self.model_Lens_pos_to_angle(X, Y, self.F)

        alpha_1 = self.define_alpha_1()

        k = self.rotate_from_lens_to_dispersive_element(k,self.alpha_c,self.delta_alpha_c,self.delta_beta_c,alpha_1)

        if self.dispersive_element_type in ["prism", "doubleprism", "amici", "tripleprism"]:
            k, list_theta_in, list_theta_out = self.propagate_through_triple_prism(k,n1,n2,n3,self.A1,self.A2,self.A3)
        elif self.dispersive_element_type == "grating":
            k = self.model_Grating_angle_to_angle(k, lba, self.m, self.G)

        k = self.rotate_from_dispersive_element_to_lens(k,self.alpha_c_transmis,self.delta_alpha_c,self.delta_beta_c,alpha_1)

        X_vec_out, Y_vec_out = self.model_Lens_angle_to_position(k, self.F)

        return X_vec_out, Y_vec_out
    
    
    def repropagate(self,X,Y,lba,n1,n2,n3):

        k = self.model_Lens_pos_to_angle(X, Y, self.F)

        alpha_1 = self.define_alpha_1()

        k = self.rotate_from_lens_to_dispersive_element(k,self.alpha_c_transmis,-1*self.delta_alpha_c,-1*self.delta_beta_c,alpha_1)

        if self.dispersive_element_type in ["prism", "doubleprism", "amici", "tripleprism"]:
            k, list_theta_in, list_theta_out = self.propagate_through_triple_prism(k,n1,n2,n3,self.A1,self.A2,self.A3)
        elif self.dispersive_element_type == "grating":
            k = self.model_Grating_angle_to_angle(k, lba, self.m, self.G)

        
        k = self.rotate_from_dispersive_element_to_lens(k,self.alpha_c,-1*self.delta_alpha_c,-1*self.delta_beta_c,alpha_1)

        # print(k)
        X_vec_out, Y_vec_out = self.model_Lens_angle_to_position(k, self.F)

        return X_vec_out, Y_vec_out

    def calculate_dispersion_with_cauchy(self,lambda_vec, nD, V):

        lambda_vec = lambda_vec * 1e-9  # Convert nm to meters for calculation
        lambda_D = 589.3e-9  # D line wavelength in meters
        lambda_F = 486.1e-9  # F line wavelength in meters
        lambda_C = 656.3e-9  # C line wavelength in meters

        # Calculate B using the given formula
        B = (nD - 1) / (V * (1 / lambda_F ** 2 - 1 / lambda_C ** 2))

        # Calculate A using the given formula
        A = nD - B / lambda_D ** 2

        # Calculate n for each wavelength in lambda_vec
        n = A + B / lambda_vec ** 2  # Note: lambda_vec is already in meters
        
        # print("lambda_vec = ",lambda_vec)
        # print("n_cauchy = ",n)
        # print("n sellemeir = ",self.sellmeir_NSF4(lambda_vec))


        # return self.sellmeir_NSF4(lambda_vec)
        return n
    
    def sellmeier(self,lambda_vec,glass_name):

        glass_pd = get_glass_catalog(self.catalog)
        df = glass_pd.df["dispersion coefficients"]

        B1 = df.loc[glass_name]["B1"]
        B2 = df.loc[glass_name]["B2"]
        B3 = df.loc[glass_name]["B3"]
        C1 = df.loc[glass_name]["C1"]
        C2 = df.loc[glass_name]["C2"]
        C3 = df.loc[glass_name]["C3"]

        # print(glass_name)
        # print("B1,B2,B3,C1,C2,C3 = ",B1,B2,B3,C1,C2,C3)

        lambda_vec = lambda_vec * 1e-3

        index_square = 1 + B1*lambda_vec**2/(lambda_vec**2-C1) + B2*lambda_vec**2/(lambda_vec**2-C2)

        # if index_square is not a tensor, convert it to a tensor
        if not isinstance(index_square, torch.Tensor):
            index_square = torch.tensor(index_square,dtype=torch.float32)

        index = torch.sqrt(index_square)

        return index


    def rotate_from_lens_to_dispersive_element(self,k,alpha_c,delta_alpha_c,delta_beta_c,alpha_1):
        angle_with_P1 = alpha_c - alpha_1 + delta_alpha_c
        #TODO : check the type of k
        k = k.to(dtype=torch.float32)
        # k_1 = rotation_y(angle_with_P1) @ k
        k = torch.matmul(k,rotation_y(angle_with_P1).T)
        # Rotation in relation to P1 around the X axis
        k = k.to(dtype=torch.float32)

        k = torch.matmul(k,rotation_x(delta_beta_c).T)
        # Rotation of P1 in relation to frame_in along the new Y axis
        k = k.to(dtype=torch.float32)

        k = torch.matmul(k,rotation_y(alpha_1).T)

        return k
    
    def rotate_from_dispersive_element_to_lens(self,k,alpha_c_transmis,delta_alpha_c,delta_beta_c,alpha_1):

        angle_with_P2 = alpha_c_transmis - alpha_1 - delta_alpha_c
        k = torch.matmul(k,rotation_y(alpha_1).T)
        # Rotation in relation to P2 around the X axis
        k = torch.matmul(k,rotation_x(-delta_beta_c).T)
        # Rotation in relation to P2 around the Y axis
        k = torch.matmul(k,rotation_y(angle_with_P2).T)
        #print(k.shape)
        return k



    def propagate_through_triple_prism(self,k,n1,n2,n3,A1,A2,A3):

        norm_k = torch.sqrt(k[...,0] ** 2 + k[...,1] ** 2 + k[...,2] ** 2)
        norm_k = norm_k.unsqueeze(-1)
        norm_k = norm_k.repeat(1, 1, 1,3)
        k_normalized = k / norm_k

        k,theta_in_1, theta_out_1,distance_from_total_intern_reflection1 = self.model_Prism_angle_to_angle_torch(k_normalized, n1, A1)
        k = k * norm_k
        k = torch.matmul(k,rotation_z(torch.tensor(np.pi,device=self.device)).T)
        k,theta_in_2, theta_out_2,distance_from_total_intern_reflection2 = self.model_Prism_angle_to_angle_torch(k, n2, A2)
        k = k * norm_k
        k = torch.matmul(k,rotation_z(torch.tensor(np.pi,device=self.device)).T)
        k,theta_in_3, theta_out_3,distance_from_total_intern_reflection3 = self.model_Prism_angle_to_angle_torch(k, n3, A3)
        k = k * norm_k

        list_theta_in = [theta_in_1,theta_in_2,theta_in_3]
        list_theta_out = [theta_out_1,theta_out_2,theta_out_3]

        self.min_distance_from_total_intern_reflection = min(torch.min(distance_from_total_intern_reflection1),
                                                             torch.min(distance_from_total_intern_reflection2),
                                                             torch.min(distance_from_total_intern_reflection3))

        return k, list_theta_in, list_theta_out



    
    def model_Grating_angle_to_angle(self, k_in, lba, m, G):
        """
        Model of the grating

        Args:
            k_in (torch.Tensor) : wave vector of the incident ray (shape = 3 x N)
            lba (torch.Tensor) : wavelengths (shape = N) -- in nm
            m (float) : diffraction order of the grating -- no units
            G (float) : lines density of the grating -- in lines/mm

        Returns:
            torch.Tensor: wave vector of the outgoing ray (shape = 3 x N)
        """


        alpha_in = torch.atan(k_in[...,0]) * torch.sqrt(1 + torch.tan(k_in[...,0])**2 + torch.tan(k_in[...,1])**2)
        beta_in = torch.atan(k_in[...,1]) * torch.sqrt(1 + torch.tan(k_in[...,0])**2 + torch.tan(k_in[...,1])**2)

        alpha_out = -1 * torch.asin(m * lba * 10**-9 * G * 10**3 - torch.sin(alpha_in))
        beta_out = beta_in

        k_out = torch.stack([
            torch.sin(alpha_out) * torch.cos(beta_out),
            torch.sin(beta_out) * torch.cos(alpha_out),
            torch.cos(alpha_out) * torch.cos(beta_out)
        ], dim=-1)

        return k_out


    def model_Lens_angle_to_position(self,k_in,F):
        """
        Model of the lens : angle to position

        Args:
            k_in (torch.tensor) : wave vector of the incident ray (shape = 3 x N)
            F (float) : focal length of the lens -- in um

        Returns:
            tuple: position in the image plane (X,Y) -- in um

        """

        alpha = torch.arctan(k_in[...,0] / k_in[...,2])
        beta = torch.arctan(k_in[...,1] / k_in[...,2])

        x = F * torch.tan(alpha)
        y = F * torch.tan(beta)

        return x, y

    def model_Prism_angle_to_angle(self,k0, n,A):
        """
        Ray tracing through the prism

        Args:
            k0 (numpy.ndarray) : wave vector of the incident ray (shape = 3 x N)
            n (numpy.ndarray) : refractive index of the prism (shape = N)
            A (float) : angle of the prism -- in radians

        Returns:
            numpy.ndarray: wave vector of the outgoing ray (shape = 3 x N)

        """

        kp = np.array([k0[0], k0[1], np.sqrt(n ** 2 - k0[0] ** 2 - k0[1] ** 2)])

        kp_r = np.matmul(rotation_y(-A), kp)

        kout = [kp_r[0], kp_r[1], np.sqrt(1 - kp_r[0] ** 2 - kp_r[1] ** 2)]

        return kout


    def model_Prism_angle_to_angle_torch(self,k0, n,A):
        """
        Ray tracing through the prism

        Args:
            k0 (torch.tensor) : wave vector of the incident ray
            n (torch.tensor) : refractive index of the prism
            A (float) : angle of the prism -- in radians

        Returns:
            torch.tensor: wave vector of the outgoing ray

        """
        kp = torch.zeros_like(k0,device=k0.device)
        kout = torch.zeros_like(k0,device=k0.device)

        theta_in = torch.atan2(k0[...,0], k0[...,2])

        kp[...,0] = k0[...,0]
        kp[...,1] = k0[...,1]

        # compare = n ** 2 - k0[...,0] ** 2 - k0[...,1] ** 2


        kp[...,2] = torch.sqrt(n ** 2 - k0[...,0] ** 2 - k0[...,1] ** 2)

        theta_out = torch.atan2(kp[...,0], kp[...,2])

        kp_r = torch.matmul(kp, rotation_y(-A).T)

        kout[...,0] = kp_r[...,0]
        kout[...,1] = kp_r[...,1]
        kout[...,2] = torch.sqrt(1 - kp_r[...,0] ** 2 - kp_r[...,1] ** 2)

        distance_from_total_intern_reflection = 1 - kp_r[...,0] ** 2 - kp_r[...,1] ** 2

        return kout, theta_in, theta_out,distance_from_total_intern_reflection
    

    def model_Lens_pos_to_angle(self,x_obj, y_obj, F):
        """
        Model of the lens : position to angle

        Args:
            x_obj (torch.tensor) : position X in the image plane -- in um
            y_obj (torch.tensor) : position Y in the image plane  -- in um
            F (torch.tensor) : focal length of the lens -- in um

        Returns:
            torch.tensor: wave vector of the outgoing ray

        """
        alpha = -1 * torch.atan(x_obj / F)
        beta = -1 * torch.atan(y_obj / F)

        # Adjusting the dimension for k_out to make it a 4D tensor
        #TODO : verify that float64 is the right type
        k_out = torch.zeros(size=(x_obj.shape[0], x_obj.shape[1], x_obj.shape[2], 3),dtype=torch.float64,device=x_obj.device)

        # the fourth dimension should have 3 components
        k_out[...,0] = torch.sin(alpha) * torch.cos(beta)
        k_out[...,1] = torch.sin(beta) * torch.cos(alpha)
        k_out[...,2] = torch.cos(alpha) * torch.cos(beta)

        return k_out



    def generate_2D_gaussian(self,radius, sample_size_x, sample_size_y, nb_of_samples):
        """
        Generate a 2D Gaussian of a given radius

        Args:
            radius (float): radius of the Gaussian
            sample_size_x (float): size of each sample along the X axis
            sample_size_y (float): size of each sample along the Y axis
            nb_of_samples (int): number of samples along each axis

        Returns:
            numpy.ndarray: 2D Gaussian shape array
        """

        # Define the grid
        grid_size_x = sample_size_x * (nb_of_samples - 1)
        grid_size_y = sample_size_y * (nb_of_samples - 1)
        x = np.linspace(-grid_size_x / 2, grid_size_x / 2, nb_of_samples)
        y = np.linspace(-grid_size_y / 2, grid_size_y / 2, nb_of_samples)
        X, Y = np.meshgrid(x, y)

        # Compute the 2D Gaussian function
        gaussian_2d = np.exp(-(X ** 2 + Y ** 2) / (2 * radius ** 2))

        return gaussian_2d
    
    def generate_psf(self, type, radius):
        """
        Generate a PSF

        Args:
            type (str): type of PSF to generate
            radius (float): radius of the PSF

        Returns:
            numpy.ndarray: PSF generated (shape = R x C)
        """

        if type == "Gaussian":
            PSF = self.generate_2D_gaussian(radius, self.system_config["detector"]["pixel size along X"],
                                             self.system_config["detector"]["pixel size along Y"], 10)
            self.psf = PSF

        return self.psf

    def check_if_sampling_is_sufficiant(self):
        """
        Check if the sampling is sufficiant to avoid aliasing.

        Returns:
            float: number of sample points per pixel

        """

        pix_size = self.system_config["detector"]["pixel size along X"]
        nb_of_system_wavelengths = self.system_wavelengths.shape[0]

        nb_of_sample_points_per_pix =  pix_size / (self.central_distorsion_in_X / nb_of_system_wavelengths )
        print("number of spectral sample points per pixel =",nb_of_sample_points_per_pix)

        if nb_of_sample_points_per_pix < 2:

            print("The 'number of spectral samples' (cf. system config) is not sufficiant to avoid weird sampling effect( aliasing ?). RAISE IT !")

        return nb_of_sample_points_per_pix
    
    

    def propagate_central_microm_through_disperser(self,lambda_):

       
        if self.index_estimation_method == "cauchy":
            n1 = self.calculate_dispersion_with_cauchy(lambda_,self.nd1,self.vd1)
            n2 = self.calculate_dispersion_with_cauchy(lambda_,self.nd2,self.vd2)
            n3 = self.calculate_dispersion_with_cauchy(lambda_,self.nd3,self.vd3)

        if self.index_estimation_method == "sellmeier":
            n1 = self.sellmeier(lambda_,self.glass1)
        
            try:
                n2 = self.sellmeier(lambda_,self.glass2)

            except:
                n2 = torch.tensor(1,device=self.device)
            try:
                n3 = self.sellmeier(lambda_,self.glass3)

            except:
                n3 = torch.tensor(1,device=self.device)

        x0 = torch.zeros((1,1,1,1)).to(device=self.device)
        y0 = torch.zeros((1,1,1,1)).to(device=self.device)

        k = self.model_Lens_pos_to_angle(x0, y0, self.F)

        alpha_1 = self.define_alpha_1()

        k = self.rotate_from_lens_to_dispersive_element(k,self.alpha_c,self.delta_alpha_c,self.delta_beta_c,alpha_1)

        if self.dispersive_element_type in ["prism", "doubleprism", "amici", "tripleprism"]:
            k, list_theta_in, list_theta_out = self.propagate_through_triple_prism(k,n1,n2,n3,self.A1,self.A2,self.A3)
            self.list_theta_in = list_theta_in
            self.list_theta_out = list_theta_out
        elif self.dispersive_element_type == "grating":
            k = self.model_Grating_angle_to_angle(k, lambda_, self.m, self.G)

        alpha = torch.arctan(k[...,0] / k[...,2])


        return alpha
    


    def define_alpha_1(self):
        """
        Define the angle alpha_1 depending on the type of dispersive element

        Returns:
            float: angle alpha_1
        """

        if self.dispersive_element_type == "prism":
            alpha_1 = self.A1 / 2
        elif self.dispersive_element_type in ["doubleprism", "amici", "tripleprism"]:
            alpha_1 = self.A1 - self.A2 / 2
        elif self.dispersive_element_type == "grating":
            alpha_1 = torch.tensor(0,device=self.device)
        else:
            raise Exception("dispersive_element_type should be prism or grating")

        return alpha_1
    


    


def rotation_z(theta):
    """
    Rotate 3D matrix around the Z axis using PyTorch

    Args:
        theta (torch.Tensor): Input angle (in rad)

    Returns:
        torch.Tensor: 3D rotation matrix
    """
    # Ensure theta is a tensor with requires_grad=True
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, requires_grad=True)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Construct the rotation matrix using torch.stack to support gradient computation
    # For rotation around the Z axis, the changes affect the first two rows
    row1 = torch.stack([cos_theta, -sin_theta, torch.zeros_like(theta)])
    row2 = torch.stack([sin_theta, cos_theta, torch.zeros_like(theta)])
    row3 = torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)])

    # Concatenate the rows to form the rotation matrix
    r = torch.stack([row1, row2, row3], dim=0)

    # Adjust the matrix to have the correct shape (3, 3) for each theta
    r = r.transpose(0, 1)  # This may need adjustment based on how you intend to use r

    return r


def rotation_y(theta):
    """
    Rotate 3D matrix around the Y axis using PyTorch

    Args:
        theta (torch.Tensor): Input angle (in rad)

    Returns:
        torch.Tensor: 3D rotation matrix
    """
    # Ensure theta is a tensor with requires_grad=True
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, requires_grad=True, dtype=torch.float32)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Construct the rotation matrix using torch.stack to support gradient computation
    # For rotation around the Y axis, the changes affect the first and third rows
    row1 = torch.stack([cos_theta, torch.zeros_like(theta), -sin_theta])  # Note the change to -sin_theta for correct Y-axis rotation
    row2 = torch.stack([torch.zeros_like(theta), torch.ones_like(theta), torch.zeros_like(theta)])
    row3 = torch.stack([sin_theta, torch.zeros_like(theta), cos_theta])

    # Concatenate the rows to form the rotation matrix
    r = torch.stack([row1, row2, row3], dim=0)

    # Adjust the matrix to have the correct shape (3, 3) for each theta
    r = r.transpose(0, 1)  # Adjust transpose for consistency with your requirements

    return r


def rotation_x(theta):
    """
    Rotate 3D matrix around the X axis using PyTorch

    Args:
        theta (tensor): Input angle (in rad)

    Returns:
        torch.Tensor: 3D rotation matrix
    """
    # Ensure theta is a tensor with requires_grad=True
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, requires_grad=True, dtype=torch.float32)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Use torch.stack and torch.cat to construct the rotation matrix
    row1 = torch.stack([torch.ones_like(theta), torch.zeros_like(theta), torch.zeros_like(theta)])
    row2 = torch.stack([torch.zeros_like(theta), cos_theta, -sin_theta])
    row3 = torch.stack([torch.zeros_like(theta), sin_theta, cos_theta])

    # Concatenate the rows to form the rotation matrix
    r = torch.stack([row1, row2, row3], dim=0)

    # Transpose the matrix to match the expected shape
    r = r.transpose(0, 1)

    return r
