import torch
import snoop


def rotation_z_torch(theta):
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


def rotation_y_torch(theta):
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


def rotation_x_torch(theta):
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

from simca.functions_general_purpose import *
# import snoop
import torch
from opticalglass.glassfactory import create_glass
import pytorch_lightning as pl

class OpticalModelTorch(pl.LightningModule):
    """
    Class that contains the optical model caracteristics and propagation models
    """
    def __init__(self, config):
        super().__init__()
        self.system_config = config
        self.dispersive_element_type = config["system architecture"]["dispersive element"]["type"]
        self.lba_c = torch.tensor(config["system architecture"]["dispersive element"]["wavelength center"])

        # Can be optimized 
        # 1 - with DE
        self.glass1 = create_glass(config["system architecture"]["dispersive element"]["glass1"], 'Schott')
        self.glass2 = create_glass(config["system architecture"]["dispersive element"]["glass2"], 'Schott')
        self.glass3 = create_glass(config["system architecture"]["dispersive element"]["glass3"], 'Schott')

        # - with any optimizer
        self.A1 = torch.tensor(math.radians(config["system architecture"]["dispersive element"]["A1"]),requires_grad=True)
        self.A2 = torch.tensor(math.radians(config["system architecture"]["dispersive element"]["A2"]),requires_grad=True)
        self.A3 = torch.tensor(math.radians(config["system architecture"]["dispersive element"]["A3"]),requires_grad=True)
        self.F = torch.tensor(config["system architecture"]["focal lens"])


        self.G = torch.tensor(config["system architecture"]["dispersive element"]["G"])
        self.m = torch.tensor(config["system architecture"]["dispersive element"]["m"])

        self.alpha_c = torch.tensor(math.radians(config["system architecture"]["dispersive element"]["alpha_c"]),requires_grad=True)
        self.delta_alpha_c = torch.tensor(math.radians(config["system architecture"]["dispersive element"]["delta alpha c"]))
        self.delta_beta_c = torch.tensor(math.radians(config["system architecture"]["dispersive element"]["delta beta c"]))

        self.continuous_glass_materials1 = config["system architecture"]["dispersive element"]["continuous glass materials 1"]
        self.continuous_glass_materials2 = config["system architecture"]["dispersive element"]["continuous glass materials 2"]
        self.continuous_glass_materials3 = config["system architecture"]["dispersive element"]["continuous glass materials 3"]

        self.nd1 = torch.tensor(config["system architecture"]["dispersive element"]["nd1"],requires_grad=True)
        self.vd1 = torch.tensor(config["system architecture"]["dispersive element"]["vd1"],requires_grad=True)
        self.nd2 = torch.tensor(config["system architecture"]["dispersive element"]["nd2"],requires_grad=True)
        self.vd2 = torch.tensor(config["system architecture"]["dispersive element"]["vd2"],requires_grad=True)
        self.nd3 = torch.tensor(config["system architecture"]["dispersive element"]["nd3"],requires_grad=True)
        self.vd3 = torch.tensor(config["system architecture"]["dispersive element"]["vd3"],requires_grad=True)

        alpha_c_transmis = -1*self.propagate_central_microm_through_disperser(self.lba_c)
        self.alpha_c_transmis = alpha_c_transmis

        print("Optica lmodel device : ",self.device)



    def rerun_central_dispersion(self):
        alpha_c_transmis = -1 * self.propagate_central_microm_through_disperser(self.lba_c)
        self.alpha_c_transmis = alpha_c_transmis

        return alpha_c_transmis

    def propagate(self,X,Y,lba,n1,n2,n3):

        k = self.model_Lens_pos_to_angle(X, Y, self.F)


        if self.dispersive_element_type == "prism":
            alpha_1 = self.A1 - self.A1/2
            k = self.rotate_from_lens_to_dispersive_element(k,self.alpha_c,self.delta_alpha_c,self.delta_beta_c,alpha_1)
            k, list_theta_in, list_theta_out = self.propagate_through_simple_prism(k,n1,self.A1)
            k = self.rotate_from_dispersive_element_to_lens(k,self.alpha_c_transmis,self.delta_alpha_c,self.delta_beta_c,alpha_1)

        elif self.dispersive_element_type == "doubleprism":
            alpha_1 = self.A1 - self.A2/2
            k = self.rotate_from_lens_to_dispersive_element(k,self.alpha_c,self.delta_alpha_c,self.delta_beta_c,alpha_1)
            k, list_theta_in, list_theta_out = self.propagate_through_double_prism(k,n1,n2,self.A1,self.A2)
            k = self.rotate_from_dispersive_element_to_lens(k,self.alpha_c_transmis,self.delta_alpha_c,self.delta_beta_c,alpha_1)

        elif self.dispersive_element_type == "tripleprism":
            alpha_1 = self.A1 - self.A2/2
            k = self.rotate_from_lens_to_dispersive_element(k,self.alpha_c,self.delta_alpha_c,self.delta_beta_c,alpha_1)
            k, list_theta_in, list_theta_out = self.propagate_through_triple_prism(k,n1,n2,n3,self.A1,self.A2,self.A3)
            k = self.rotate_from_dispersive_element_to_lens(k,self.alpha_c_transmis,self.delta_alpha_c,self.delta_beta_c,alpha_1)
        
        elif self.dispersive_element_type == "grating":
            alpha_1 = 0
            k = self.rotate_from_lens_to_dispersive_element(k,self.alpha_c,self.delta_alpha_c,self.delta_beta_c,alpha_1)
            k = self.model_Grating_angle_to_angle(k, lba, self.m, self.G)
            k = self.rotate_from_dispersive_element_to_lens(k,self.alpha_c_transmis,self.delta_alpha_c,self.delta_beta_c,alpha_1)

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

        return n

    def rotate_from_lens_to_dispersive_element(self,k,alpha_c,delta_alpha_c,delta_beta_c,alpha_1):


        angle_with_P1 = alpha_c - alpha_1 + delta_alpha_c
        k = k.to(dtype=torch.float32)
        # k_1 = rotation_y_torch(angle_with_P1) @ k
        k = torch.matmul(k,rotation_y_torch(angle_with_P1).T)
        # Rotation in relation to P1 around the X axis
        k = k.to(dtype=torch.float32)
        k = torch.matmul(k,rotation_x_torch(delta_beta_c).T)
        # Rotation of P1 in relation to frame_in along the new Y axis
        k = k.to(dtype=torch.float32)

        k = torch.matmul(k,rotation_y_torch(alpha_1).T)



        return k
    
    def rotate_from_dispersive_element_to_lens(self,k,alpha_c_transmis,delta_alpha_c,delta_beta_c,alpha_1):

        angle_with_P2 = alpha_c_transmis - alpha_1 - delta_alpha_c
        k = torch.matmul(k,rotation_y_torch(alpha_1).T)
        # Rotation in relation to P2 around the X axis
        k = torch.matmul(k,rotation_x_torch(-delta_beta_c).T)
        # Rotation in relation to P2 around the Y axis
        k = torch.matmul(k,rotation_y_torch(angle_with_P2).T)
        #print(k.shape)
        return k

    def propagate_through_simple_prism(self, k, n, A):
        norm_k = torch.sqrt(k[..., 0] ** 2 + k[..., 1] ** 2 + k[..., 2] ** 2)
        norm_k = norm_k.unsqueeze(-1)
        norm_k = norm_k.repeat(1, 1, 1, 3)
        # Use out-of-place operation instead of in-place
        k_normalized = k / norm_k

        k_updated, theta_in, theta_out,distance_from_total_intern_reflection = self.model_Prism_angle_to_angle_torch(k_normalized, n, A)
        # Apply the normalization factor to k_updated if necessary
        k_updated = k_updated * norm_k

        list_theta_in = [theta_in]
        list_theta_out = [theta_out]

        return k_updated, list_theta_in, list_theta_out
    
    def propagate_through_double_prism(self,k,n1,n2,A1,A2):

        norm_k = torch.sqrt(k[...,0] ** 2 + k[...,1] ** 2 + k[...,2] ** 2)
        norm_k = norm_k.unsqueeze(-1)
        norm_k = norm_k.repeat(1, 1, 1,3)
        k_normalized = k / norm_k

        k,theta_in_1, theta_out_1,distance_from_total_intern_reflection = self.model_Prism_angle_to_angle_torch(k_normalized, n1, A1)
        k = k * norm_k
        k = torch.matmul(k,rotation_z_torch(torch.tensor(np.pi)).T)
        k,theta_in_2, theta_out_2,distance_from_total_intern_reflection = self.model_Prism_angle_to_angle_torch(k, n2, A2)
        k = k * norm_k
        k = torch.matmul(k, rotation_z_torch(torch.tensor(np.pi)).T)

        list_theta_in = [theta_in_1,theta_in_2]
        list_theta_out = [theta_out_1,theta_out_2]

        return k,list_theta_in, list_theta_out



    def propagate_through_triple_prism(self,k,n1,n2,n3,A1,A2,A3):

        norm_k = torch.sqrt(k[...,0] ** 2 + k[...,1] ** 2 + k[...,2] ** 2)
        norm_k = norm_k.unsqueeze(-1)
        norm_k = norm_k.repeat(1, 1, 1,3)
        k_normalized = k / norm_k


        k,theta_in_1, theta_out_1,distance_from_total_intern_reflection1 = self.model_Prism_angle_to_angle_torch(k_normalized, n1, A1)
        k = k * norm_k
        k = torch.matmul(k,rotation_z_torch(torch.tensor(np.pi)).T)
        k,theta_in_2, theta_out_2,distance_from_total_intern_reflection2 = self.model_Prism_angle_to_angle_torch(k, n2, A2)
        k = k * norm_k
        k = torch.matmul(k,rotation_z_torch(torch.tensor(np.pi)).T)
        k,theta_in_3, theta_out_3,distance_from_total_intern_reflection3 = self.model_Prism_angle_to_angle_torch(k, n3, A3)
        k = k * norm_k

        list_theta_in = [theta_in_1,theta_in_2,theta_in_3]
        list_theta_out = [theta_out_1,theta_out_2,theta_out_3]

        self.min_distance_from_total_intern_reflection = min(torch.min(distance_from_total_intern_reflection1),
                                                             torch.min(distance_from_total_intern_reflection2),
                                                             torch.min(distance_from_total_intern_reflection3))

        return k, list_theta_in, list_theta_out



    def propagation_with_distorsions_torch(self, X_input_grid, Y_input_grid):
        """
        Propagate the coded aperture coded_aperture through one CASSI system

        Args:
            X_input_grid (numpy.ndarray): x coordinates grid
            Y_input_grid (numpy.ndarray): y coordinates grid

        Returns:
            tuple: X coordinates of the propagated coded aperture grids, Y coordinates of the propagated coded aperture grids
        """

        self.calculate_central_dispersion()

        X_input_grid = torch.from_numpy(X_input_grid) if isinstance(X_input_grid, np.ndarray) else X_input_grid
        Y_input_grid = torch.from_numpy(Y_input_grid) if isinstance(Y_input_grid, np.ndarray) else Y_input_grid
        wavelength_vec = torch.from_numpy(self.system_wavelengths) if isinstance(self.system_wavelengths, np.ndarray) else self.system_wavelengths

        X_input_grid_3D = X_input_grid[:,:,None].repeat(1, 1,self.nb_of_spectral_samples)
        Y_input_grid_3D = Y_input_grid[:,:,None].repeat(1, 1,self.nb_of_spectral_samples)
        lba_3D = wavelength_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        n_tensor = self.sellmeier_torch(lba_3D)

        X, Y = self.propagate_through_arm_torch(X_input_grid_3D,Y_input_grid_3D,n=n_tensor,lba=lba_3D)

        return X, Y

    def calculate_central_dispersion(self):
        """
        Calculate the dispersion related to the central pixel of the coded aperture

        Returns:
            numpy.float: spectral dispersion of the central pixel of the coded aperture
        """

        self.alpha_c = self.calculate_alpha_c()

        X0_coordinates_array_flatten = np.zeros(self.system_wavelengths.shape[0])
        Y0_coordinates_array_flatten = np.zeros(self.system_wavelengths.shape[0])
        lba_array_flatten = self.system_wavelengths

        n_array_flatten = np.full(lba_array_flatten.shape, self.sellmeier(lba_array_flatten))

        X0_propagated, Y0_propagated = self.propagate_through_arm(X_vec_in=X0_coordinates_array_flatten,Y_vec_in=Y0_coordinates_array_flatten,n=n_array_flatten,lba=lba_array_flatten)

        self.X0_propagated, self.Y0_propagated = X0_propagated, Y0_propagated

        self.central_distorsion_in_X = np.abs(self.X0_propagated[-1] - self.X0_propagated[0])

        return self.central_distorsion_in_X

    def propagate_through_arm(self, X_vec_in, Y_vec_in, n, lba):

        """
        Propagate the light through one system arm : (lens + dispersive element + lens)

        Args:
            X_vec_in (numpy.ndarray) : X coordinates of the coded aperture pixels (1D array)
            Y_vec_in (numpy.ndarray) : Y coordinates of the coded aperture pixels (1D array)
            n (numpy.ndarray) : refractive indexes of the system (at the corresponding wavelength)
            lba (numpy.ndarray) : wavelengths

        Returns:
            tuple: flatten arrays corresponding to the propagated X and Y coordinates
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

            X_vec_out, Y_vec_out = self.model_Lens_angle_to_position(k_1_bis, F)


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

            X_vec_out, Y_vec_out = self.model_Lens_angle_to_position(k_1_bis, F)

        else:
            raise Exception("dispersive_element_type should be prism or grating")

        return X_vec_out, Y_vec_out
    

    def propagate_through_arm_torch(self, X_tensor_in, Y_tensor_in, n, lba):

        """
        Propagate the light through one system arm : (lens + dispersive element + lens)

        Args:
            X_tensor_in (torch.tensor) : X coordinates of the coded aperture pixels (3D array)
            Y_tensor_in (torch.tensor) : Y coordinates of the coded aperture pixels (3D array)
            n (torch.tensor) : refractive indexes of the system (at the corresponding wavelength)
            lba (torch.tensor) : wavelengths

        Returns:
            tuple: tensors corresponding to the propagated X and Y coordinates
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

        alpha_c =   torch.tensor(alpha_c, dtype=torch.float64)
        A = torch.tensor(A, dtype=torch.float64)
        delta_alpha_c = torch.tensor(delta_alpha_c, dtype=torch.float64)
        alpha_c_transmis = torch.tensor(alpha_c_transmis, dtype=torch.float64)
        delta_beta_c = torch.tensor(delta_beta_c, dtype=torch.float64)




        if dispersive_element_type == "prism":

            angle_with_P1 = alpha_c - A / 2 + delta_alpha_c
            angle_with_P2 = alpha_c_transmis - A / 2 - delta_alpha_c

            # THERE IS A PROBLEM HERE ???
            k = self.model_Lens_pos_to_angle_torch(X_tensor_in, Y_tensor_in, F)
            # Rotation in relation to P1 around the Y axis

            # k_1 = rotation_y_torch(angle_with_P1) @ k
            k_1 = torch.matmul(k,rotation_y_torch(angle_with_P1).T)
            # Rotation in relation to P1 around the X axis
            k_2 = torch.matmul(k_1,rotation_x_torch(delta_beta_c).T)
            # Rotation of P1 in relation to frame_in along the new Y axis
            k_3 = torch.matmul(k_2,rotation_y_torch(A / 2).T)

            norm_k = torch.sqrt(k_3[...,0] ** 2 + k_3[...,1] ** 2 + k_3[...,2] ** 2)
            norm_k = norm_k.unsqueeze(-1)
            norm_k = norm_k.repeat(1, 1, 1,3)
            k_3 /= norm_k

            k_out_p = self.model_Prism_angle_to_angle_torch(k_3, n, A)
            k_out_p = k_out_p * norm_k

            k_3_bis = torch.matmul(k_out_p,rotation_y_torch(A / 2).T)

            # Rotation in relation to P2 around the X axis
            k_2_bis = torch.matmul(k_3_bis,rotation_x_torch(-delta_beta_c).T)
            # Rotation in relation to P2 around the Y axis
            k_1_bis = torch.matmul(k_2_bis,rotation_y_torch(angle_with_P2).T)

            X_vec_out, Y_vec_out = self.model_Lens_angle_to_position_torch(k_1_bis, F)


        # elif dispersive_element_type == "grating":
        #
        #     angle_with_P1 = alpha_c - delta_alpha_c
        #     angle_with_P2 = alpha_c_transmis + delta_alpha_c
        #
        #     k = self.model_Lens_pos_to_angle(X_vec_in, Y_vec_in, F)
        #     # Rotation in relation to P1 around the Y axis
        #
        #     k_1 = rotation_y(angle_with_P1) @ k[:, 0, :]
        #     # Rotation in relation to P1 around the X axis
        #     k_2 = rotation_x(delta_beta_c) @ k_1
        #
        #     k_3 = rotation_y(0) @ k_2
        #     norm_k = np.sqrt(k_3[0] ** 2 + k_3[1] ** 2 + k_3[2] ** 2)
        #     k_3 /= norm_k
        #
        #     k_out_p = self.model_Grating_angle_to_angle(k_3, lba, m, G)
        #     k_out_p = k_out_p * norm_k
        #
        #     k_3_bis = np.dot(rotation_y(0), k_out_p)
        #
        #     # Rotation in relation to P2 around the X axis
        #     k_2_bis = np.dot(rotation_x(-delta_beta_c), k_3_bis)
        #     # Rotation in relation to P2 around the Y axis
        #     k_1_bis = np.dot(rotation_y(angle_with_P2), k_2_bis)
        #
        #     X_vec_out, Y_vec_out = self.model_Lens_angle_to_position(k_1_bis, F)
        #
        # else:
        #     raise Exception("dispersive_element_type should be prism or grating")
        #
        return X_vec_out, Y_vec_out

    def model_Grating_angle_to_angle(self,k_in, lba, m, G):
        """
        Model of the grating

        Args:
            k_in (numpy.ndarray) : wave vector of the incident ray (shape = 3 x N)
            lba (numpy.ndarray) : wavelengths (shape = N) -- in nm
            m (float) : diffraction order of the grating -- no units
            G (float) : lines density of the grating -- in lines/mm

        Returns:
            numpy.ndarray: wave vector of the outgoing ray (shape = 3 x N)

        """

        alpha_in = np.arctan(k_in[0]) * np.sqrt(1 + np.tan(k_in[0])**2 + np.tan(k_in[1])**2)
        beta_in = np.arctan(k_in[1]) * np.sqrt(1 + np.tan(k_in[0])**2 + np.tan(k_in[1])**2)

        alpha_out = -1*np.arcsin(m * lba*10**-9  * G * 10**3 - np.sin(alpha_in))
        beta_out = beta_in


        k_out = [np.sin(alpha_out) * np.cos(beta_out),
                 np.sin(beta_out)*np.cos(alpha_out),
                 np.cos(alpha_out) * np.cos(beta_out)]

        return k_out

    def simplified_grating_in_out(self, alpha,lba,m,G):
        """
        Model 1D of the grating in the dispersion direction

        Args:
            alpha (numpy.ndarray or float) : angle of the incident ray (shape = N) -- in radians
            lba (numpy.ndarray or float) : wavelengths (shape = N) -- in nm
            m (float) : diffraction order of the grating -- no units
            G (float) : lines density of the grating -- in lines/mm

        Returns:
            numpy.ndarray: angle of the outgoing ray (shape = N) -- in radians

        """

        alpha_out = np.arcsin(m * lba * 10 ** -9 * G * 10 ** 3 - np.sin(alpha))

        return alpha_out

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

        print("k0",k0.get_device())
        print("n",n.get_device())

        kp[...,2] = torch.sqrt(n ** 2 - k0[...,0] ** 2 - k0[...,1] ** 2)

        theta_out = torch.atan2(kp[...,0], kp[...,2])

        kp_r = torch.matmul(kp, rotation_y_torch(-A).T)

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
        k_out = torch.zeros(size=(x_obj.shape[0], x_obj.shape[1], x_obj.shape[2], 3),dtype=torch.float64,device=x_obj.device)

        # the fourth dimension should have 3 components
        k_out[...,0] = torch.sin(alpha) * torch.cos(beta)
        k_out[...,1] = torch.sin(beta) * torch.cos(alpha)
        k_out[...,2] = torch.cos(alpha) * torch.cos(beta)

        # k_out = torch.cat([
        #     sin_alpha * cos_beta,
        #     sin_beta * cos_alpha,
        #     cos_alpha * cos_beta
        # ], dim=-1)

        return k_out

    def calculate_alpha_c(self):
        """
        Calculate the relative angle of incidence between the lenses and the dispersive element

        Returns:
            float: angle of incidence
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
            n (float or numpy.ndarray): index of the prism -- no units
            A (float): apex angle of the prism -- in radians

        Returns:
            float or numpy.ndarray: minimum deviation angle -- in radians
        """
        return 2 * np.arcsin(n * np.sin(A / 2)) - A

    def get_incident_angle_min_dev(self,A, D_m):
        """
        Calculate the angle of incidence corresponding to the minimum deviation angle

        Args:
            A (float): apex angle of the prism -- in radians
            D_m (float): minimum deviation angle -- in radians

        Returns:
            float: angle of incidence corresponding to minimum of deviation -- in radians
        """
        return (A + D_m) / 2


    def sellmeier(self,lambda_, glass_type="BK7"):
        """
        Evaluating the refractive index value of a prism for a given lambda based on Sellmeier equation

        Args:
            lambda_ (numpy.ndarray of float) : wavelength in nm

        Returns:
            numpy.ndarray of float: index value corresponding to the input wavelength

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

    def sellmeier_torch(self,lambda_, glass_type="BK7"):
        """
        Evaluating the refractive index value of a prism for a given lambda based on Sellmeier equation

        Args:
            lambda_ (numpy.ndarray of float) : wavelength in nm

        Returns:
            numpy.ndarray of float: index value corresponding to the input wavelength

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

        n = torch.sqrt(1 + B1 * lambda_in_mm ** 2 / (lambda_in_mm ** 2 - C1) + B2 * lambda_in_mm ** 2 / (
                    lambda_in_mm ** 2 - C2) + B3 * lambda_in_mm ** 2 / (lambda_in_mm ** 2 - C3))

        return n

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

        if self.continuous_glass_materials1:
            n1 = self.calculate_dispersion_with_cauchy(lambda_,self.nd1,self.vd1)
        else:
            n1 = self.glass1.calc_rindex(lambda_)
        if self.continuous_glass_materials2:
            n2 = self.calculate_dispersion_with_cauchy(lambda_,self.nd2,self.vd2)
        else:
            n2 = self.glass2.calc_rindex(lambda_)
        if self.continuous_glass_materials3:
            n3 = self.calculate_dispersion_with_cauchy(lambda_,self.nd3,self.vd3)
        else:
            n3 = self.glass3.calc_rindex(lambda_)

        # n1 = 1.5
        # n2 = 1.8
        # n3 = 1.5

        x0 = torch.zeros((1,1,1,1))
        y0 = torch.zeros((1,1,1,1))

        k = self.model_Lens_pos_to_angle(x0, y0, self.F)

        # torch.autograd.set_detect_anomaly(True)

        if self.dispersive_element_type == "prism":

            alpha_1 = self.A1 - self.A1/2
            # alpha_1.backward(retain_graph=True)  # This requires alpha_1 to be a scalar
            # print("A1 requires grd",self.A1.grad)
            k = self.rotate_from_lens_to_dispersive_element(k,self.alpha_c,self.delta_alpha_c,self.delta_beta_c,alpha_1)
            # k.backward(torch.ones_like(k), retain_graph=True)
            # print("A1 requires grd", self.A1.grad)
            k, list_theta_in, list_theta_out = self.propagate_through_simple_prism(k,n1,self.A1)
            # k.backward(torch.ones_like(k), retain_graph=True)
            # print("A1 requires grd", self.A1.grad)

        elif self.dispersive_element_type == "doubleprism":
            alpha_1 = self.A1 - self.A2/2
            k = self.rotate_from_lens_to_dispersive_element(k,self.alpha_c,self.delta_alpha_c,self.delta_beta_c,alpha_1)
            k, list_theta_in, list_theta_out = self.propagate_through_double_prism(k,n1,n2,self.A1,self.A2)

        elif self.dispersive_element_type == "tripleprism":
            alpha_1 = self.A1 - self.A2/2
            k = self.rotate_from_lens_to_dispersive_element(k,self.alpha_c,self.delta_alpha_c,self.delta_beta_c,alpha_1)
            k, list_theta_in, list_theta_out = self.propagate_through_triple_prism(k,n1,n2,n3,self.A1,self.A2,self.A3)

        elif self.dispersive_element_type == "grating":
            alpha_1 = 0
            k = self.rotate_from_lens_to_dispersive_element(k,self.alpha_c,self.delta_alpha_c,self.delta_beta_c,alpha_1)
            k = self.model_Grating_angle_to_angle(k, lambda_, self.m, self.G)

        alpha = torch.arctan(k[...,0] / k[...,2])
        beta = torch.arctan(k[...,1] / k[...,2])
        #
        # alpha.backward(retain_graph=True)  # This requires alpha_1 to be a scalar
        # print("A1 requires grd", self.A1.grad)

        self.list_theta_in = list_theta_in
        self.list_theta_out = list_theta_out

        return alpha
