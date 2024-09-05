from simca.OpticalModelTorch import OpticalModelTorch
from simca.functions_acquisition import *
from simca.functions_patterns_generation import *
from simca.functions_scenes import *
from simca.functions_general_purpose import *
from simca.functions_acquisition_torch import *
import pytorch_lightning as pl
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn


class CassiSystemOptim(pl.LightningModule):
    """Class that contains the cassi system main attributes and methods"""

    def __init__(self, system_config=None,index_estimation_method="cauchy",device="cuda"):

        # Initialize LightningModule
        pl.LightningModule.__init__(self)
        self.to(device)

        self.system_config = system_config

        self.X_coded_aper_coordinates, self.Y_coded_aper_coordinates = self.generate_grid_coordinates(grid_name="coded aperture")
        self.X_detector_coordinates_grid, self.Y_detector_coordinates_grid = self.generate_grid_coordinates(grid_name="detector")
                
        self.wavelengths, self.nb_of_spectral_samples = self.generate_wavelengths()

        self.optical_model = OpticalModelTorch(self.system_config,index_estimation_method=index_estimation_method,device=self.device)



    def generate_grid_coordinates(self, grid_name):
         
        X_coordinates, Y_coordinates = self.create_coordinates_grid(
                self.system_config[f"{grid_name}"]["number of pixels along X"],
                self.system_config[f"{grid_name}"]["number of pixels along Y"],
                self.system_config[f"{grid_name}"]["pixel size along X"],
                self.system_config[f"{grid_name}"]["pixel size along Y"])


        return X_coordinates, Y_coordinates
    
    def generate_wavelengths(self):
        wavelengths = torch.linspace(self.system_config["spectral range"]["wavelength min"],
                                          self.system_config["spectral range"]["wavelength max"],
                                          self.system_config["spectral range"]["number of spectral samples"])
        wavelengths = wavelengths.to(self.device)
        nb_of_spectral_samples = wavelengths.shape[0]
        
        return wavelengths, nb_of_spectral_samples
        

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

        return torch.from_numpy(X_input_grid).float().to(self.device), torch.from_numpy(Y_input_grid).float().to(self.device)


    
    def update_optical_model(self, system_config=None):
        """
        Update the optical model of the system

        Args:
            system_config (dict): configuration of the system

        Returns:
            None
        """
        if system_config is not None:
            self.system_config = system_config
        self.optical_model = OpticalModelTorch(self.system_config)

    def propagate_coded_aperture_grid(self):

        # if self.device == "cuda":
        #     wavelengths = self.wavelengths.cpu()
        # else:
        #     wavelengths = self.wavelengths
        if self.optical_model.index_estimation_method == "cauchy":
            n1 = self.optical_model.calculate_dispersion_with_cauchy(self.wavelengths,self.optical_model.nd1,self.optical_model.vd1)
            n2 = self.optical_model.calculate_dispersion_with_cauchy(self.wavelengths,self.optical_model.nd2,self.optical_model.vd2)
            n3 = self.optical_model.calculate_dispersion_with_cauchy(self.wavelengths,self.optical_model.nd3,self.optical_model.vd3)
        if self.optical_model.index_estimation_method == "sellmeier":
            n1 = self.optical_model.sellmeier(self.wavelengths,self.optical_model.glass1).clone().detach().requires_grad_(True).to(device=self.device)
            try:
                n2 = self.optical_model.sellmeier(self.wavelengths,self.optical_model.glass2).clone().detach().requires_grad_(True).to(device=self.device)
            except:
                n2 = torch.tensor(1,device=self.device).expand_as(self.wavelengths)
            try:
                n3 = self.optical_model.sellmeier(self.wavelengths,self.optical_model.glass3).clone().detach().requires_grad_(True).to(device=self.device)
            except:
                n3 = torch.tensor(1,device=self.device).expand_as(self.wavelengths)


        n1 = n1.to(self.device)
        n2 = n2.to(self.device)
        n3 = n3.to(self.device)


        X_input_grid = torch.from_numpy(self.X_coded_aper_coordinates) if isinstance(self.X_coded_aper_coordinates, np.ndarray) else self.X_coded_aper_coordinates
        Y_input_grid = torch.from_numpy(self.Y_coded_aper_coordinates) if isinstance(self.Y_coded_aper_coordinates, np.ndarray) else self.Y_coded_aper_coordinates
        wavelength_vec = torch.from_numpy(self.wavelengths) if isinstance(self.wavelengths, np.ndarray) else self.wavelengths
        n1_vec = torch.from_numpy(n1) if isinstance(n1, np.ndarray) else n1
        n2_vec = torch.from_numpy(n2) if isinstance(n2, np.ndarray) else n2
        n3_vec = torch.from_numpy(n3) if isinstance(n3, np.ndarray) else n3

        X_input_grid_3D = X_input_grid[:,:,None].repeat(1, 1,self.nb_of_spectral_samples).to(self.device)
        Y_input_grid_3D = Y_input_grid[:,:,None].repeat(1, 1,self.nb_of_spectral_samples).to(self.device)
        lba_3D = wavelength_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1).to(self.device)
        n1_3D = n1_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        n2_3D = n2_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        n3_3D = n3_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        

        self.X_coordinates_propagated_coded_aperture, self.Y_coordinates_propagated_coded_aperture = self.optical_model.propagate(X_input_grid_3D, Y_input_grid_3D, lba_3D, n1_3D, n2_3D, n3_3D)

        return self.X_coordinates_propagated_coded_aperture, self.Y_coordinates_propagated_coded_aperture

    def retropropagate_coded_aperture_grid(self):

        n1 = self.optical_model.calculate_dispersion_with_cauchy(self.wavelengths ,self.optical_model.nd1,self.optical_model.vd1).to(self.device)
        n2 = self.optical_model.calculate_dispersion_with_cauchy(self.wavelengths ,self.optical_model.nd2,self.optical_model.vd2).to(self.device)
        n3 = self.optical_model.calculate_dispersion_with_cauchy(self.wavelengths ,self.optical_model.nd3,self.optical_model.vd3).to(self.device)

        X_input_grid = torch.from_numpy(self.X_detector_coordinates_grid) if isinstance(self.X_detector_coordinates_grid, np.ndarray) else self.X_detector_coordinates_grid
        Y_input_grid = torch.from_numpy(self.Y_detector_coordinates_grid) if isinstance(self.Y_detector_coordinates_grid, np.ndarray) else self.Y_detector_coordinates_grid
        wavelength_vec = torch.from_numpy(self.wavelengths) if isinstance(self.wavelengths, np.ndarray) else self.wavelengths
        n1_vec = torch.from_numpy(n1) if isinstance(n1, np.ndarray) else n1
        n2_vec = torch.from_numpy(n2) if isinstance(n2, np.ndarray) else n2
        n3_vec = torch.from_numpy(n3) if isinstance(n3, np.ndarray) else n3

        X_input_grid_3D = X_input_grid[:,:,None].repeat(1, 1,self.nb_of_spectral_samples).to(self.device)
        Y_input_grid_3D = Y_input_grid[:,:,None].repeat(1, 1,self.nb_of_spectral_samples).to(self.device)
        lba_3D = wavelength_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1).to(self.device)
        n1_3D = n1_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        n2_3D = n2_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        n3_3D = n3_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        

        self.retro_X_detect_coords, self.retro_Y_detect_coords = self.optical_model.repropagate(X_input_grid_3D, Y_input_grid_3D, lba_3D, n1_3D, n2_3D, n3_3D)

        return self.retro_X_detect_coords, self.retro_Y_detect_coords
    
    
    def generate_filtering_cube(self):
        """
        Generate filtering cube : each slice of the cube is a propagated pattern interpolated on the detector grid

        Returns:
        numpy.ndarray: filtering cube generated according to the optical system & the pattern configuration (R x C x W)

        """

        self.filtering_cube = interpolate_data_on_grid_positions_torch(data=self.pattern.unsqueeze(-1).repeat(1, 1, 1, self.wavelengths.shape[0]).to(self.device),
                                                                X_init=self.X_coordinates_propagated_coded_aperture,
                                                                Y_init=self.Y_coordinates_propagated_coded_aperture,
                                                                X_target=self.X_detector_coordinates_grid,
                                                                Y_target=self.Y_detector_coordinates_grid).to(self.device)

        return self.filtering_cube

    def get_displacement_in_pixels(self,dataset_wavelengths):

        central_coordinates_X = self.X_coordinates_propagated_coded_aperture[0,self.X_coordinates_propagated_coded_aperture.shape[1] // 2, self.X_coordinates_propagated_coded_aperture.shape[2] // 2, :]
        displacement_in_pix = [float(coord/self.system_config["detector"]["pixel size along X"]) for coord in list(central_coordinates_X)]

        current_wavelengths = self.wavelengths.cpu().numpy()
        # Interpolate displacement values onto dataset_wavelengths
        interpolated_displacement_in_pix = np.interp(dataset_wavelengths.cpu().numpy(), current_wavelengths, displacement_in_pix)

        return interpolated_displacement_in_pix
    

    def interpolate_and_crop_scene(self,cube, wavelengths,chunk_size=256):

        min_X_idx, max_X_idx, min_Y_idx, max_Y_idx = self.find_englobing_indices(self.retro_X_detect_coords, 
                                                                                 self.retro_Y_detect_coords, 
                                                                                 self.X_coded_aper_coordinates, 
                                                                                 self.Y_coded_aper_coordinates)

        
        X_cube_grid = torch.linspace(self.X_coded_aper_coordinates[0,min_X_idx],self.X_coded_aper_coordinates[0,max_X_idx],cube.shape[2])
        X_cube_grid_step = X_cube_grid[1] - X_cube_grid[0]
        Y_cube_grid = [X_cube_grid_step*i for i in range(cube.shape[1])]
        Y_cube_grid = [coord - Y_cube_grid[-1]/2 for coord in Y_cube_grid]
        Y_cube_grid = torch.tensor(Y_cube_grid)

        target_grid_X = self.X_coded_aper_coordinates[0,min_X_idx:max_X_idx]/torch.max(torch.abs(X_cube_grid))
        target_grid_Y = self.Y_coded_aper_coordinates[min_Y_idx:max_Y_idx,0]/torch.max(torch.abs(Y_cube_grid))
        target_grid_lambda = self.wavelengths - torch.min(self.wavelengths)
        target_grid_lambda = (target_grid_lambda/torch.max(target_grid_lambda))*2 -1

        target_grids = (target_grid_lambda.to(self.device), 
                        target_grid_Y.to(self.device), 
                        target_grid_lambda.to(self.device))

        cube = cube.to(self.device)

        grid_lambda, grid_Y, grid_X = torch.meshgrid(target_grid_lambda, target_grid_Y, target_grid_X, indexing='ij')
        target_grids = torch.stack((grid_X, grid_Y,grid_lambda), dim=-1).to(self.device)  # (D_prime, H_prime, W_prime, 3)

        if cube.ndim == 3:
            cube = cube.unsqueeze(0)

        interpolated_cube = self.interpolate_data_3D_in_chunks(cube, target_grids, chunk_size)

        # if cube.ndim == 3:
        #     interpolated_cube = interpolated_cube.squeeze(0)

        return interpolated_cube
    

    def image_acquisition(self, hyperspectral_cube, pattern,wavelengths,use_psf=False, chunck_size=50):
        """
        Run the acquisition/measurement process depending on the cassi system type

        Args:
            chunck_size (int): default block size for the interpolation

        Returns:
            numpy.ndarray: compressed measurement (R x C)
        """
        self.wavelengths= self.wavelengths.to(self.device)

        dataset = self.interpolate_dataset_along_wavelengths_torch(hyperspectral_cube, wavelengths,self.wavelengths, chunck_size)

        if dataset is None:
            return None

        try:
            dataset_labels = self.dataset_labels
        except:
            dataset_labels = None

        if self.system_config["system architecture"]["system type"] == "DD-CASSI":

            try:
                self.filtering_cube
            except:
                return print("Please generate filtering cube first")


            scene = match_dataset_to_instrument(dataset, self.filtering_cube[0,:,:,0])

            # scene = torch.from_numpy(match_dataset_to_instrument(dataset, self.filtering_cube)).to(self.device) if isinstance(scene, np.ndarray) else scene.to(self.device)

            measurement_in_3D = generate_dd_measurement_torch(scene, self.filtering_cube, chunck_size)

            self.last_filtered_interpolated_scene = measurement_in_3D
            self.interpolated_scene = scene

            if dataset_labels is not None:
                scene_labels = torch.from_numpy(match_dataset_labels_to_instrument(dataset_labels, self.filtering_cube)) 
                self.scene_labels = scene_labels


        elif self.system_config["system architecture"]["system type"] == "SD-CASSI":

            X_coded_aper_coordinates_crop = crop_center(self.X_coded_aper_coordinates,dataset.shape[1], dataset.shape[2])
            Y_coded_aper_coordinates_crop = crop_center(self.Y_coded_aper_coordinates,dataset.shape[1], dataset.shape[2])

            self.X_coded_aper_coordinates = X_coded_aper_coordinates_crop
            self.Y_coded_aper_coordinates = Y_coded_aper_coordinates_crop

            # print("dataset shape: ", dataset.shape)
            # print("X coded shape: ", X_coded_aper_coordinates_crop.shape)


            scene = match_dataset_to_instrument(dataset, X_coded_aper_coordinates_crop)

            pattern_crop = crop_center_3D(pattern, scene.shape[2], scene.shape[1]).to(self.device)
            
            self.pattern_crop = pattern_crop
            
            pattern_crop = pattern_crop.unsqueeze(-1).repeat(1, 1, 1, scene.size(-1))

            #print(scene.get_device())
            #print(pattern_crop.get_device())

            plt.imshow(scene[0,:,:,0].cpu().numpy())
            plt.title("scene")
            plt.show()

            # filtered_scene = scene * pattern_crop[..., None].repeat((1, 1, scene.shape[2]))
            # print(f"scene: {scene.shape}")
            # print(f"pattern_crop: {pattern_crop.shape}")
            filtered_scene = scene * pattern_crop


            self.propagate_coded_aperture_grid()


            sd_measurement = interpolate_data_on_grid_positions_torch(filtered_scene,
                                                                self.X_coordinates_propagated_coded_aperture,
                                                                self.Y_coordinates_propagated_coded_aperture,
                                                                self.X_detector_coordinates_grid,
                                                                self.Y_detector_coordinates_grid)
            
            self.filtering_cube = interpolate_data_on_grid_positions_torch(pattern_crop,
                                                                self.X_coordinates_propagated_coded_aperture,
                                                                self.Y_coordinates_propagated_coded_aperture,
                                                                self.X_detector_coordinates_grid,
                                                                self.Y_detector_coordinates_grid)

            self.last_filtered_interpolated_scene = sd_measurement
            self.interpolated_scene = scene


            if dataset_labels is not None:
                scene_labels = torch.from_numpy(match_dataset_labels_to_instrument(dataset_labels, self.last_filtered_interpolated_scene))
                self.scene_labels = scene_labels

        self.panchro = torch.sum(self.interpolated_scene, dim=3)

        if use_psf:
            self.apply_psf_torch()
        else:
            print("")
            #print("No PSF was applied")

        # Calculate the other two arrays
        self.measurement = torch.sum(self.last_filtered_interpolated_scene, dim=3)

        return self.measurement
    
    def find_englobing_indices(self,X_propagated, Y_propagated, X_fixed, Y_fixed):

        # Find the extreme values in X_out and Y_out
        min_X_propagated, max_X_propagated = X_propagated.min().item(), X_propagated.max().item()
        min_Y_propagated, max_Y_propagated = Y_propagated.min().item(), Y_propagated.max().item()


        min_X_idx = (X_fixed >= min_X_propagated).nonzero(as_tuple=True)[1].min().item() -1
        max_X_idx = (X_fixed <= max_X_propagated).nonzero(as_tuple=True)[1].max().item() +1

        min_Y_idx = (Y_fixed >= min_Y_propagated).nonzero(as_tuple=True)[0].min().item() -1
        max_Y_idx = (Y_fixed <= max_Y_propagated).nonzero(as_tuple=True)[0].max().item() +1

        if (min_X_idx - max_X_idx) % 2 != 0:
            min_X_idx -= 1
        
        if (min_Y_idx - max_Y_idx) % 2 != 0:
            min_Y_idx -= 1

        return min_X_idx, max_X_idx, min_Y_idx, max_Y_idx
    


    def interpolate_chunk(self,input_chunk, grid_chunk):
        """ Interpolate a chunk of data using grid_sample """
        # Add batch and channel dimensions
        input_chunk = input_chunk.unsqueeze(0)
        grid_chunk = grid_chunk.unsqueeze(0)
        # Perform interpolation

        interpolated_chunk = F.grid_sample(input_chunk, grid_chunk, align_corners=True)
        # Remove batch and channel dimensions

        interpolated_chunk = interpolated_chunk.squeeze(1)

        return interpolated_chunk

    def interpolate_data_3D_in_chunks(self,input_data, target_grids, chunk_size):
        # Get input dimensions
        batch_size,lambda_dim, Y_dim, X_dim = input_data.shape
        
        lambda_prime, Y_prime, X_prime,_= target_grids.shape[0:]

        print(lambda_prime, Y_prime, X_prime)
        print(batch_size,lambda_dim, Y_dim, X_dim)
        
        # Initialize the result tensor
        interpolated_data = torch.zeros((batch_size,lambda_prime, Y_prime, X_prime), dtype=input_data.dtype, device=input_data.device)
        # Iterate over the chunks
        # for i in range(0, lambda_dim, chunk_size):
        for j in range(0, Y_prime, chunk_size):
            for k in range(0, X_prime, chunk_size):
                # Define the chunk ranges
                j_end = min(j + chunk_size, Y_prime)
                k_end = min(k + chunk_size, X_prime)

                # Extract the input chunk
                
                # Create a corresponding target grid chunk
                grid_chunk = target_grids[:,j:j_end, k:k_end,:]

                # Interpolate the chunk
                interpolated_chunk = self.interpolate_chunk(input_data, grid_chunk)


                # Place the interpolated chunk in the result tensor
                interpolated_data[:, :,j:j_end, k:k_end] = interpolated_chunk
        
        return interpolated_data




     
    def generate_2D_pattern(self, config_pattern, nb_of_patterns=1):
        """
        Generate multiple coded aperture 2D patterns based on the "pattern" configuration file
        and stack them to match the desired number of patterns.

        Args:
            config_pattern (dict): coded-aperture pattern configuration.
            nb_of_patterns (int): Number of patterns to generate.

        Returns:
            torch.Tensor: Stacked coded-aperture 2D patterns (shape = nb_of_patterns x H x L).
        """

        pattern_list = []  # List to hold individual patterns

        for _ in range(nb_of_patterns):
            pattern_type = config_pattern['pattern']['type']


            if pattern_type == "random":
                pattern = generate_random_pattern((self.system_config["coded aperture"]["number of pixels along Y"],
                                                   self.system_config["coded aperture"]["number of pixels along X"]),
                                                  config_pattern['pattern']['ROM'])
            elif pattern_type == "ones":
                pattern = generate_ones_pattern((self.system_config["coded aperture"]["number of pixels along Y"],
                                                self.system_config["coded aperture"]["number of pixels along X"]))

            elif pattern_type == "slit":
                pattern = generate_slit_pattern((self.system_config["coded aperture"]["number of pixels along Y"],
                                                 self.system_config["coded aperture"]["number of pixels along X"]),
                                                config_pattern['pattern']['slit position'],
                                                config_pattern['pattern']['slit width'])
            elif pattern_type == "blue-noise type 1":
                pattern = generate_blue_noise_type_1_pattern((self.system_config["coded aperture"][
                                                                  "number of pixels along Y"],
                                                              self.system_config["coded aperture"][
                                                                  "number of pixels along X"]))
            elif pattern_type == "blue-noise type 2":
                pattern = generate_blue_noise_type_2_pattern((self.system_config["coded aperture"][
                                                                  "number of pixels along Y"],
                                                              self.system_config["coded aperture"][
                                                                  "number of pixels along X"]))
            elif pattern_type == "custom h5 pattern":
                pattern = load_custom_pattern((self.system_config["coded aperture"]["number of pixels along Y"],
                                               self.system_config["coded aperture"]["number of pixels along X"]),
                                              config_pattern['pattern']['file path'])
            else:
                
                raise ValueError(f"Pattern type {pattern_type} is not supported, change it in the 'pattern.yml' config file")

            # Assume pattern is a numpy array; convert to tensor
            pattern_tensor = torch.from_numpy(pattern)
            pattern_list.append(pattern_tensor)

        # Stack all generated pattern tensors along a new dimension
        stacked_patterns = torch.stack(pattern_list)  # Shape: (nb_of_patterns, y, x)

        self.pattern = stacked_patterns
        return stacked_patterns