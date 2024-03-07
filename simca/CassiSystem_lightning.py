from simca.OpticalModelTorch import OpticalModelTorch
from simca.functions_acquisition import *
from simca.functions_patterns_generation import *
from simca.functions_scenes import *
from simca.functions_scenes_torch import *
from simca.functions_general_purpose import *
from simca.CassiSystem import CassiSystem
from simca.functions_acquisition_torch import *
import pytorch_lightning as pl
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CassiSystemOptim(pl.LightningModule, CassiSystem):
    """Class that contains the cassi system main attributes and methods"""

    def __init__(self, system_config=None):

        # Initialize LightningModule
        pl.LightningModule.__init__(self)
        # Initialize CassiSystem with system_config
        CassiSystem.__init__(self, system_config=system_config)

        self.system_config = system_config
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
        
        
        self.wavelengths = self.set_wavelengths(self.system_config["spectral range"]["wavelength min"],
                                                self.system_config["spectral range"]["wavelength max"],
                                                self.system_config["spectral range"]["number of spectral samples"])
        self.optical_model = OpticalModelTorch(self.system_config)

        
        self.empty_grid = torch.zeros((self.system_config["coded aperture"]["number of pixels along Y"],
            self.system_config["coded aperture"]["number of pixels along X"]))
        # self.array_x_positions = torch.rand(-1,1,self.system_config["coded aperture"]["number of pixels along X"])
        self.array_x_positions = torch.zeros((self.system_config["coded aperture"]["number of pixels along Y"]))+ 0.5

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

        print( "My DEVICE IS: ", self.device)
        return torch.from_numpy(X_input_grid).float().to(self.device), torch.from_numpy(Y_input_grid).float().to(self.device)

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

        self.system_wavelengths = torch.linspace(self.wavelength_min,self.wavelength_max,self.nb_of_spectral_samples)

        return self.system_wavelengths
    
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

        if self.optical_model.continuous_glass_materials1 == True:
            n1 = self.optical_model.calculate_dispersion_with_cauchy(self.wavelengths.cpu() ,self.optical_model.nd1,self.optical_model.vd1)
        else:
            n1 = self.optical_model.glass1.calc_rindex(self.wavelengths.cpu() )
        if self.optical_model.continuous_glass_materials2 == True:
            n2 = self.optical_model.calculate_dispersion_with_cauchy(self.wavelengths.cpu() ,self.optical_model.nd2,self.optical_model.vd2)
        else:
            n2 = self.optical_model.glass2.calc_rindex(self.wavelengths.cpu() )
        if self.optical_model.continuous_glass_materials3 == True:
            n3 = self.optical_model.calculate_dispersion_with_cauchy(self.wavelengths.cpu() ,self.optical_model.nd3,self.optical_model.vd3)
        else:
            n3 = self.optical_model.glass3.calc_rindex(self.wavelengths.cpu() )

        n1 = n1.to(self.device)
        n2 = n2.to(self.device)
        n3 = n3.to(self.device)

        #print("New X coded aperture_coordinates: ", self.X_coded_aper_coordinates.shape)
        #print("New Y coded aperture_coordinates: ", self.Y_coded_aper_coordinates.shape)

        X_input_grid = torch.from_numpy(self.X_coded_aper_coordinates) if isinstance(self.X_coded_aper_coordinates, np.ndarray) else self.X_coded_aper_coordinates
        Y_input_grid = torch.from_numpy(self.Y_coded_aper_coordinates) if isinstance(self.Y_coded_aper_coordinates, np.ndarray) else self.Y_coded_aper_coordinates
        wavelength_vec = torch.from_numpy(self.wavelengths) if isinstance(self.wavelengths, np.ndarray) else self.wavelengths
        n1_vec = torch.from_numpy(n1) if isinstance(n1, np.ndarray) else n1
        n2_vec = torch.from_numpy(n2) if isinstance(n2, np.ndarray) else n2
        n3_vec = torch.from_numpy(n3) if isinstance(n3, np.ndarray) else n3

        X_input_grid_3D = X_input_grid[:,:,None].repeat(1, 1,self.nb_of_spectral_samples).to(self.device)
        Y_input_grid_3D = Y_input_grid[:,:,None].repeat(1, 1,self.nb_of_spectral_samples).to(self.device)
        lba_3D = wavelength_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        n1_3D = n1_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        n2_3D = n2_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        n3_3D = n3_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)

        starting_time = time.time()
        #print('Propagating the coded aperture grid...')
        #print(X_input_grid_3D.get_device())
        #print(Y_input_grid_3D.get_device())
        #print(lba_3D.get_device())
        #print(n1_3D.get_device())
        #print(n2_3D.get_device())

        self.X_coordinates_propagated_coded_aperture, self.Y_coordinates_propagated_coded_aperture = self.optical_model.propagate(X_input_grid_3D, Y_input_grid_3D, lba_3D, n1_3D, n2_3D, n3_3D)

        return self.X_coordinates_propagated_coded_aperture, self.Y_coordinates_propagated_coded_aperture

    def generate_filtering_cube(self):
        """
        Generate filtering cube : each slice of the cube is a propagated pattern interpolated on the detector grid

        Returns:
        numpy.ndarray: filtering cube generated according to the optical system & the pattern configuration (R x C x W)

        """
        self.filtering_cube = interpolate_data_on_grid_positions_torch(data=self.pattern,
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
    

    def image_acquisition(self, hyperspectral_cube, pattern,wavelengths,use_psf=False, chunck_size=50):
        """
        Run the acquisition/measurement process depending on the cassi system type

        Args:
            chunck_size (int): default block size for the interpolation

        Returns:
            numpy.ndarray: compressed measurement (R x C)
        """
        self.wavelengths= self.wavelengths.to(self.device)



        #
        # plt.plot(hyperspectral_cube[0,0,0,:].cpu().numpy())
        # plt.title("Original spectrum")
        # plt.show()
        # print("cube shape: ", hyperspectral_cube.shape)
        # print("wavelengths shape: ", wavelengths.shape)
        # print("self.wavelengths shape: ", self.wavelengths.shape)
        dataset = self.interpolate_dataset_along_wavelengths_torch(hyperspectral_cube, wavelengths,self.wavelengths, chunck_size)


        # plt.plot(dataset[0,0,0,:].cpu().numpy())
        # plt.title("Interpolated spectrum")
        # plt.show()



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

            scene = match_dataset_to_instrument(dataset, self.filtering_cube)
            scene = torch.from_numpy(match_dataset_to_instrument(dataset, self.filtering_cube)).to(self.device) if isinstance(scene, np.ndarray) else scene.to(self.device)

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
            
            self.pattern_crop_dataset = interpolate_data_on_grid_positions_torch(pattern_crop.unsqueeze(-1),
                                                                self.X_coordinates_propagated_coded_aperture,
                                                                self.Y_coordinates_propagated_coded_aperture,
                                                                self.X_detector_coordinates_grid,
                                                                self.Y_detector_coordinates_grid)
            
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
            
            # print(f"filtered_scene: {filtered_scene.shape}")
            

            plt.imshow(pattern_crop[0,:,:,0].cpu().numpy())
            plt.title("Pattern crop1")
            plt.show()



            plt.imshow(filtered_scene[0,:,:,0].cpu().numpy())
            plt.title("Filtered scene")
            plt.show()

            #print("filtered_scene",filtered_scene.shape)

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

            for i in range(5):
                plt.imshow(sd_measurement[0,:,:,i*8].cpu().numpy())
                plt.title("SD measurement")
                plt.show()

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
    
    def generate_custom_pattern_parameters_slit(self, position=0.5):
        # Position is a float: 0 means slit is on the left edge, 1 means the slit is on the right edge
        self.array_x_positions = torch.zeros((self.system_config["coded aperture"]["number of pixels along Y"]))+ position
        self.array_x_positions = self.array_x_positions.to(self.device)
        return self.array_x_positions
    
    def generate_custom_pattern_parameters_slit_width(self, nb_slits=1, nb_rows=1, start_width=1):
        # Situation where we have nb_slits per row, and nb_rows rows of slits
        # Values in self.array_x_positions correspond to the width of the slit
        # self.array_x_positions is of shape (self.system_config["coded aperture"]["number of pixels along Y"], nb_rows)

        self.array_x_positions = torch.zeros((nb_slits, nb_rows))+start_width # Every slit starts with the same width
        self.array_x_positions = self.array_x_positions.to(self.device)
        self.array_x_positions_normalized = torch.zeros((nb_slits, nb_rows))+start_width
        return self.array_x_positions

    def generate_custom_slit_pattern_width(self, start_pattern = "line", start_position = 0):
        nb_slits, nb_rows = self.array_x_positions.shape
        pos_slits = self.system_config["coded aperture"]["number of pixels along X"]//(nb_slits+1) # Equally spaced slits
        height_slits = self.system_config["coded aperture"]["number of pixels along Y"]//nb_rows # Same length slits

        self.pattern = torch.zeros((self.system_config["coded aperture"]["number of pixels along Y"], self.system_config["coded aperture"]["number of pixels along X"])).to(self.device) # Pattern of correct size
        if start_pattern == "line":
            if start_position != 0:
                start_position = start_position - pos_slits/self.system_config["coded aperture"]["number of pixels along X"]
            for j in range(nb_slits):
                for i in range(nb_rows):
                    top_pad = i*height_slits # Padding necessary above slit (j,i)
                    if i == nb_rows-1:
                        bottom_pad = 0 # Padding necessary below slit (j,i)
                        # In that case, the last slit might be longer than the other ones in case size_X isn't divisible by nb_rows
                        array_x_pos = torch.zeros((height_slits+self.system_config["coded aperture"]["number of pixels along Y"] % nb_rows)) + start_position +  (j+1)*pos_slits/self.system_config["coded aperture"]["number of pixels along X"]
                    else:
                        # Set the position of the slit (j,i)
                        array_x_pos = torch.zeros((height_slits)) + start_position + (j+1)*pos_slits/self.system_config["coded aperture"]["number of pixels along X"]
                        bottom_pad = (nb_rows - i-1)*height_slits + self.system_config["coded aperture"]["number of pixels along Y"] % nb_rows # Padding necessary below slit (j,i)
                        top_pad = i*height_slits
                    
                    # Create a grid to represent positions
                    grid_positions = torch.arange(self.empty_grid.shape[1], dtype=torch.float32)
                    # Expand dimensions for broadcasting
                    expanded_x_positions = (array_x_pos.unsqueeze(-1)) * (self.empty_grid.shape[1]-1)
                    expanded_grid_positions = grid_positions.unsqueeze(0)

                    # Apply Gaussian-like function
                    sigma = (self.array_x_positions[j,i]+1)/2
                    gaussian = torch.exp(-(((expanded_grid_positions - expanded_x_positions)) ** 2) / (2 * sigma ** 2))

                    padded = torch.nn.functional.pad(gaussian, (0,0,top_pad,bottom_pad)) # padding: left - right - top - bottom

                    # Normalize to make sure the maximum value is 1
                    self.pattern = self.pattern + padded/padded.max()
        elif start_pattern == "corrected":
            for j in range(nb_slits):
                for i in range(nb_rows):
                    top_pad = i*height_slits # Padding necessary above slit (j,i)
                    if i == nb_rows-1:
                        bottom_pad = 0 # Padding necessary below slit (j,i)
                        # In that case, the last slit might be longer than the other ones in case size_X isn't divisible by nb_rows
                    else:
                        # Set the position of the slit (j,i)
                        bottom_pad = (nb_rows - i-1)*height_slits + self.system_config["coded aperture"]["number of pixels along Y"] % nb_rows # Padding necessary below slit (j,i)
                        top_pad = i*height_slits
                    """ array_x_pos = torch.tensor(start_position[i])

                    # Create a grid to represent positions
                    grid_positions = torch.arange(self.empty_grid.shape[1], dtype=torch.float32)
                    # Expand dimensions for broadcasting
                    expanded_x_positions = (array_x_pos.unsqueeze(-1)) * (self.empty_grid.shape[1]-1)
                    expanded_grid_positions = grid_positions.unsqueeze(0)

                    # Apply Gaussian-like function
                    sigma = (self.array_x_positions[j,i]+1)/2
                    gaussian = torch.exp(-(((expanded_grid_positions - expanded_x_positions)) ** 2) / (2 * sigma ** 2))

                    padded = torch.nn.functional.pad(gaussian, (0,0,top_pad,bottom_pad)) # padding: left - right - top - bottom

                    # Normalize to make sure the maximum value is 1
                    self.pattern = self.pattern + padded/padded.max() """

                    c = start_position[i].clone().detach() # center of the slit
                    #d = ((torch.tanh(1.1*self.array_x_positions[j,i])+1)/2)/2 # width of the slit at pos 
                    d = self.array_x_positions[j,i]/2 # width of the slit at pos 
                    m = (c-d)*(self.system_config["coded aperture"]["number of pixels along X"]-1) # left bound
                    M = (c+d)*(self.system_config["coded aperture"]["number of pixels along X"]-1) # right bound
                    rect = torch.arange(self.system_config["coded aperture"]["number of pixels along X"]).to(self.device)
                    clamp_M = torch.clamp(M-rect, 0, 1)

                    clamp_m = torch.clamp(rect-m, 0, 1)
                    diff = 1-clamp_m
                    reg = torch.where(diff < 1, diff, -1)
                    clamp_m = torch.where(reg!=0, reg, 1)
                    clamp_m = torch.where(clamp_m!=-1, clamp_m, 0)
                    clamp_m = torch.roll(clamp_m, -1)
                    clamp_m[-1]=1

                    rect = clamp_M - clamp_m +1
                    rect = torch.where(rect!=2, rect, 0)
                    rect = torch.where(rect <= 1, rect, rect-1)
                    #rect = torch.clamp(-(rect-m)*(rect-M)+1,0,1).to(self.device)

                    gaussian_range = torch.arange(self.system_config["coded aperture"]["number of pixels along X"], dtype=torch.float32)
                    center_pos = 0.5*(len(gaussian_range)-1)
                    sigma = 1.5
                    gaussian_peaks = torch.exp(-((center_pos - gaussian_range) ** 2) / (2 * sigma ** 2)).to(self.device)
                    gaussian = gaussian_peaks /gaussian_peaks.max()
                    res = torch.nn.functional.conv1d(rect.unsqueeze(0), gaussian.unsqueeze(0).unsqueeze(0), padding = (len(gaussian_range)-1)//2).squeeze().to(self.device)

                    res = res/res.max()
                    self.pattern[i, :] = self.pattern[i, :] + res

        # Normalize to make sure the maximum value is 1
        self.pattern = self.pattern / self.pattern.max(dim=1).values.unsqueeze(-1)

        return self.pattern

    def generate_custom_slit_pattern(self):

        # Create a grid to represent positions
        grid_positions = torch.arange(self.empty_grid.shape[1], dtype=torch.float32).to(self.device)
        # Expand dimensions for broadcasting
        expanded_x_positions = ((self.array_x_positions.unsqueeze(-1)) * (self.empty_grid.shape[1]-1)).to(self.device)
        expanded_grid_positions = grid_positions.unsqueeze(0).to(self.device)

        # Apply Gaussian-like function
        # Adjust 'sigma' to control the sharpness
        sigma = 1.5
        gaussian_peaks = torch.exp(-((expanded_grid_positions - expanded_x_positions) ** 2) / (2 * sigma ** 2)).to(self.device)

        # Normalize to make sure the maximum value is 1
        self.pattern = gaussian_peaks / gaussian_peaks.max()
        return self.pattern


    
    # def generate_custom_slit_pattern(self):
    #     """
    #     Generate a custom slit pattern

    #     Args:
    #         array_x_positions (numpy.ndarray): array of the x positions of the slits between -1 and 1

    #     Returns:
    #         numpy.ndarray: generated slit pattern
    #     """
    #     pattern = torch.clone(self.empty_grid)
    #     self.array_x_positions += 1
    #     self.array_x_positions *= self.empty_grid.shape[1] // 2
    #     self.array_x_positions = self.array_x_positions.type(torch.int32)
    #     for i in range(self.array_x_positions.shape[0]):
    #         pattern[0, self.array_x_positions[i]] = 1

    #     return self.pattern

    def interpolate_dataset_along_wavelengths_torch(self, hyperspectral_cube, wavelengths, new_wavelengths_sampling,
                                                    chunk_size):
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
        except:
            self.dataset = hyperspectral_cube
            self.dataset_wavelengths = wavelengths
            #print(self.dataset.shape)
            #print(self.dataset_wavelengths.shape)
        
        self.dataset = hyperspectral_cube
        self.dataset_wavelengths = wavelengths

        

        self.dataset_wavelengths = torch.from_numpy(self.dataset_wavelengths) if isinstance(self.dataset_wavelengths,
                                                                                            np.ndarray) else self.dataset_wavelengths
        new_wavelengths_sampling = torch.from_numpy(new_wavelengths_sampling).float() if isinstance(
            new_wavelengths_sampling, np.ndarray) else new_wavelengths_sampling
        self.dataset = torch.from_numpy(self.dataset).float() if isinstance(self.dataset, np.ndarray) else self.dataset

        self.dataset_interpolated = self.interpolate_data_along_wavelength_torch(self.dataset, self.dataset_wavelengths[0],
                                                                                 new_wavelengths_sampling)
        return self.dataset_interpolated

    def interpolate_data_along_wavelength_torch(self, data, current_sampling, new_sampling):
        """Interpolate the input 3D data along a new sampling in the third axis (lambda).

        Args:
            data (torch.Tensor): 4D data to interpolate with shape (batch_size, lambda, x, y).
            current_sampling (torch.Tensor): 1D tensor of current sampling for the lambda axis.
            new_sampling (torch.Tensor): 1D tensor of new sampling for the lambda axis.
        """
        # Assuming data is (batch, y, x, lambda)
        data = data.to(self.device)

        # Reshape data to treat lambda as a separate dimension, e.g., combining batch, y, x into one dimension
        original_shape = data.shape
        data_reshaped = data.reshape(-1, original_shape[-1])  # Shape: (batch*y*x, lambda)

        # Calculate the scale factor or directly use the new size
        scale_factor = len(new_sampling) / original_shape[-1]

        # Interpolate using 1D interpolation along the last dimension
        interpolated_data_reshaped = F.interpolate(data_reshaped.unsqueeze(1),  # Add a fake channel dimension
                                                   size=len(new_sampling),
                                                   mode='linear',
                                                   align_corners=True).squeeze(1)  # Remove the fake channel dimension

        # Reshape back to the original batch, y, x format but with new lambda size
        interpolated_data = interpolated_data_reshaped.reshape(original_shape[0], original_shape[1], original_shape[2],
                                                               -1)

        return interpolated_data

    def interpolate_data_along_wavelength_torch_old(self, data, current_sampling, new_sampling, chunk_size=50):
        """Interpolate the input 3D data along a new sampling in the third axis.

        Args:
            data (numpy.ndarray): 3D data to interpolate
            current_sampling (numpy.ndarray): current sampling for the 3rd axis
            new_sampling (numpy.ndarray): new sampling for the 3rd axis
            chunk_size (int): size of the chunks to use for the interpolation
        """
        # print(self.device)
        # Generate the coordinates for the original grid
        x = torch.arange(data.shape[0]).float()
        y = torch.arange(data.shape[1]).float()
        z = current_sampling

        # Initialize an empty array for the result
        interpolated_data = torch.empty((data.shape[0], data.shape[1], len(new_sampling)))

        # Perform the interpolation in chunks
        for i in range(0, data.shape[0], chunk_size):
            for j in range(0, data.shape[1], chunk_size):
                new_coordinates = torch.meshgrid(x[i:i + chunk_size], y[j:j + chunk_size], new_sampling, indexing='ij')
                new_grid = torch.stack(new_coordinates, axis=-1).to(self.device)

                min_bound = torch.tensor([torch.min(x[i:i + chunk_size]), torch.min(y[j:j + chunk_size]), torch.min(z)]).to(self.device)
                max_bound = torch.tensor([torch.max(x[i:i + chunk_size]), torch.max(y[j:j + chunk_size]), torch.max(z)]).to(self.device)

                new_grid = (2 * ((new_grid - min_bound) / (max_bound - min_bound)) - 1).unsqueeze(0).flip(
                    -1).float()  # Normalize between -1 and 1

                """new_coordinates = np.meshgrid(x[i:i+chunk_size], y[j:j+chunk_size], new_sampling, indexing='ij')
                new_grid = np.stack(new_coordinates, axis=-1)

                new_grid = torch.from_numpy(2*((new_grid-min_bound)/(max_bound - min_bound))-1).unsqueeze(0).flip(-1).double()"""
                interpolated_data[i:i + chunk_size, j:j + chunk_size, :] = torch.nn.functional.grid_sample(
                    input=data[i:i + chunk_size, j:j + chunk_size, :][None, None, ...],
                    grid=new_grid,
                    padding_mode="zeros",
                    mode="bilinear",
                    align_corners=True).squeeze(0, 1)

        return interpolated_data
    
    def apply_psf(self):
        """
        Apply the PSF to the last measurement

        Returns:
            numpy.ndarray: last measurement cube convolved with by PSF (shape= R x C x W). Each slice of the 3D filtered scene is convolved with the PSF
        """
        if (self.optical_model.psf is not None) and (self.last_filtered_interpolated_scene is not None):
            # Expand the dimensions of the 2D matrix to match the 3D matrix
            psf_3D = self.optical_model.psf[..., None]

            # Perform the convolution using convolve
            result = torch.nn.functional.conv3d(self.last_filtered_interpolated_scene[None, None, ...], torch.flip(psf_3D, (0,1,2))[None, None, ...], padding = tuple((np.array(psf_3D.shape)-1)//2)).squeeze(0,1)
            result_panchro = torch.nn.functional.conv2d(self.panchro[None, None, ...], torch.flip(self.optical_model.psf, (0,1))[None, None, ...], padding = tuple((np.array(self.optical_model.psf.shape)-1)//2)).squeeze(0,1)

        else:
            print("No PSF or last measurement to apply PSF")
            result = self.last_filtered_interpolated_scene
            result_panchro = self.panchro

        self.last_filtered_interpolated_scene = result
        self.panchro = result_panchro

        return self.last_filtered_interpolated_scene