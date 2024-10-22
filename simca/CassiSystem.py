from simca.OpticalModel import OpticalModel
from simca.helper import *
import pytorch_lightning as pl


class CassiSystem(pl.LightningModule):

    """Class that contains the cassi system main attributes and methods"""

    def __init__(self, system_config=None,index_estimation_method="cauchy",device="cuda"):

        # Initialize LightningModule
        pl.LightningModule.__init__(self)
        self.to(device)

        self.system_config = system_config

        self.X_coded_aper_coordinates, self.Y_coded_aper_coordinates = self.generate_grid_coordinates(grid_name="coded aperture")
        self.X_detector_coordinates_grid, self.Y_detector_coordinates_grid = self.generate_grid_coordinates(grid_name="detector")
                
        self.wavelengths, self.nb_of_spectral_samples = self.generate_wavelengths()

        self.optical_model = OpticalModel(self.system_config,index_estimation_method=index_estimation_method,device=self.device)



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
            tuple: X coordinates grid (torch.Tensor) and Y coordinates grid (torch.Tensor)
        """
        x = torch.linspace(-nb_of_pixels_along_x * delta_x / 2, 
                           nb_of_pixels_along_x * delta_x / 2, 
                           nb_of_pixels_along_x)
        y = torch.linspace(-nb_of_pixels_along_y * delta_y / 2, 
                           nb_of_pixels_along_y * delta_y / 2, 
                           nb_of_pixels_along_y)

        # Create a two-dimensional grid of coordinates
        X_input_grid, Y_input_grid = torch.meshgrid(x, y, indexing='xy')

        return X_input_grid.to(self.device), Y_input_grid.to(self.device)


    
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
        self.optical_model = OpticalModel(self.system_config)

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
    
    
    
