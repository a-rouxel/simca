from simca.OpticalModelTorch import OpticalModelTorch
from simca.functions_acquisition import *
from simca.functions_patterns_generation import *
from simca.functions_scenes import *
from simca.functions_scenes_torch import *
from simca.functions_general_purpose import *
from simca.CassiSystem import CassiSystem
from simca.functions_acquisition_torch import *


class CassiSystemOptim(CassiSystem):
    """Class that contains the cassi system main attributes and methods"""

    def __init__(self, system_config=None):

        super().__init__(system_config=system_config)

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


        n1 = self.optical_model.glass1.calc_rindex(self.wavelengths)
        n2 = self.optical_model.glass2.calc_rindex(self.wavelengths)
        n3 = self.optical_model.glass3.calc_rindex(self.wavelengths)

        X_input_grid = torch.from_numpy(self.X_coded_aper_coordinates) if isinstance(self.X_coded_aper_coordinates, np.ndarray) else self.X_coded_aper_coordinates
        Y_input_grid = torch.from_numpy(self.Y_coded_aper_coordinates) if isinstance(self.Y_coded_aper_coordinates, np.ndarray) else self.Y_coded_aper_coordinates
        wavelength_vec = torch.from_numpy(self.wavelengths) if isinstance(self.wavelengths, np.ndarray) else self.wavelengths
        n1_vec = torch.from_numpy(n1) if isinstance(n1, np.ndarray) else n1
        n2_vec = torch.from_numpy(n2) if isinstance(n2, np.ndarray) else n2
        n3_vec = torch.from_numpy(n3) if isinstance(n3, np.ndarray) else n3

        X_input_grid_3D = X_input_grid[:,:,None].repeat(1, 1,self.nb_of_spectral_samples)
        Y_input_grid_3D = Y_input_grid[:,:,None].repeat(1, 1,self.nb_of_spectral_samples)
        lba_3D = wavelength_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        n1_3D = n1_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        n2_3D = n2_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)
        n3_3D = n3_vec[None,None,:].repeat(X_input_grid.shape[0], X_input_grid.shape[1],1)

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
                                                                 Y_target=self.Y_detector_coordinates_grid)


        return self.filtering_cube
    

    def image_acquisition(self, use_psf=False, chunck_size=50):
        """
        Run the acquisition/measurement process depending on the cassi system type

        Args:
            chunck_size (int): default block size for the interpolation

        Returns:
            numpy.ndarray: compressed measurement (R x C)
        """

        dataset = self.interpolate_dataset_along_wavelengths_torch(self.wavelengths, chunck_size)

        if dataset is None:
            return None
        dataset_labels = self.dataset_labels

        if self.system_config["system architecture"]["system type"] == "DD-CASSI":

            try:
                self.filtering_cube
            except:
                return print("Please generate filtering cube first")

            scene = torch.from_numpy(match_dataset_to_instrument(dataset, self.filtering_cube))

            measurement_in_3D = generate_dd_measurement_torch(scene, self.filtering_cube, chunck_size)

            self.last_filtered_interpolated_scene = measurement_in_3D
            self.interpolated_scene = scene

            if dataset_labels is not None:
                scene_labels = torch.from_numpy(match_dataset_labels_to_instrument(dataset_labels, self.filtering_cube))
                self.scene_labels = scene_labels


        elif self.system_config["system architecture"]["system type"] == "SD-CASSI":

            X_coded_aper_coordinates_crop = crop_center(self.X_coded_aper_coordinates,dataset.shape[1], dataset.shape[0])
            Y_coded_aper_coordinates_crop = crop_center(self.Y_coded_aper_coordinates,dataset.shape[1], dataset.shape[0])

            scene = torch.from_numpy(match_dataset_to_instrument(dataset, X_coded_aper_coordinates_crop))

            pattern_crop = crop_center(self.pattern, scene.shape[1], scene.shape[0])

            filtered_scene = scene * pattern_crop[..., None].repeat((1, 1, scene.shape[2]))

            self.propagate_coded_aperture_grid(X_input_grid=X_coded_aper_coordinates_crop, Y_input_grid=Y_coded_aper_coordinates_crop, use_torch = True)

            sd_measurement = interpolate_data_on_grid_positions_torch(filtered_scene,
                                                                self.X_coordinates_propagated_coded_aperture,
                                                                self.Y_coordinates_propagated_coded_aperture,
                                                                self.X_detector_coordinates_grid,
                                                                self.Y_detector_coordinates_grid)

            self.last_filtered_interpolated_scene = sd_measurement
            self.interpolated_scene = scene

            if dataset_labels is not None:
                scene_labels = torch.from_numpy(match_dataset_labels_to_instrument(dataset_labels, self.last_filtered_interpolated_scene))
                self.scene_labels = scene_labels

        self.panchro = torch.sum(self.interpolated_scene, dim=2)

        if use_psf:
            self.apply_psf_torch()
        else:
            print("No PSF was applied")

        # Calculate the other two arrays
        self.measurement = torch.sum(self.last_filtered_interpolated_scene, dim=2)

        return self.measurement
    
    def interpolate_dataset_along_wavelengths_torch(self, new_wavelengths_sampling, chunk_size):
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
        except :
            raise ValueError("The dataset must be loaded first")
        
        self.dataset_wavelengths = torch.from_numpy(self.dataset_wavelengths) if isinstance(self.dataset_wavelengths, np.ndarray) else self.dataset_wavelengths
        new_wavelengths_sampling = torch.from_numpy(new_wavelengths_sampling).float() if isinstance(new_wavelengths_sampling, np.ndarray) else new_wavelengths_sampling
        self.dataset = torch.from_numpy(self.dataset).float() if isinstance(self.dataset, np.ndarray) else self.dataset

        if self.dataset_wavelengths[0] <= new_wavelengths_sampling[0] and self.dataset_wavelengths[-1] >= new_wavelengths_sampling[-1]:

            self.dataset_interpolated = interpolate_data_along_wavelength_torch(self.dataset,self.dataset_wavelengths,new_wavelengths_sampling, chunk_size)
            return self.dataset_interpolated
        else:
            raise ValueError("The new wavelengths sampling must be inside the dataset wavelengths range")
    
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