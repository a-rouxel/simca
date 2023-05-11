import numpy as np

from utils import *
import logging
import matplotlib.pyplot as plt
from utils.functions_retropropagating import *
class Dimensionner():

    def __init__(self,config):

        self.config = config
        self.result_directory = initialize_directory(self.config)
        # configure_logging(self.result_directory)
        #
        # logging.info("--- Experiment Started ---")
        #
        # logging.info("--- LOADING AND BUILDING SYSTEM CARACTERISTICS --- ")
        # logging.info('Chosen optical system : {}'.format(config['infos']['system name']))

        self.X_cam, self.Y_cam = self.define_camera_sampling()
        # self.X_dmd, self.Y_dmd = self.define_DMD_sampling()

    def propagate(self):

        self.n_array_center = np.full(self.X_cam.shape, sellmeier(self.config["spectral range"]["wavelength center"]))
        self.n_array_min= np.full(self.X_cam.shape, sellmeier(self.config["spectral range"]["wavelength min"]))
        self.n_array_max = np.full(self.X_cam.shape, sellmeier(self.config["spectral range"]["wavelength max"]))


        self.X_cam_flatten = self.X_cam.flatten()
        self.Y_cam_flatten = self.Y_cam.flatten()

        self.Dm = D_m(sellmeier(self.config["spectral range"]["wavelength center"]),
                      np.radians(self.config["system architecture"]["dispersive_element_1_A"]))
        self.alpha_c = alpha_c(np.radians(self.config["system architecture"]["dispersive_element_1_A"]), self.Dm)

        self.list_wavelengths= list()
        self.list_X_dmd = list()
        self.list_Y_dmd = list()

        for lba in np.linspace(self.config["spectral range"]["wavelength min"],self.config["spectral range"]["wavelength max"],self.config["spectral_samples"]):
            n_array_flatten = np.full(self.X_cam_flatten.shape, sellmeier(lba))

            print(self.X_cam_flatten.shape,self.Y_cam_flatten.shape,n_array_flatten.shape,self.config["system architecture"]["dispersive_element_1_A"],self.config["system architecture"]["focal_lens_1"],self.alpha_c,self.config["system architecture"]["dispersive_element_1_delta_alpha_c"],self.config["system architecture"]["dispersive_element_1_delta_beta_c"])
            X_dmd, Y_dmd = find_cam_position_curve_fit_new_model_no_A_oneinput(self.X_cam_flatten , self.Y_cam_flatten,n_array_flatten, np.radians(self.config["system architecture"]["dispersive_element_1_A"]), self.config["system architecture"]["focal_lens_1"], self.alpha_c, np.radians(self.config["system architecture"]["dispersive_element_1_delta_alpha_c"]), np.radians(self.config["system architecture"]["dispersive_element_1_delta_beta_c"]))
            self.list_X_dmd.append(X_dmd.reshape(self.X_cam.shape))
            self.list_Y_dmd.append(Y_dmd.reshape(self.Y_cam.shape))
            self.list_wavelengths.append(np.full(self.X_cam.shape, lba).reshape(self.Y_cam.shape))

        return self.list_X_dmd, self.list_Y_dmd, self.list_wavelengths


    def define_camera_sampling(self):
            nb_pix_along_disp = self.config['camera']['number of pixels along dispersion']
            nb_pix_across_disp = self.config['camera']['number of pixels across dispersion']

            # Generate one-dimensional arrays for x and y coordinates
            x = np.linspace(-nb_pix_along_disp * self.config['camera']['pixel size']/2,nb_pix_along_disp * self.config['camera']['pixel size']/2, nb_pix_along_disp)
            y = np.linspace(-nb_pix_across_disp * self.config['camera']['pixel size']/2, nb_pix_across_disp * self.config['camera']['pixel size']/2, nb_pix_across_disp)

            # Create a two-dimensional grid of coordinates
            X_cam, Y_cam = np.meshgrid(x, y)


            return [X_cam, Y_cam]

    def define_DMD_sampling(self):
            nb_um_along_disp= self.config['DMD']['number of micromirrors along dispersion']
            nb_um_across_disp = self.config['DMD']['number of micromirrors across dispersion']

            # Generate one-dimensional arrays for x and y coordinates
            x = np.linspace(-nb_um_along_disp * self.config['DMD']['micromirror size']/2,nb_um_along_disp * self.config['DMD']['micromirror size']/2, nb_um_along_disp)
            y = np.linspace(-nb_um_across_disp * self.config['DMD']['micromirror size']/2, nb_um_across_disp * self.config['DMD']['micromirror size']/2, nb_um_across_disp)

            # Create a two-dimensional grid of coordinates
            X_dmd, Y_dmd = np.meshgrid(x, y)

            return [X_dmd, Y_dmd]

    def retropropagate(self):


        if self.config["system architecture"]["dispersive_element_1_type"] == "grating":
            self.X_cam_flatten = self.X_cam.flatten()
            self.Y_cam_flatten = self.Y_cam.flatten()
            print("grating")
        elif self.config["system architecture"]["dispersive_element_1_type"] == "prism":

            self.X_cam_flatten = self.X_cam.flatten()
            self.Y_cam_flatten = self.Y_cam.flatten()

            self.Dm = D_m(sellmeier(self.config["spectral range"]["wavelength center"]), np.radians(self.config["system architecture"]["dispersive_element_1_A"]))
            self.alpha_c = alpha_c(np.radians(self.config["system architecture"]["dispersive_element_1_A"]), self.Dm)

            X_dmd_center, Y_dmd_center = find_cam_position_curve_fit_new_model_no_A_oneinput(self.X_cam_flatten , self.Y_cam_flatten,self.n_array_center.flatten(), np.radians(self.config["system architecture"]["dispersive_element_1_A"]), self.config["system architecture"]["focal_lens_1"], self.alpha_c, np.radians(self.config["system architecture"]["dispersive_element_1_delta_alpha_c"]), np.radians(self.config["system architecture"]["dispersive_element_1_delta_beta_c"]))
            X_dmd_min, Y_dmd_min = find_cam_position_curve_fit_new_model_no_A_oneinput(self.X_cam_flatten , self.Y_cam_flatten,self.n_array_min.flatten(), np.radians(self.config["system architecture"]["dispersive_element_1_A"]), self.config["system architecture"]["focal_lens_1"], self.alpha_c, np.radians(self.config["system architecture"]["dispersive_element_1_delta_alpha_c"]), np.radians(self.config["system architecture"]["dispersive_element_1_delta_beta_c"]))
            X_dmd_max, Y_dmd_max = find_cam_position_curve_fit_new_model_no_A_oneinput(self.X_cam_flatten , self.Y_cam_flatten,self.n_array_max.flatten(), np.radians(self.config["system architecture"]["dispersive_element_1_A"]), self.config["system architecture"]["focal_lens_1"], self.alpha_c, np.radians(self.config["system architecture"]["dispersive_element_1_delta_alpha_c"]), np.radians(self.config["system architecture"]["dispersive_element_1_delta_beta_c"]))


            X_dmd_center = X_dmd_center.reshape((self.X_cam.shape))
            Y_dmd_center = Y_dmd_center.reshape((self.Y_cam.shape))
            X_dmd_min= X_dmd_min.reshape((self.X_cam.shape))
            Y_dmd_min = Y_dmd_min.reshape((self.Y_cam.shape))
            X_dmd_max = X_dmd_max.reshape((self.X_cam.shape))
            Y_dmd_max = Y_dmd_max.reshape((self.Y_cam.shape))

        return list_X_dmd, list_Y_dmd



        # """ Defining oversampled system coordinates"""
        # X_dmd_oversampled, X_dmd_diamond, Y_dmd_diamond, Y_cam_oversampled, X_cam_oversampled, matrix_position_centers_mirrors = defsys.construct_oversampled_dimensions(
        #     X_dmd, Y_cam, X_cam, cfg_system, cfg_simulation)
        #
        # oversampled_shape_cube = (Y_cam_oversampled.size, X_cam_oversampled.size, X_dmd_oversampled.size)
        # shape_cube = (Y_cam.size, X_cam.size, X_dmd.size)
        #
        # # --------------- Display the size of the different elements -----------------#
        # print("\n*  DMD dimensions are : ", (Y_dmd.size, X_dmd.size))
        # print("*  camera dimensions are : ", (Y_cam.size, X_cam.size))
        # print("*  CFNR dimensions are : ", oversampled_shape_cube)
        #
        # print("\n...pre-calculating PSFs....")
        #
        # # --------- Construct the two PSF (corresponding to the sampling) ---------#
        # array_arm1_psf = None  # Constructed later (when we know the spatial sampling of the scene)
        # o_arm2_psf = construct_psf_arm_2(X_dmd, Y_cam, X_cam, cfg_system, cfg_simulation)
        #
        # # --------- Futures results storage init (when we have the scene)
        # [band_scale, etendue, X_scale, Y_scale] = [None, None, None, None]
        #
        # print("...storing initialization data....")
        # # --------- Store all theses constructed parameters in an object ----------#
        #
        # list_system_parameters = defsys.class_sys_parameters(
        #     X_cam, Y_cam, X_dmd, Y_dmd, etendue, X_scale, Y_scale, shape_cube,
        #     array_arm1_psf, o_arm2_psf,
        #     X_dmd_oversampled, Y_cam_oversampled, X_cam_oversampled, matrix_position_centers_mirrors,
        #     X_dmd_diamond, Y_dmd_diamond, band_scale, oversampled_shape_cube)
        #
        # # ----------- Pre-processing the mapping of the DMD oversampled -----------#
        # preprocessing_oversampling_DMD(cfg_system, cfg_simulation, list_system_parameters)





