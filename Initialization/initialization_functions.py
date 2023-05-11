# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:32:39 2020

@author: vportmann
"""

import Initialization.User_inputs.system_caracteristics_functions as defsys
from Initialization.Storage.storage import InitStoreData, store_configuration_files
from PSF.PSF_arm_n_2.Construct_psf_arm_2 import construct_psf_arm_2
from Filtering.oversampling_DMD import preprocessing_oversampling_DMD
import numpy as np



def initialization(cfg_system, cfg_simulation):

    """
     - Set up the system and the scene coordinates.
     - Initialize the data cubes.

    Parameters
    ----------
    cfg_system : dict
        The configuration file of the system generating the spectro-spatial sampling
    cfg_simulation : dict
        The configuration file of the simulation

    Returns
    -------
    list_system_sampling_cubes : list of memmap
        List of system sampling cubes
    list_system_parameters : object of the class class_sys_parameters
        Object containing multiple system parameters

    """

    print("\n\n --- LOADING AND BUILDING SYSTEM CARACTERISTICS --- ")

    global list_system_parameters


    print('\n Chosen optical system : {}'.format(cfg_system['name']['system_name']))

    # ----------------------- Set up the Optical system -----------------------#
    """ Defining optical design parameters from system configuration file"""

    [X_cam, Y_cam] = defsys.define_camera_sampling(cfg_system)
    [X_dmd, Y_dmd] = defsys.define_DMD_sampling(cfg_system)

    """ Defining oversampled system coordinates"""
    X_dmd_oversampled, X_dmd_diamond, Y_dmd_diamond, Y_cam_oversampled, X_cam_oversampled, matrix_position_centers_mirrors = defsys.construct_oversampled_dimensions(
        X_dmd, Y_cam, X_cam, cfg_system, cfg_simulation)

    oversampled_shape_cube = (Y_cam_oversampled.size, X_cam_oversampled.size, X_dmd_oversampled.size)
    shape_cube = (Y_cam.size, X_cam.size, X_dmd.size)

    # --------------- Display the size of the different elements -----------------#
    print("\n*  DMD dimensions are : ", (Y_dmd.size, X_dmd.size))
    print("*  camera dimensions are : ", (Y_cam.size, X_cam.size))
    print("*  CFNR dimensions are : ", oversampled_shape_cube)

    print("\n...pre-calculating PSFs....")

    # --------- Construct the two PSF (corresponding to the sampling) ---------#
    array_arm1_psf = None  # Constructed later (when we know the spatial sampling of the scene)
    o_arm2_psf = construct_psf_arm_2(X_dmd, Y_cam, X_cam, cfg_system, cfg_simulation)

    # --------- Futures results storage init (when we have the scene)
    [band_scale, etendue, X_scale, Y_scale] = [None, None, None, None]

    print("...storing initialization data....")
    # --------- Store all theses constructed parameters in an object ----------#

    list_system_parameters = defsys.class_sys_parameters(
        X_cam, Y_cam, X_dmd, Y_dmd, etendue, X_scale, Y_scale, shape_cube,
        array_arm1_psf, o_arm2_psf,
        X_dmd_oversampled, Y_cam_oversampled, X_cam_oversampled, matrix_position_centers_mirrors,
        X_dmd_diamond, Y_dmd_diamond, band_scale,oversampled_shape_cube)

    # ----------- Pre-processing the mapping of the DMD oversampled -----------#
    preprocessing_oversampling_DMD(cfg_system, cfg_simulation, list_system_parameters)

    # ----- Storing initialization data
    sampling_data_path = cfg_simulation['sampling_data_path'] + cfg_system['name']['system_name'] + "/"

    list_system_sampling_cubes = InitStoreData(oversampled_shape_cube,sampling_data_path, cfg_system)
    np.savez(sampling_data_path + 'list_system_parameters.npz', **list_system_parameters.__dict__)
    store_configuration_files([cfg_system, cfg_simulation], sampling_data_path)

    return list_system_sampling_cubes, list_system_parameters










         

    









