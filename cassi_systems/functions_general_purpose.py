import yaml
import math
import os
import numpy as np
from datetime import datetime
import h5py


def load_yaml_config(file_path):
    """
    Load a YAML configuration file.

    :param file_path: Path to the YAML file
    :return: A dictionary containing the configuration data
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def initialize_acquisitions_directory(config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_directory = os.path.join(config["results directory"], config["acquisition name"], timestamp)
    os.makedirs(result_directory, exist_ok=True)
    return result_directory


def save_interpolated_scene(scene_name,interpolated_scene,result_directory):

    with h5py.File(result_directory + f'/{scene_name}.h5', 'w') as f:
        f.create_dataset('interpolated_scene', data=interpolated_scene)
    print(f"Interpolated scene saved in {result_directory}")

def save_filtered_interpolated_scene(filtered_scene_name,last_filtered_interpolated_scene,result_directory):

    with h5py.File(result_directory + f'/{filtered_scene_name}.h5', 'w') as f:
        f.create_dataset('filtered_image', data=last_filtered_interpolated_scene)
    print(f"Filtered interpolated scene saved in {result_directory}")

def save_measurement(measurement_name,measurement,result_directory):

    with h5py.File(result_directory + f'/{measurement_name}.h5', 'w') as f:
        f.create_dataset('measurement', data=measurement)
    print(f"Measurement saved in {result_directory}")

def save_list_of_measurements(measurements_name,list_of_measurements,result_directory):

    with h5py.File(result_directory + f'/{measurements_name}.h5', 'w') as f:
        f.create_dataset('list_of_measurements', data=list_of_measurements)
    print(f"list of measurements saved in {result_directory}")

def save_panchromatic_image(panchromatic_image_name,panchro,result_directory):

    with h5py.File(result_directory + f'/{panchromatic_image_name}.h5', 'w') as f:
        f.create_dataset('panchromatic_image', data=panchro)
    print(f"Panchromatic image saved in {result_directory}")

def save_filtering_cube(filtering_cube_name,filtering_cube,result_directory):

    with h5py.File(result_directory + f'/{filtering_cube_name}.h5', 'w') as f:
        f.create_dataset('filtering_cube', data=filtering_cube)
    print(f"Filtering cube saved in {result_directory}")

def save_list_of_filtering_cubes(list_of_filtering_cubes_name,list_of_filtering_cubes,result_directory):

    with h5py.File(result_directory + f'/{list_of_filtering_cubes_name}.h5', 'w') as f:
        f.create_dataset('list_of_filtering_cubes', data=list_of_filtering_cubes)
    print(f"list of filtering cubes saved in {result_directory}")

def save_mask(mask_name,mask,result_directory):

    with h5py.File(result_directory + f'/{mask_name}.h5', 'w') as f:
        f.create_dataset('mask', data=mask)
    print(f"Mask saved in {result_directory}")

def save_list_of_masks(list_of_masks_name,list_of_masks,result_directory):

    with h5py.File(result_directory + f'/{list_of_masks_name}.h5', 'w') as f:
        f.create_dataset('list_of_masks', data=list_of_masks)
    print(f"list of masks saved in {result_directory}")
def save_wavelengths(wavelengths_name,system_wavelengths,result_directory):

    with h5py.File(result_directory + f'/{wavelengths_name}.h5', 'w') as f:
        f.create_dataset('wavelengths', data=system_wavelengths)
    print(f"Wavelengths saved in {result_directory}")

def save_config_system(config_system_name,system_config,result_directory):

    with open(result_directory + f"/{config_system_name}.yml", 'w') as file:
        yaml.safe_dump(system_config, file)
    print(f"System configuration saved in {result_directory}")

def save_config_mask_and_filtering(config_mask_and_filtering_name,
                                   config_mask_and_filtering,
                                   result_directory):

    with open(result_directory + f"/{config_mask_and_filtering_name}.yml", 'w') as file:
        yaml.safe_dump(config_mask_and_filtering, file)
    print(f"Mask and filtering configuration saved in {result_directory}")

def save_config_acquisition(config_acquisition_name,config_acquisition,result_directory):

    with open(result_directory + f"/{config_acquisition_name}.yml", 'w') as file:
        yaml.safe_dump(config_acquisition, file)
    print(f"Acquisition configuration saved in {result_directory}")

def rotation_z(theta):
    """
    Rotate 3D matrix around the z axis

    Parameters
    ----------
    theta : float -- in rad
        Input angle.

    Returns
    -------
    r : 2D numpy array
        The rotation matrix.

    """

    r = np.array(((np.cos(theta), -np.sin(theta), 0),
                  (np.sin(theta), np.cos(theta), 0),
                  (0, 0, 1)));

    return r


def rotation_y(theta):
    """
    Rotate 3D matrix around the y axis

    Parameters
    ----------
    theta : float -- in rad
        Input angle.

    Returns
    -------
    r : 2D numpy array
        The rotation matrix.
    """

    r = np.array(((np.cos(theta), 0, np.sin(theta)),
                  (0, 1, 0),
                  (-np.sin(theta), 0, np.cos(theta))));

    return r


def rotation_x(theta):
    """
    Rotate 3D matrix around the x axis

    Parameters
    ----------
    theta : float -- in rad
        Input angle.

    Returns
    -------
    r : 2D numpy array
        The rotation matrix.
    """

    r = np.array(((1, 0, 0),
                  (0, math.cos(theta), -math.sin(theta)),
                  (0, math.sin(theta), math.cos(theta))));

    return r