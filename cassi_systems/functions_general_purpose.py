import yaml
import math
import os
import numpy as np
from datetime import datetime
import h5py


def load_yaml_config(file_path):
    """
    Load a YAML configuration file as a dictionary

    Args:
        file_path (str): path to the YAML configuration file

    Returns:
        dict: configuration dictionary
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def initialize_acquisitions_directory(config):
    """
    Initialize the directory where the results of the acquisition will be stored

    Args:
        config (dict): a configuration dictionary containing storing information

    Returns:
        str: path to the directory where the results will be stored
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_directory = os.path.join(config["results directory"], config["acquisition name"], timestamp)
    os.makedirs(result_directory, exist_ok=True)
    return result_directory

def save_data_in_hdf5(file_name, data,result_directory):
    """
    Save a dataset in a HDF5 file

    Args:
        file_name (str): name of the file
        data (any type): data to save
        result_directory (str): path to the directory where the results will be stored

    """

    with h5py.File(result_directory + f'/{file_name}.h5', 'w') as f:
        f.create_dataset(f'{file_name}', data=data)
    print(f"'{file_name}' dataset saved in '{file_name}' file, stored in {result_directory} directory")

def save_config_file(config_file_name,config_file,result_directory):
    """
    Save a configuration file in a YAML file

    Args:
        config_file_name (str): name of the file
        config_file (dict): configuration file to save
        result_directory (str): path to the directory where the results will be stored

    """
    with open(result_directory + f"/{config_file_name}.yml", 'w') as file:
        yaml.safe_dump(config_file, file)

def rotation_z(theta):
    """
    Rotate 3D matrix around the Z axis

    Args:
        theta (float): Input angle (in rad)

    Returns:
        numpy.ndarray : 2D rotation matrix

    """

    r = np.array(((np.cos(theta), -np.sin(theta), 0),
                  (np.sin(theta), np.cos(theta), 0),
                  (0, 0, 1)))

    return r


def rotation_y(theta):
    """
    Rotate 3D matrix around the Y axis

    Args:
        theta (float): Input angle (in rad)

    Returns:
        numpy.ndarray : 2D rotation matrix

    """

    r = np.array(((np.cos(theta), 0, np.sin(theta)),
                  (0, 1, 0),
                  (-np.sin(theta), 0, np.cos(theta))))

    return r


def rotation_x(theta):
    """
    Rotate 3D matrix around the X axis

    Args:
        theta (float): Input angle (in rad)

    Returns:
        numpy.ndarray : 2D rotation matrix

    """

    r = np.array(((1, 0, 0),
                  (0, math.cos(theta), -math.sin(theta)),
                  (0, math.sin(theta), math.cos(theta))))

    return r