import yaml
import math
import os
import numpy as np
from datetime import datetime
import h5py
import torch

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


