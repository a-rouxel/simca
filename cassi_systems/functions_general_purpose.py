import yaml
import math
import os
import numpy as np
from datetime import datetime



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