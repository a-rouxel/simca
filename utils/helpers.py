import logging
import os
import yaml
from datetime import datetime
# import h5py
import numpy as np
import math
def configure_logging(result_directory, log_directory="logs"):
    log_directory = os.path.join(result_directory, log_directory)
    os.makedirs(log_directory, exist_ok=True)

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")


    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    log_file_name = f"experiment_{timestamp}.log"

    file_handler = logging.FileHandler(os.path.join(log_directory, log_file_name))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

def initialize_directory(config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_directory = os.path.join(config["infos"]["results directory"], timestamp)
    os.makedirs(result_directory, exist_ok=True)
    return result_directory

def load_yaml_config(file_path):
    """
    Load a YAML configuration file.

    :param file_path: Path to the YAML file
    :return: A dictionary containing the configuration data
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def save_frames_to_h5(frames_data, result_directory, file_name="frames.h5"):
    """
    Save a list of images of timestamps (frames) to an h5 file.

    :param frames_data: A tuple containing the list of frames and the list of frame times
    :param result_directory: Directory where the h5 file will be saved
    :param file_name: Name of the h5 file (default: "frames.h5")
    """
    frames_list, frame_times = frames_data

    # Ensure the result_directory exists
    os.makedirs(result_directory, exist_ok=True)

    # Create an h5 file in the specified directory with the given file name
    with h5py.File(os.path.join(result_directory, file_name), "w") as h5_file:
        # Create a dataset for the frames
        frames_dataset = h5_file.create_dataset(
            "frames",
            shape=(len(frames_list), *frames_list[0].shape),
            dtype=np.float32
        )

        # Create a dataset for the frame times
        frame_times_dataset = h5_file.create_dataset(
            "frame_times",
            shape=(len(frame_times),),
            dtype=np.float64
        )

        # Save each frame and frame time to the datasets
        for i, (frame, frame_time) in enumerate(zip(frames_list, frame_times)):
            frames_dataset[i] = frame
            frame_times_dataset[i] = frame_time

"""
Created on Sat Jun  6 01:12:20 2020

@author: arouxel
"""


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


def _find_nearest(arrayX, X_s):
    """
    Find the nearest sample in the arrayX to the value X_s.

    Parameters
    ----------
    arrayX : 1D numpy array
        Array of values.
    X_s : float
        The value for which we want to find the nearest in the array.

    Returns
    -------
    value : float
        The corresponding nearest value in the array.
    idx_x : int
        The index of the corresponding nearest value.

    """

    arrayX = np.asarray(arrayX);
    idx_x = (np.abs(arrayX - X_s)).argmin();

    return (arrayX[idx_x], idx_x)




def rescale_array(vector_to_rescale, new_size):
    """
    Rescale (resample) an array to get a different number of sample.

    Parameters
    ----------
    vector_to_rescale : 1D numpy array
        The array to resample.
    new_size : int
        The new number of sample.

    Returns
    -------
    array_rescaled : 1D numpy array
        The rescaled array.

    """
    # Note: different interpolation possible (linear, cubic...)
    x = np.linspace(0, 1, num=len(vector_to_rescale), endpoint=True)
    f = interp1d(x, vector_to_rescale)
    xnew = np.linspace(0, 1, num=new_size, endpoint=True)
    array_rescaled = f(xnew)

    return array_rescaled


def voxel_exist(id_row, id_col, id_band, nb_row, nb_col, nb_band):
    """
    Check if the voxel at the position (id_row, id_col, id_band)
    exist in the cube of the cube of size (nb_row, nb_col, nb_band).

    Parameters
    ----------
    id_row : int
        Row index.
    id_col : int
        Column index.
    id_band : int
        Spectral band (wavelength) index.
    nb_row : int
        The number of rows of the cube.
    nb_col : int
        The number of columns of the cube.
    nb_band : int
        The number of spectral bands of the cube.

    Returns
    -------
    bool
        True if the pixel exist.

    """
    if (id_row < 0) or (id_col < 0) or (id_band < 0):
        return False

    if (nb_row - 1 < id_row) or (nb_col - 1 < id_col) or (nb_band - 1 < id_band):
        return False

    return True



def sellmeier(lambda_):
    """
    Evaluating the refractive index value of a BK7 prism for a given lambda

    Parameters
    ----------
    lambda_ : float -- in nm
        input wavelength on the prism.

    Returns
    -------
    n : float
        index value corresponding to the input wavelength

    """

    B1 = 1.03961212;
    B2 = 0.231792344;
    B3 = 1.01046945;
    C1 = 6.00069867 * (10 ** -3);
    C2 = 2.00179144 * (10 ** -2);
    C3 = 1.03560653 * (10 ** 2);

    lambda_in_mm = lambda_ / 1000

    n = math.sqrt(1 + B1 * lambda_in_mm ** 2 / (lambda_in_mm ** 2 - C1) + B2 * lambda_in_mm ** 2 / (
                lambda_in_mm ** 2 - C2) + B3 * lambda_in_mm ** 2 / (lambda_in_mm ** 2 - C3));

    return n

import numpy as np

def D_m(n, A):
    # A should be given in radians
    print(2 * np.arcsin(n * np.sin(A / 2)) - A)
    return 2 * np.arcsin(n * np.sin(A / 2)) - A

def alpha_c(A, D_m):
    return (A + D_m) / 2
