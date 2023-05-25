import logging
import os
import yaml
from datetime import datetime
import math
import h5py
import numpy as np
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
    print(config)
    result_directory = os.path.join(config["results directory"], config["acquisition name"], timestamp)
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

def undersample_grid(grid, target_size=40):
    factor = grid.shape[0] // target_size
    if factor == 0:
        return grid[::1,::1]
    return grid[::factor,::factor]