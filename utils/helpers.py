import logging
import os
import yaml
from datetime import datetime

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



def load_yaml_config(file_path):
    """
    Load a YAML configuration file.

    :param file_path: Path to the YAML file
    :return: A dictionary containing the configuration data
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def undersample_grid(grid, target_size=40):
    factor = grid.shape[0] // target_size
    if factor == 0:
        return grid[::1,::1]
    return grid[::factor,::factor]