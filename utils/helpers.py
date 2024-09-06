import yaml
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

