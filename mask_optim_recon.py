from simca import load_yaml_config
from simca.CassiSystemOptim import CassiSystemOptim
from simca.CassiSystem import CassiSystem
from data_handler import CubesDataModule
import numpy as np
import snoop
import matplotlib.pyplot as plt
import matplotlib.animation as anim
#import matplotlib
import torch
import time, datetime
import os

config_dataset = load_yaml_config("simca/configs/dataset.yml")
config_patterns = load_yaml_config("simca/configs/pattern.yml")
config_acquisition = load_yaml_config("simca/configs/acquisition.yml")
config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_max_center.yml")

# Load datacubes
# Generate random mask
# Run SIMCA to make acquisition 1
# ResNET -> mask
# Run SIMCA to make acquisition 2
# Reconstruction MST/CST -> out_cube
# Compare out_cube with datacube to compute loss

data_dir = "/local/users/ademaio/lpaillet/mst_datasets"
datamodule = CubesDataModule(data_dir, batch_size=5, num_workers=2)

# cassi_system.dataset = datamodule.train_dataloader[i][0]
# cassi_system.wavelengths = datamodule.train_dataloader[i][1]