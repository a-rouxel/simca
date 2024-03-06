from simca import load_yaml_config
from simca.CassiSystemOptim import CassiSystemOptim
from simca.CassiSystem import CassiSystem
from data_handler import CubesDataModule
from MST.simulation.train_code.architecture import *
from MST.simulation.train_code.utils import *
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
config_system = load_yaml_config()

# Load datacubes
# Generate random mask
# Run SIMCA to make acquisition 1
# ResNET -> mask
# Run SIMCA to make acquisition 2
# Reconstruction MST/CST -> out_cube
# Compare out_cube with datacube to compute loss

data_dir = "/local/users/ademaio/lpaillet/mst_datasets"
datamodule = CubesDataModule(data_dir, batch_size=5, num_workers=2)
model_name = 'mst_plus_plus'
device = 'cuda'
lr = 0.001

cassi_system = CassiSystemOptim(system_config=config_system)
cassi_system.device = device

cassi_system.update_optical_model(system_config=config_system)
X_vec_out, Y_vec_out = cassi_system.propagate_coded_aperture_grid()

model = model_generator(model_name, None)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
mse = torch.nn.MSELoss()

def expand_mask_3d(mask):
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    return mask3d

def train(model_name):

    optimizer.zero_grad()
    # cassi_system.dataset = datamodule.train_dataloader[i][0]
    # cassi_system.wavelengths = datamodule.train_dataloader[i][1]
    input_mask = np.random.randint(0,1,size=(128,128))
    cassi_system.pattern = input_mask
    cassi_system.generate_filtering_cube()

    input_acq = cassi_system.image_acquisition(use_psf=False) # H x (W + d*(28-1))
    d=2
    input_acq = shift_back(input_acq, step=d)

    model.train()
    input_mask_3d = expand_mask_3d(input_mask) #TODO, like in train_method/utils.py
    output = model(input_acq, input_mask_3d)
    loss = torch.sqrt(mse(output, cassi_system.dataset))

    loss.backward()
    optimizer.step()

    model.eval()
    input_mask = np.random.randint(0,1,size=(128,128))
    cassi_system.pattern = input_mask
    cassi_system.generate_filtering_cube()
    # cassi_system.dataset = datamodule.test_dataloader[i][0]
    # cassi_system.wavelengths = datamodule.test_dataloader[i][1]
    input_acq = cassi_system.image_acquisition(use_psf=False)
    input_acq = shift_back(input_acq, step=d)

    output = model(input_acq, input_mask_3d)
    loss = torch.sqrt(mse(output, cassi_system.dataset))
    model.train()
