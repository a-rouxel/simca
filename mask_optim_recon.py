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
import tqdm

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
model_name = 'mst_plus_plus'
device = 'cuda'
lr = 0.001
nb_epochs = 100

def expand_mask_3d(mask):
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    return mask3d

def train_recons(model, cassi_system, nb_epochs, display_iter=10):
    train_losses = []
    test_losses = []
    for e in range(1, nb_epochs + 1):
        avg_loss = 0.0
        mean_losses = []
        # Run the training loop for one epoch
        model.train()
        for batch_idx, (cube, wavelengths) in tqdm(
            enumerate(datamodule.train_dataloader), total=len(datamodule.train_dataloader)):
            optimizer.zero_grad()
            #for i in range(cube.shape[0]):
            # cube_i = cube[i,...]
            # wavelengths_i = wavelengths[i,...]

            
            # cassi_system.dataset = cube # b x 28 x H x W
            # cassi_system.wavelengths = wavelengths
            input_mask = np.random.randint(0,1,size=(128,128))
            cassi_system.pattern = torch.tensor(input_mask)
            cassi_system.generate_filtering_cube()

            input_acq = cassi_system.image_acquisition(use_psf=False) # b x H x (W + d*(28-1))
            d=2
            input_acq = shift_back(input_acq, step=d)

            input_mask_3d = expand_mask_3d(input_mask) #TODO, like in train_method/utils.py
            output = model(input_acq, input_mask_3d)
            loss = torch.sqrt(mse(output, cube))

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            mean_losses.append(np.mean(np.array(mean_losses)))
            if display_iter and batch_idx % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                string = string.format(
                    e,
                    nb_epochs,
                    batch_idx * len(cube),
                    len(cube) * len(datamodule.train_dataloader),
                    100.0 * batch_idx / len(datamodule.train_dataloader),
                    mean_losses[batch_idx],
                )
                tqdm.write(string)
        
        avg_loss /= len(datamodule.train_dataloader)
        train_losses.append(avg_loss)

        # Run the test loop for one epoch
        avg_loss = 0
        model.eval()
        for batch_idx, (cube, wavelengths) in enumerate(datamodule.val_dataloader):
            with torch.no_grad():
                input_mask = np.random.randint(0,1,size=(128,128))
                cassi_system.pattern = torch.tensor(input_mask)
                cassi_system.generate_filtering_cube()
                # cassi_system.dataset = cube
                # cassi_system.wavelengths = wavelengths
                input_acq = cassi_system.image_acquisition(use_psf=False)
                input_acq = shift_back(input_acq, step=d)
                input_mask_3d = expand_mask_3d(input_mask) #TODO, like in train_method/utils.py

                output = model(input_acq, input_mask_3d)
                loss = torch.sqrt(mse(output, cube))
                avg_loss += loss.item()

        avg_loss /= len(datamodule.val_dataloader)
        string = "Test (epoch {}/{})\tLoss:{:.6f}".format(e, nb_epochs, avg_loss)
        tqdm.write(string)
        test_losses.append(avg_loss)

def test_recons(model, cassi_system):
    model.eval()
    avg_loss = 0

    for batch_idx, (cube, wavelengths) in enumerate(datamodule.test_dataloader):
        with torch.no_grad():
            input_mask = np.random.randint(0,1,size=(128,128))
            cassi_system.pattern = torch.tensor(input_mask)
            cassi_system.generate_filtering_cube()
            # cassi_system.dataset = cube
            # cassi_system.wavelengths = wavelengths
            input_acq = cassi_system.image_acquisition(use_psf=False)
            input_acq = shift_back(input_acq, step=d)
            input_mask_3d = expand_mask_3d(input_mask) #TODO, like in train_method/utils.py

            output = model(input_acq, input_mask_3d)
            loss = torch.sqrt(mse(output, cube))
            avg_loss += loss.item()
    avg_loss /= len(datamodule.test_dataloader)
    print(f"\nTest loss: {avg_loss:.6f}")

if __name__ == '__main__':
    datamodule = CubesDataModule(data_dir, batch_size=5, num_workers=2)
    cassi_system = CassiSystemOptim(system_config=config_system)
    cassi_system.device = device

    cassi_system.update_optical_model(system_config=config_system)
    X_vec_out, Y_vec_out = cassi_system.propagate_coded_aperture_grid()

    model = model_generator(model_name, None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    mse = torch.nn.MSELoss()

    model = train_recons(model, cassi_system, nb_epochs)
    test_recons(model, cassi_system)