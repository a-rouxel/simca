import pytorch_lightning as pl
import torch
import torch.nn as nn
from simca.CassiSystem_lightning import CassiSystemOptim
from MST.simulation.train_code.architecture import *
from simca import  load_yaml_config
import matplotlib.pyplot as plt
import numpy as np


class JointReconstructionModule_V1(pl.LightningModule):

    def __init__(self, model_name):
        super().__init__()

        # TODO : use a real reconstruction module
        self.reconstruction_model = model_generator(model_name, None)
        """ if torch.cuda.is_available():
            self.reconstruction_model = self.reconstruction_model.cuda()
        else:
            self.reconstruction_model.to('cpu') """
        #self.reconstruction_model = EmptyModule()
        self.loss_fn = nn.MSELoss()

    def on_validation_start(self,stage=None):
        print("---VALIDATION START---")
        self.config = "simca/configs/cassi_system_optim_optics_full_triplet_sd_cassi.yml"
        # self.config = "simca/configs/cassi_system_optim_optics_full_triplet_sd_cassi_shifted.yml"
        config_system = load_yaml_config(self.config)
        self.config_patterns = load_yaml_config("simca/configs/pattern.yml")
        self.cassi_system = CassiSystemOptim(system_config=config_system)
        self.cassi_system.propagate_coded_aperture_grid()

    def forward(self, x):
        print("---FORWARD---")

        hyperspectral_cube, wavelengths = x
        hyperspectral_cube = hyperspectral_cube.permute(0, 3, 2, 1).to(self.device)
        batch_size, H, W, C = hyperspectral_cube.shape
        # print(f"batch size:{batch_size}")
        # generate pattern
        pattern = self.cassi_system.generate_2D_pattern(self.config_patterns,nb_of_patterns=batch_size)
        pattern = pattern.to(self.device)

        plt.imshow(pattern[0, :, :].cpu().detach().numpy())
        plt.show()

        # print(f"pattern_size: {pattern.shape}")

        # generate first acquisition with simca
        acquired_image1 = self.cassi_system.image_acquisition(hyperspectral_cube, pattern, wavelengths)
        filtering_cubes = subsample(self.cassi_system.filtering_cube, np.linspace(450, 650, self.cassi_system.filtering_cube.shape[-1]), np.linspace(450, 650, 28))
        filtering_cubes = filtering_cubes.permute(0, 3, 1, 2).float().to(self.device)
        filtering_cubes = torch.flip(filtering_cubes, dims=(2,3)) # -1 magnification
        displacement_in_pix = self.cassi_system.get_displacement_in_pixels(dataset_wavelengths=wavelengths)
        #print("displacement_in_pix", displacement_in_pix)

        # # vizualize first image acquisition
        # plt.imshow(acquired_image1[0, :, :].cpu().detach().numpy())
        # plt.show()

        # process first acquisition with reconstruction model
        # TODO : replace by the real reconstruction model
        if self.config == "simca/configs/cassi_system_optim_optics_full_triplet_sd_cassi.yml":
            acquired_cubes = acquired_image1.unsqueeze(1).repeat((1, 28, 1, 1)).float().to(self.device) # b x W x R x C
            acquired_cubes = torch.flip(acquired_cubes, dims=(2,3)) # -1 magnification

            reconstructed_cube = self.reconstruction_model(acquired_cubes, filtering_cubes)
        else:
            mask_3d = expand_mask_3d(pattern.flip(dims=(1, 2))).float().to(self.device)
            shifted_image = shift_back(acquired_image1.flip(dims=(1, 2)), displacement_in_pix).float().to(self.device)


            for i in range(10):

                fig,ax = plt.subplots(1,2)
                # plt.title(f"acquired_cubes {i}")
                ax[0].imshow(shifted_image[0, i, :, :].cpu().detach().numpy())
                ax[1].imshow(filtering_cubes[0, i, :, :].cpu().detach().numpy())
                plt.show()

            reconstructed_cube = self.reconstruction_model(shifted_image, mask_3d)

        

        #print(acquired_cubes.shape)
        #print(filtering_cubes.shape)
        
        # 

        return reconstructed_cube


    def training_step(self, batch, batch_idx):
        print("Training step")

        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log_dict(
            { "train_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "scores":y_hat, "y":y}

    def validation_step(self, batch, batch_idx):

        print("Validation step")
        loss, y_hat, y = self._common_step(batch, batch_idx)

        self.log_dict(
            { "val_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "scores":y_hat, "y":y}

    def test_step(self, batch, batch_idx):
        print("Test step")
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log_dict(
            { "test_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "scores":y_hat, "y":y}

    def predict_step(self, batch, batch_idx):
        print("Predict step")
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log('predict_step', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _common_step(self, batch, batch_idx):

        y_hat = self.forward(batch)

        hyperspectral_cube, wavelengths = batch
        #hyperspectral_cube = hyperspectral_cube.permute(0, 3, 2, 1)
        hyperspectral_cube = hyperspectral_cube[:,:, 8:-8, 8:-8]

        fig, ax = plt.subplots(1, 1)
        # plt.title(f"acquired_cubes {i}")
        ax.imshow(hyperspectral_cube[0, 0, :, :].cpu().detach().numpy())
        plt.show()

        #print("y_hat shape", y_hat.shape)
        #print("hyperspectral_cube shape", hyperspectral_cube.shape)

        loss = torch.sqrt(self.loss_fn(y_hat, hyperspectral_cube))

        return loss, y_hat, hyperspectral_cube

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        return optimizer

def subsample(input, origin_sampling, target_sampling):
    [bs, row, col, nC] = input.shape
    output = torch.zeros(bs, row, col, len(target_sampling))
    for i in range(len(target_sampling)):
        sample = target_sampling[i]
        idx = np.abs(origin_sampling-sample).argmin()
        output[:,:,:,i] = input[:,:,:,idx]
    return output

def expand_mask_3d(mask_batch):
    if len(mask_batch.shape)==3:
        mask3d = mask_batch.unsqueeze(-1).repeat((1, 1, 1, 28))
    else:
        mask3d = mask_batch.repeat((1, 1, 1, 28))
    mask3d = torch.permute(mask3d, (0, 3, 1, 2))
    return mask3d

def shift_back(inputs, d):  # input [bs,256,310], [bs, 28]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, row).cuda().float()
    d = d[0]
    d -= d.min()
    for i in range(nC):
        shift = int(np.round(d[i]))
        #output[:, i, :, :] = inputs[:, :, step * i:step * i + col - 27 * step] step = 2
        # if shift >=0:
        #     output[:, i, :, :] = inputs[:, :, shift:row+shift]
        # else:
        #     output[:, i, :, :] = inputs[:, :, shift-row:shift]
        output[:, i, :, :] = inputs[:, :, shift:shift + row]
    return output
    
class EmptyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.useless_linear = nn.Linear(1, 1)
    def forward(self, x):
        return x
