import pytorch_lightning as pl
import torch
import torch.nn as nn
from simca.CassiSystem_lightning import CassiSystemOptim
from simca import  load_yaml_config
import matplotlib.pyplot as plt


class JointReconstructionModule_V1(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # TODO : use a real reconstruction module
        # self.reconstruction_model = ReconstructionModel()
        self.reconstruction_model = EmptyModule()
        self.loss_fn = nn.MSELoss()

    def on_validation_start(self,stage=None):
        print("---VALIDATION START---")

        config_system = load_yaml_config("simca/configs/cassi_system_optim_optics_full_triplet_sd_cassi.yml")
        self.config_patterns = load_yaml_config("simca/configs/pattern.yml")
        self.cassi_system = CassiSystemOptim(system_config=config_system)
        self.cassi_system.propagate_coded_aperture_grid()

    def forward(self, x):
        print("---FORWARD---")

        hyperspectral_cube, wavelengths = x
        hyperspectral_cube = hyperspectral_cube.permute(0, 3, 2, 1)
        batch_size, H, W, C = hyperspectral_cube.shape

        # generate pattern
        pattern = self.cassi_system.generate_2D_pattern(self.config_patterns,nb_of_patterns=batch_size)
        pattern = pattern.to(self.device)

        # generate first acquisition with simca
        acquired_image1 = self.cassi_system.image_acquisition(hyperspectral_cube, pattern,wavelengths)
        displacement_in_pix = self.cassi_system.get_displacement_in_pixels(dataset_wavelengths=wavelengths)
        print("displacement_in_pix", displacement_in_pix)

        # vizualize first image acquisition
        plt.imshow(acquired_image1[0, :, :].cpu().detach().numpy())
        plt.show()

        # process first acquisition with reconstruction model
        # TODO : replace by the real reconstruction model
        reconstructed_cube = acquired_image1

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
        hyperspectral_cube = hyperspectral_cube.permute(0, 3, 2, 1)

        print("y_hat shape", y_hat.shape)
        print("hyperspectral_cube shape", hyperspectral_cube.shape)

        loss = self.loss_fn(y_hat, hyperspectral_cube)

        return loss, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    
class EmptyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.useless_linear = nn.Linear(1, 1)
    def forward(self, x):
        return x
