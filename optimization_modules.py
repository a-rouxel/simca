import pytorch_lightning as pl
import torch
import torch.nn as nn
from simca.CassiSystemOptim import CassiSystemOptim
from simca import  load_yaml_config


class EmptyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.useless_linear = nn.Linear(1, 1)
    def forward(self, x):
        
        return x


class JointReconstructionModule_V1(pl.LightningModule):

    def __init__(self):
        super().__init__()
        
        self.inittialize_cassi_system()
        #TODO
        # self.reconstruction_model = ReconstructionModel()
        self.reconstruction_model = EmptyModule()

        self.loss_fn = nn.MSELoss()

    def inittialize_cassi_system(self):

        config_system = load_yaml_config("simca/configs/cassi_system_optim_optics_full_triplet_sd_cassi.yml")
        self.config_patterns = load_yaml_config("simca/configs/pattern.yml")
        self.cassi_system = CassiSystemOptim(system_config=config_system)
        self.cassi_system.propagate_coded_aperture_grid()

    
    def forward(self, x):

        # generate random patterns (one for scene in the batch)
        patterns = self.cassi_system.generate_2D_pattern(self.config_patterns)

        # generate first acquisition with simca
        filtering_cube = self.cassi_system.generate_filtering_cube()
        acquired_image1 = self.cassi_system.image_acquisition(x)
        # process first acquisition with reconstruction model
        # TODO : replace by the reconstruction model
        reconstructed_cube = acquired_image1

        return reconstructed_cube


    def training_step(self, batch, batch_idx):

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
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log('predict_step', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _common_step(self, batch, batch_idx):

        hyperspectral_cube = batch
        y_hat = self.forward(hyperspectral_cube)
        loss = self.loss_fn(y_hat, hyperspectral_cube)

        return loss, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    
