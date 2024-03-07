import pytorch_lightning as pl
from data_handler import CubesDataModule
from optimization_modules import JointReconstructionModule_V1
import torch

data_dir = "./datasets_reconstruction/mst_datasets/cave_1024_28"
#data_dir = "/local/users/ademaio/lpaillet/mst_datasets/cave_1024_28"

datamodule = CubesDataModule(data_dir, batch_size=16, num_workers=11)

name = "testing_simca_reconstruction"

model_name = "mst_plus_plus"

reconstruction_module = JointReconstructionModule_V1(model_name)

if torch.cuda.is_available():
    trainer = pl.Trainer( accelerator="gpu",
                            max_epochs=500,
                            log_every_n_steps=5)
else:
    trainer = pl.Trainer( accelerator="cpu",
                            max_epochs=500,
                            log_every_n_steps=5)

trainer.fit(reconstruction_module, datamodule)
