import pytorch_lightning as pl
from data_handler import CubesDataModule
from optimization_modules import JointReconstructionModule_V1

data_dir = "./datasets_reconstruction/"

datamodule = CubesDataModule(data_dir, batch_size=2, num_workers=1)

name = "testing_simca_reconstruction"


reconstruction_module = JointReconstructionModule_V1()

trainer = pl.Trainer( accelerator="gpu",
                        max_epochs=500,
                        log_every_n_steps=100)

trainer.fit(reconstruction_module, datamodule)
