import pytorch_lightning as pl
from data_handler import CubesDataModule
from optimization_modules import JointReconstructionModule_V1

data_dir = "/local/users/ademaio/lpaillet/mst_datasets/cave_1024_28/"

datamodule = CubesDataModule(data_dir, batch_size=2, num_workers=1)

name = "testing_simca_reconstruction"


reconstruction_module = JointReconstructionModule_V1()

trainer = pl.Trainer( accelerator="cpu",
                        max_epochs=500,
                        log_every_n_steps=100)

trainer.fit(reconstruction_module, datamodule)
