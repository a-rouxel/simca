import pytorch_lightning as pl
from data_handler import CubesDataModule
from optimization_modules_full import JointReconstructionModule_V2
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import datetime


predict_data_dir = "./datasets_reconstruction/mst_datasets/cave_1024_28_test" # Folder where the test dataset is

predict_datamodule = CubesDataModule(predict_data_dir, batch_size=1, num_workers=5, augment=False)

datetime_ = datetime.datetime.now().strftime('%y-%m-%d_%Hh%M')

name = "test_simca_reconstruction_full_binary"
model_name = "dauhst_9"

reconstruction_checkpoint = "./saved_checkpoints/best-checkpoint-recons-only.ckpt"

full_model_checkpoint = "./saved_checkpoints/best-checkpoint-full-binary.ckpt"

mask_model = "learned_mask"

log_dir = 'tb_logs'

train = False
retrain_recons = False
run_on_cpu = False # Set to True if you prefer to run it on cpu

logger = TensorBoardLogger(log_dir, name=name)


reconstruction_module = JointReconstructionModule_V2(model_name,
                                                     log_dir=log_dir+'/'+ name,
                                                     mask_model=mask_model,
                                                     reconstruction_checkpoint=reconstruction_checkpoint,
                                                     full_checkpoint=full_model_checkpoint,
                                                     train_reconstruction=retrain_recons)


max_epoch = 150

if (not run_on_cpu) and (torch.cuda.is_available()):
    trainer = pl.Trainer( logger=logger,
                            accelerator="gpu",
                            max_epochs=max_epoch,
                            log_every_n_steps=1)
else:
    trainer = pl.Trainer( logger=logger,
                        accelerator="cpu",
                        max_epochs=max_epoch,
                        log_every_n_steps=1)

reconstruction_module.eval()
trainer.predict(reconstruction_module, predict_datamodule)
