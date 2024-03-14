import pytorch_lightning as pl
from data_handler import CubesDataModule
from optimization_modules import JointReconstructionModule_V1
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch


data_dir = "./datasets_reconstruction/mst_datasets/cave_1024_28_train" # Folder where the train dataset is

datamodule = CubesDataModule(data_dir, batch_size=4, num_workers=11)

name = "training_simca_reconstruction"
model_name = "dauhst_9"

log_dir = 'tb_logs'

train = True
fix_random_pattern = False # Set to True to fix the random pattern to only learn reconstruction for a single fixed pattern
run_on_cpu = False # Set to True if you prefer to run it on cpu


logger = TensorBoardLogger(log_dir, name=name)

early_stop_callback = EarlyStopping(
                            monitor='val_loss',  # Metric to monitor
                            patience=500,        # Number of epochs to wait for improvement
                            verbose=True,
                            mode='min'          # 'min' for metrics where lower is better, 'max' for vice versa
                            )

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',      # Metric to monitor
    dirpath='checkpoints/',  # Directory path for saving checkpoints
    filename=f'best-checkpoint_{model_name}',  # Checkpoint file name
    save_top_k=1,            # Save the top k models
    mode='min',              # 'min' for metrics where lower is better, 'max' for vice versa
    save_last=True           # Additionally, save the last checkpoint to a file named 'last.ckpt'
)

reconstruction_module = JointReconstructionModule_V1(model_name,log_dir=log_dir+'/'+ name,
                                                     reconstruction_checkpoint=None, 
                                                     fix_random_pattern=fix_random_pattern)

max_epoch = 330

if (not run_on_cpu) and (torch.cuda.is_available()):
    trainer = pl.Trainer( logger=logger,
                            accelerator="gpu",
                            max_epochs=max_epoch,
                            log_every_n_steps=1,
                            callbacks=[early_stop_callback, checkpoint_callback])
else:
    trainer = pl.Trainer( logger=logger,
                            accelerator="cpu",
                            max_epochs=max_epoch,
                            log_every_n_steps=1,
                            callbacks=[early_stop_callback, checkpoint_callback])


trainer.fit(reconstruction_module, datamodule)
