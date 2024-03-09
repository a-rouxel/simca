import pytorch_lightning as pl
from data_handler import CubesDataModule
from optimization_modules_with_resnet import JointReconstructionModule_V2
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import datetime


# data_dir = "./datasets_reconstruction/"
data_dir = "/local/users/ademaio/lpaillet/mst_datasets/cave_1024_28"
data_dir = "./datasets_reconstruction/mst_datasets/cave_1024_28"
datamodule = CubesDataModule(data_dir, batch_size=4, num_workers=5)

datetime_ = datetime.datetime.now().strftime('%y-%m-%d_%Hh%M')

name = "testing_simca_reconstruction_full"
model_name = "dauhst_9"
reconstruction_checkpoint = "/home/lpaillet/Documents/simca/tb_logs/testing_simca_reconstruction/version_24/checkpoints/epoch=499-step=18000.ckpt"

log_dir = 'tb_logs'

train = True

logger = TensorBoardLogger(log_dir, name=name)

early_stop_callback = EarlyStopping(
                            monitor='val_loss',  # Metric to monitor
                            patience=40,        # Number of epochs to wait for improvement
                            verbose=True,
                            mode='min'          # 'min' for metrics where lower is better, 'max' for vice versa
                            )

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',      # Metric to monitor
    dirpath='checkpoints_with_resnet/',  # Directory path for saving checkpoints
    filename=f'best-checkpoint_{model_name}_{datetime_}',  # Checkpoint file name
    save_top_k=1,            # Save the top k models
    mode='min',              # 'min' for metrics where lower is better, 'max' for vice versa
    save_last=True           # Additionally, save the last checkpoint to a file named 'last.ckpt'
)

reconstruction_module = JointReconstructionModule_V2(model_name,
                                                     log_dir=log_dir+'/'+ name,
                                                     reconstruction_checkpoint = reconstruction_checkpoint)


if torch.cuda.is_available():
    trainer = pl.Trainer( logger=logger,
                            accelerator="gpu",
                            max_epochs=500,
                            log_every_n_steps=30,
                            callbacks=[early_stop_callback, checkpoint_callback])
else:
    trainer = pl.Trainer( logger=logger,
                        accelerator="gpu",
                        max_epochs=500,
                        log_every_n_steps=30,
                        callbacks=[early_stop_callback, checkpoint_callback])

if train:
    #reconstruction_checkpoint = None
    #trainer.fit(reconstruction_module, datamodule, ckpt_path = reconstruction_checkpoint)
    trainer.fit(reconstruction_module, datamodule)
else:
    trainer.predict(reconstruction_module, datamodule, ckpt_path=reconstruction_checkpoint)