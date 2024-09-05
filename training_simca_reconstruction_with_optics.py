import pytorch_lightning as pl
from data_handler import CubesDataModule
from optimization_modules_with_optics import JointReconstructionModule_with_optics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import datetime


data_dir  = "./datasets_reconstruction/"
datamodule = CubesDataModule(data_dir, batch_size=4, num_workers=11)

datetime_ = datetime.datetime.now().strftime('%y-%m-%d_%Hh%M')

name = "testing_simca_reconstruction"
model_name = "dauhst_9"
reconstruction_checkpoint = "./checkpoints/epoch=499-step=18000.ckpt"
# reconstruction_checkpoint = None

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
    dirpath='checkpoints/',  # Directory path for saving checkpoints
    filename=f'best-checkpoint_{model_name}_{datetime_}',  # Checkpoint file name
    save_top_k=1,            # Save the top k models
    mode='min',              # 'min' for metrics where lower is better, 'max' for vice versa
    save_last=True           # Additionally, save the last checkpoint to a file named 'last.ckpt'
)

reconstruction_module = JointReconstructionModule_with_optics(model_name,log_dir=log_dir+'/'+ name,
                                                     reconstruction_checkpoint=reconstruction_checkpoint)


# if torch.cuda.is_available():
#     trainer = pl.Trainer( logger=logger,
#                             accelerator="gpu",
#                             max_epochs=500,
#                             log_every_n_steps=1,
#                             callbacks=[early_stop_callback, checkpoint_callback])
# else:
trainer = pl.Trainer( logger=logger,
                        accelerator="cpu",
                        max_epochs=500,
                        log_every_n_steps=1,
                        callbacks=[early_stop_callback, checkpoint_callback])

if train:
    trainer.fit(reconstruction_module, datamodule)
else:
    #trainer.predict(reconstruction_module, datamodule)
    trainer.predict(reconstruction_module, datamodule, ckpt_path=reconstruction_checkpoint)
