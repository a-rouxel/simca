import pytorch_lightning as pl
from data_handler import CubesDataModule
from optimization_modules_with_resnet import JointReconstructionModule_V2
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch


# data_dir = "./datasets_reconstruction/"
data_dir = "/local/users/ademaio/lpaillet/mst_datasets/cave_1024_28"
datamodule = CubesDataModule(data_dir, batch_size=4, num_workers=5)


name = "testing_simca_reconstruction_full"
model_name = "dauhst_9"

log_dir = 'tb_logs'

logger = TensorBoardLogger(log_dir, name=name)

early_stop_callback = EarlyStopping(
                            monitor='val_loss',  # Metric to monitor
                            patience=20,        # Number of epochs to wait for improvement
                            verbose=True,
                            mode='min'          # 'min' for metrics where lower is better, 'max' for vice versa
                            )

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',      # Metric to monitor
    dirpath='checkpoints_with_resnet/',  # Directory path for saving checkpoints
    filename='best-checkpoint',  # Checkpoint file name
    save_top_k=1,            # Save the top k models
    mode='min',              # 'min' for metrics where lower is better, 'max' for vice versa
    save_last=True           # Additionally, save the last checkpoint to a file named 'last.ckpt'
)

reconstruction_module = JointReconstructionModule_V2(model_name,
                                                     log_dir=log_dir+'/'+ name,
                                                     reconstruction_checkpoint=None)

# if torch.cuda.is_available():
#     trainer = pl.Trainer( logger=logger,
#                             accelerator="gpu",
#                             max_epochs=500,
#                             log_every_n_steps=1)
# else:
trainer = pl.Trainer( logger=logger,
                        accelerator="gpu",
                        max_epochs=500,
                        log_every_n_steps=30,
                        callbacks=[early_stop_callback, checkpoint_callback])

trainer.fit(reconstruction_module, datamodule)
