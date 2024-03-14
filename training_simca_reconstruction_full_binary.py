import pytorch_lightning as pl
from data_handler import CubesDataModule
from optimization_modules_full import JointReconstructionModule_V2
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

data_dir = "./datasets_reconstruction/mst_datasets/cave_1024_28_train" # Folder where the train dataset is

datamodule = CubesDataModule(data_dir, batch_size=4, num_workers=5, augment=True)

name = "training_simca_reconstruction_full_binary"
model_name = "dauhst_9"

reconstruction_checkpoint = f"./checkpoints/best-checkpoint_{model_name}.ckpt"

mask_model = "learned_mask"

log_dir = 'tb_logs'

train = True
retrain_recons = True # Set to False if you don't want to fine-tune the reconstruction network
run_on_cpu = False # Set to True if you prefer to run it on cpu

logger = TensorBoardLogger(log_dir, name=name)


early_stop_callback = EarlyStopping(
                            monitor='val_loss',  # Metric to monitor
                            patience=500,        # Number of epochs to wait for improvement
                            verbose=True,
                            mode='min'          # 'min' for metrics where lower is better, 'max' for vice versa
                            )

checkpoint_callback = ModelCheckpoint(
    monitor='val_ssim_loss',      # Metric to monitor
    dirpath='checkpoints_full_binary/',  # Directory path for saving checkpoints
    filename=f'best-checkpoint_full_binary_{model_name}',  # Checkpoint file name
    save_top_k=1,            # Save the top k models
    mode='max',              # 'min' for metrics where lower is better, 'max' for vice versa
    save_last=True           # Additionally, save the last checkpoint to a file named 'last.ckpt'
)


reconstruction_module = JointReconstructionModule_V2(model_name,
                                                     log_dir=log_dir+'/'+ name,
                                                     mask_model=mask_model,
                                                     reconstruction_checkpoint=reconstruction_checkpoint,
                                                     full_checkpoint=None,
                                                     train_reconstruction=retrain_recons)


max_epoch = 150

if (not run_on_cpu) and (torch.cuda.is_available()):
    trainer = pl.Trainer( logger=logger,
                            accelerator="gpu",
                            max_epochs=max_epoch,
                            log_every_n_steps=30,
                            callbacks=[early_stop_callback, checkpoint_callback])
else:
    trainer = pl.Trainer( logger=logger,
                        accelerator="cpu",
                        max_epochs=max_epoch,
                        log_every_n_steps=30,
                        callbacks=[early_stop_callback, checkpoint_callback])

trainer.fit(reconstruction_module, datamodule)
