import pytorch_lightning as pl
import torch
import torch.nn as nn
from simca.CassiSystem_lightning import CassiSystemOptim
from MST.simulation.train_code.architecture import *
from simca import  load_yaml_config
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from simca.functions_acquisition import *
from piqa import SSIM
from torch.utils.tensorboard import SummaryWriter
import io
import torchvision.transforms as transforms
from PIL import Image
from optimization_modules_with_resnet import UnetModel

class ResnetOnly(pl.LightningModule):

    def __init__(self, model_name,log_dir="tb_logs", reconstruction_checkpoint=None):
        super().__init__()

        self.mask_generation = UnetModel(classes=1,encoder_weights=None,in_channels=1)
        self.loss_fn = nn.MSELoss()
        self.writer = SummaryWriter(log_dir)


    def _normalize_data_by_itself(self, data):
        # Calculate the mean and std for each batch individually
        # Keep dimensions for broadcasting
        mean = torch.mean(data, dim=[1, 2], keepdim=True)
        std = torch.std(data, dim=[1, 2], keepdim=True)

        # Normalize each batch by its mean and std
        normalized_data = (data - mean) / std
        return normalized_data


    def forward(self, x, pattern=None):
        print("---FORWARD---")

        hyperspectral_cube, wavelengths = x
        hyperspectral_cube = hyperspectral_cube.permute(0, 2, 3, 1).to(self.device)
        batch_size, H, W, C = hyperspectral_cube.shape

        #generate stupid acq
        self.acquisition = torch.sum(hyperspectral_cube, dim=-1)
        self.acquisition = self.acquisition.flip(1)
        self.acquisition = self.acquisition.flip(2) 
        self.acquisition = self.acquisition.unsqueeze(1).float()

        print("acquisition shape: ", self.acquisition.shape)
        plt.imshow(self.acquisition[0,0,:,:].cpu().numpy())
        plt.show()

        self.pattern = self.mask_generation(self.acquisition)

        print("pattern shape: ", self.pattern.shape)
        plt.imshow(self.pattern[0,0,:,:].cpu().numpy())
        plt.show()

        return self.pattern


    def training_step(self, batch, batch_idx):
        print("Training step")

        loss = self._common_step(batch, batch_idx)

        input_images = self._convert_output_to_images(self._normalize_image_tensor(self.input_image))
        patterns = self._convert_output_to_images(self._normalize_image_tensor(self.pattern))

        if self.global_step % 30 == 0:
            self._log_images('train/input_images', input_images, self.global_step)
            self._log_images('train/patterns', patterns, self.global_step)

        self.log_dict(
            { "train_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )


        return {"loss": loss}

    def _normalize_image_tensor(self, tensor):
        # Normalize the tensor to the range [0, 1]
        min_val = tensor.min()
        max_val = tensor.max()
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor

    def validation_step(self, batch, batch_idx):

        print("Validation step")
        loss  = self._common_step(batch, batch_idx)

        self.log_dict(
            { "val_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        print("Test step")
        loss = self._common_step(batch, batch_idx)
        self.log_dict(
            { "test_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        print("Predict step")
        loss = self._common_step(batch, batch_idx)

        return loss

    def _common_step(self, batch, batch_idx):

        output_pattern = self.forward(batch)

        sum_result = torch.mean(output_pattern,dim=(1,2))
        sum_final = torch.sum(sum_result - 0.5)
        loss1 = sum_final

        loss2 = calculate_spectral_flatness(output_pattern)
        loss2 = torch.sum(1 - loss2)

        print("mean loss1: ", loss1)
        print("spectral flatness loss 2: ", loss2)

        loss = loss1 + loss2

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        return { "optimizer":optimizer,
                "lr_scheduler":{
                "scheduler":torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6),
                "interval": "epoch"
                }
        }

    def _log_images(self, tag, images, global_step):
        # Convert model output to image grid and log to TensorBoard
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, img_grid, global_step)

    def _convert_output_to_images(self, acquired_images):

        acquired_images = acquired_images.unsqueeze(1)

        # Create a grid of images for visualization
        img_grid = torchvision.utils.make_grid(acquired_images)
        return img_grid

    def plot_spectral_filter(self,ref_hyperspectral_cube,recontructed_hyperspectral_cube):


        batch_size, y,x, lmabda_ = ref_hyperspectral_cube.shape

        # Create a figure with subplots arranged horizontally
        fig, axs = plt.subplots(1, batch_size, figsize=(batch_size * 5, 4))  # Adjust figure size as needed

        # Check if batch_size is 1, axs might not be iterable
        if batch_size == 1:
            axs = [axs]

        # Plot each spectral filter in its own subplot
        for i in range(batch_size):
            colors = ['b', 'g', 'r']
            for j in range(3):
                pix_j_row_value = np.random.randint(0,y)
                pix_j_col_value = np.random.randint(0,x)

                pix_j_ref = ref_hyperspectral_cube[i, pix_j_row_value,pix_j_col_value,:].cpu().detach().numpy()
                pix_j_reconstructed = recontructed_hyperspectral_cube[i, pix_j_row_value,pix_j_col_value,:].cpu().detach().numpy()
                axs[i].plot(pix_j_reconstructed, label="pix reconstructed" + str(j),c=colors[j])
                axs[i].plot(pix_j_ref, label="pix" + str(j), linestyle='--',c=colors[j])

            axs[i].set_title(f"Reconstruction quality")

            axs[i].set_xlabel("Wavelength index")
            axs[i].set_ylabel("pix values")
            axs[i].grid(True)

        plt.legend()
        # Adjust layout
        plt.tight_layout()

        # Create a buffer to save plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # Convert PNG buffer to PIL Image
        image = Image.open(buf)

        # Convert PIL Image to Tensor
        image_tensor = transforms.ToTensor()(image)
        return image_tensor


def subsample(input, origin_sampling, target_sampling):
    [bs, row, col, nC] = input.shape
    indices = torch.zeros(len(target_sampling), dtype=torch.int)
    for i in range(len(target_sampling)):
        sample = target_sampling[i]
        idx = torch.abs(origin_sampling-sample).argmin()
        indices[i] = idx
    return input[:,:,:,indices]

def expand_mask_3d(mask_batch):
    if len(mask_batch.shape)==3:
        mask3d = mask_batch.unsqueeze(-1).repeat((1, 1, 1, 28))
    else:
        mask3d = mask_batch.repeat((1, 1, 1, 28))
    mask3d = torch.permute(mask3d, (0, 3, 1, 2))
    return mask3d
    
class EmptyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.useless_linear = nn.Linear(1, 1)
    def forward(self, x):
        return x
    
def calculate_spectral_flatness(pattern):

    fft_result = torch.fft.fft2(pattern)

    # Calculate the Power Spectrum
    power_spectrum = torch.abs(fft_result)**2

    # Calculate the Geometric Mean of the power spectrum
    # Use torch.log and torch.exp for differentiability, adding a small epsilon to avoid log(0)
    epsilon = 1e-10
    geometric_mean = torch.exp(torch.mean(torch.log(power_spectrum + epsilon)))

    # Calculate the Arithmetic Mean of the power spectrum
    arithmetic_mean = torch.mean(power_spectrum)

    # Compute the Spectral Flatness
    spectral_flatness = geometric_mean / arithmetic_mean

    return spectral_flatness
