import pytorch_lightning as pl
import torch
import torch.nn as nn
from simca.CassiSystem_lightning import CassiSystemOptim
from MST.simulation.train_code.architecture import *
from simca import  load_yaml_config, save_config_file
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from simca.functions_acquisition import *
from piqa import SSIM
from torch.utils.tensorboard import SummaryWriter
import io
import torchvision.transforms as transforms
from torchmetrics.image import PeakSignalNoiseRatio
import os
from PIL import Image

class JointReconstructionModule_V1(pl.LightningModule):

    def __init__(self, model_name,log_dir="tb_logs", reconstruction_checkpoint=None, fix_random_pattern=False):
        super().__init__()

        self.model_name = model_name
        self.reconstruction_model = model_generator(self.model_name, reconstruction_checkpoint)
        self.fix_random_pattern = fix_random_pattern

        self.loss_fn = nn.MSELoss()
        self.ssim_loss = SSIM(window_size=11, n_channels=28)

        # for param in self.reconstruction_model.parameters():
        #     param.requires_grad = False

        self.writer = SummaryWriter(log_dir)

    def on_validation_start(self,stage=None):
        print("---VALIDATION START---")
        self.config = "simca/configs/cassi_system_optim_optics_full_triplet_dd_cassi.yml"

        config_system = load_yaml_config(self.config)
        self.config_patterns = load_yaml_config("simca/configs/pattern.yml")
        self.cassi_system = CassiSystemOptim(system_config=config_system)
        self.cassi_system.propagate_coded_aperture_grid()

    def on_predict_start(self,stage=None):
        print("---PREDICT START---")

        if not os.path.exists('./results'):
            os.makedirs('./results')

        if self.ssim_loss.kernel.shape[0]!=28:
            self.ssim_loss = SSIM(window_size=11, n_channels=28).to(self.device)
        
        self.psnr = PeakSignalNoiseRatio().to(self.device)

        self.config = "simca/configs/cassi_system_optim_optics_full_triplet_dd_cassi.yml"

        config_system = load_yaml_config(self.config)
        self.config_patterns = load_yaml_config("simca/configs/pattern.yml")
        self.cassi_system = CassiSystemOptim(system_config=config_system)
        self.cassi_system.propagate_coded_aperture_grid()

        try:
            if self.fix_random_pattern:
                file_name = "predict_results_recons_fixed_random.yml"
            else:
                file_name = "predict_results_recons.yml"
            self.predict_results = load_yaml_config(file_name)
        except:
            self.predict_results = {}

    def on_predict_end(self):
        if self.fix_random_pattern:
            file_name = "predict_results_recons_fixed_random"
        else:
            file_name = "predict_results_recons"
        save_config_file('./results/' + file_name,self.predict_results,".")

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

        # Generate pattern
        if pattern is None:
            self.pattern = self.cassi_system.generate_2D_pattern(self.config_patterns,nb_of_patterns=batch_size, fix_random_pattern=self.fix_random_pattern)
            self.pattern = self.pattern.to(self.device)
        else:
            self.pattern = pattern.to(self.device)
            self.cassi_system.pattern = pattern.to(self.device)

        # Generate first acquisition with simca
        filtering_cube = self.cassi_system.generate_filtering_cube().to(self.device)
        self.acquired_image1 = self.cassi_system.image_acquisition(hyperspectral_cube, self.pattern, wavelengths).to(self.device)


        filtering_cubes = subsample(filtering_cube, torch.linspace(450, 650, filtering_cube.shape[-1]), torch.linspace(450, 650, 28)).permute((0, 3, 1, 2)).float().to(self.device)

        if self.model_name == "birnat":
            acquisition = self.acquired_image1.float()
            filtering_cubes = filtering_cubes.float()
        elif "dauhst" in self.model_name:
            acquisition = self.acquired_image1.float()

            filtering_cubes_s = torch.sum(filtering_cubes**2,1)
            filtering_cubes_s[filtering_cubes_s==0] = 1
            filtering_cubes = (filtering_cubes.float(), filtering_cubes_s.float())
            
        elif self.model_name == "mst_plus_plus":
            acquisition = self.acquired_image1.unsqueeze(1).repeat((1, 28, 1, 1)).float().to(self.device)

        reconstructed_cube = self.reconstruction_model(acquisition, filtering_cubes)


        return reconstructed_cube


    def training_step(self, batch, batch_idx):
        print("Training step")

        loss, ssim_loss, reconstructed_cube, ref_cube = self._common_step(batch, batch_idx)


        output_images = self._convert_output_to_images(self._normalize_image_tensor(self.acquired_image1))
        patterns = self._convert_output_to_images(self._normalize_image_tensor(self.pattern))
        input_images = self._convert_output_to_images(self._normalize_image_tensor(ref_cube[:,:,:,0]))
        reconstructed_image = self._convert_output_to_images(self._normalize_image_tensor(reconstructed_cube[:,:,:,0]))

        if self.global_step % 30 == 0:
            self._log_images('train/acquisition', output_images, self.global_step)
            self._log_images('train/ground_truth', input_images, self.global_step)
            self._log_images('train/reconstructed', reconstructed_image, self.global_step)
            self._log_images('train/patterns', patterns, self.global_step)

            spectral_filter_plot = self.plot_spectral_filter(ref_cube,reconstructed_cube)

            self.writer.add_image('Spectral Filter', spectral_filter_plot, self.global_step)

        self.log_dict(
            { "train_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.log_dict(
            { "train_ssim_loss": ssim_loss,
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
        loss, ssim_loss, reconstructed_cube, ref_cube = self._common_step(batch, batch_idx)

        self.log_dict(
            { "val_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.log_dict(
            { "val_ssim_loss": ssim_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        print("Test step")
        loss, ssim_loss, reconstructed_cube, ref_cube = self._common_step(batch, batch_idx)
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
        loss, ssim_loss, reconstructed_cube, ref_cube = self._common_step(batch, batch_idx)
        psnr_loss = self.psnr(reconstructed_cube.permute(0, 3, 1, 2), ref_cube.permute(0, 3, 1, 2))

        print("Predict PSNR loss: ", psnr_loss.item())
        print("Predict RMSE loss: ", loss.item())
        print("Predict SSIM loss: ", ssim_loss.item())

        ignored_ids = [2, 3, 9, 10, 11, 12, 13, 17] # Those contain a low amount of signal and are thus less interesting
        if (batch_idx+1) not in ignored_ids:
            try:
                self.predict_results[f"RMSE_scene{batch_idx+1}"].append(loss.item())
                self.predict_results[f"SSIM_scene{batch_idx+1}"].append(ssim_loss.item())
                self.predict_results[f"PSNR_scene{batch_idx+1}"].append(psnr_loss.item())
            except:
                self.predict_results[f"RMSE_scene{batch_idx+1}"] = [loss.item()]
                self.predict_results[f"SSIM_scene{batch_idx+1}"] = [ssim_loss.item()]
                self.predict_results[f"PSNR_scene{batch_idx+1}"] = [psnr_loss.item()]

        if batch_idx == 19-1:
            torch.save(reconstructed_cube, f'./results/recons_cube_random.pt')

        return loss

    def _common_step(self, batch, batch_idx):

        reconstructed_cube = self.forward(batch)
        hyperspectral_cube, wavelengths = batch

        hyperspectral_cube = hyperspectral_cube.permute(0, 2, 3, 1).to(self.device)
        reconstructed_cube = reconstructed_cube.permute(0, 2, 3, 1).to(self.device)
        ref_cube = match_dataset_to_instrument(hyperspectral_cube, reconstructed_cube[0, :, :,0])

        loss = torch.sqrt(self.loss_fn(reconstructed_cube, ref_cube))
        ssim_loss = self.ssim_loss(torch.clamp(reconstructed_cube.permute(0, 3, 1, 2), 0, 1), ref_cube.permute(0, 3, 1, 2))

        return loss, ssim_loss, reconstructed_cube, ref_cube

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


        batch_size, y,x, lambda_ = ref_hyperspectral_cube.shape

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
    # Subsample input from origin_sampling to target_sampling
    indices = torch.zeros(len(target_sampling), dtype=torch.int)
    for i in range(len(target_sampling)):
        sample = target_sampling[i]
        idx = torch.abs(origin_sampling-sample).argmin()
        indices[i] = idx
    return input[:,:,:,indices]
