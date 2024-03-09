import matplotlib.pyplot as plt

from simca.functions_general_purpose import load_yaml_config
from simca.CassiSystemOptim import CassiSystemOptim
import torch

config = "simca/configs/cassi_system_optim_optics_full_triplet_dd_cassi.yml"
config_patterns = load_yaml_config("simca/configs/pattern.yml")

config_system = load_yaml_config(config)
cassi_system = CassiSystemOptim(system_config=config_system)
cassi_system.propagate_coded_aperture_grid()

pattern = cassi_system.generate_2D_pattern(config_patterns,nb_of_patterns=1)


# Compute the 2D FFT of the tensor
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

print("spectral_flatness: ", spectral_flatness)