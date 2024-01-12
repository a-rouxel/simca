from simca import load_yaml_config
from simca.CassiSystemOptim import CassiSystemOptim
from simca.CassiSystem import CassiSystem
import numpy as np
import snoop
import matplotlib.pyplot as plt
import torch
import time
from pprint import pprint
from simca.cost_functions import evalute_slit_scanning_straightness

config_dataset = load_yaml_config("simca/configs/dataset.yml")
config_system = load_yaml_config("simca/configs/cassi_system_simple_optim.yml")
config_patterns = load_yaml_config("simca/configs/pattern.yml")
config_acquisition = load_yaml_config("simca/configs/acquisition.yml")

dataset_name = "indian_pines"


# Initialize the CASSI system
cassi_system = CassiSystemOptim(system_config=config_system)

if __name__ == '__main__':

    # time0 = time.time()
        # DATASET : Load the hyperspectral dataset
    cassi_system.load_dataset(dataset_name, config_dataset["datasets directory"])

    # Loop beginning if optics optim.
    cassi_system.update_optical_model(system_config=config_system)
    X_vec_out, Y_vec_out = cassi_system.propagate_coded_aperture_grid()

    num_iterations = 1000  # Define num_iterations as needed
    # Ensure array_x_positions is a tensor with gradient tracking

    cassi_system.array_x_positions = cassi_system.array_x_positions.clone().detach().requires_grad_(True)
    # Define the optimizer
    optimizer = torch.optim.Adam([cassi_system.array_x_positions], lr=0.005)  # Adjust the learning rate as needed

    # Main optimization loop
    for iteration in range(num_iterations):  # Define num_iterations as needed
        optimizer.zero_grad()  # Clear previous gradients

        pattern = cassi_system.generate_custom_slit_pattern()
        cassi_system.generate_filtering_cube()

        cost_value = evalute_slit_scanning_straightness(cassi_system.filtering_cube, threshold=0.5)

        cost_value.backward()  # Perform backpropagation
        # print("Gradients after backward:", cassi_system.array_x_positions.grad)
        optimizer.step()  # Update x positions

        # Optional: Print cost_value every few iterations to monitor progress
        if iteration % 5 == 0:  # Adjust printing frequency as needed
            print(f"Iteration {iteration}, Cost: {cost_value.item()}")
        
        if iteration % 200 == 0:
            plt.imshow(pattern.detach().numpy())
            plt.show()

            plt.imshow(cassi_system.filtering_cube[:, :, 1].detach().numpy())
            plt.show()

            plt.plot(np.sum(cassi_system.filtering_cube[:, :, 1].detach().numpy(),axis=0))
            plt.show()

    
    # cassi_system.image_acquisition()


