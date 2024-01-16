from simca import load_yaml_config
from simca.CassiSystemOptim import CassiSystemOptim
from simca.CassiSystem import CassiSystem
import numpy as np
import snoop
import matplotlib.pyplot as plt
#import matplotlib
import torch
import time
from pprint import pprint
from simca.cost_functions import evaluate_slit_scanning_straightness, evaluate_center, evaluate_mean_lighting

#matplotlib.use('Agg')
config_dataset = load_yaml_config("simca/configs/dataset.yml")
config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_max_center.yml")
config_patterns = load_yaml_config("simca/configs/pattern.yml")
config_acquisition = load_yaml_config("simca/configs/acquisition.yml")

dataset_name = "indian_pines"

test = "EQUAL_LIGHT"

if test=="SMILE":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim.yml")
    aspect = 0.1
elif test=="EQUAL_LIGHT" or test=="MAX_CENTER":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_max_center.yml")
    aspect = 1


if __name__ == '__main__':
    # Initialize the CASSI system
    cassi_system = CassiSystemOptim(system_config=config_system)

    # time0 = time.time()
        # DATASET : Load the hyperspectral dataset
    cassi_system.load_dataset(dataset_name, config_dataset["datasets directory"])

    # Loop beginning if optics optim.
    cassi_system.update_optical_model(system_config=config_system)
    X_vec_out, Y_vec_out = cassi_system.propagate_coded_aperture_grid()

    num_iterations = 3000  # Define num_iterations as needed
    # Ensure array_x_positions is a tensor with gradient tracking

    if test == "EQUAL_LIGHT":
        cassi_system.generate_custom_pattern_parameters_slit_width(nb_slits=15, nb_rows=3, start_width = 1)

    cassi_system.array_x_positions = cassi_system.array_x_positions.clone().detach().requires_grad_(True)

    # Define the optimizer
    lr = 0.01 # default: 0.005
    optimizer = torch.optim.Adam([cassi_system.array_x_positions], lr=lr)  # Adjust the learning rate as needed

    # Main optimization loop
    for iteration in range(num_iterations):  # Define num_iterations as needed
        optimizer.zero_grad()  # Clear previous gradients
        if test == "EQUAL_LIGHT":
            pattern = cassi_system.generate_custom_slit_pattern_width(start_position = "line")
        else:
            pattern = cassi_system.generate_custom_slit_pattern()
        #print(pattern[:, pattern.shape[1]//2-4:pattern.shape[1]//2+4])
        cassi_system.generate_filtering_cube()
        if test == "SMILE":
            cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, threshold=0.5)
        elif test == "MAX_CENTER":
            cassi_system.image_acquisition(use_psf=False, chunck_size=50)
            cost_value = evaluate_center(cassi_system.measurement)
        elif test == "EQUAL_LIGHT":
            cassi_system.image_acquisition(use_psf=False, chunck_size=50)
            cost_value = evaluate_mean_lighting(cassi_system.measurement)

        cost_value.backward()  # Perform backpropagation
        # print("Gradients after backward:", cassi_system.array_x_positions.grad)
        optimizer.step()  # Update x positions
        cassi_system.array_x_positions.data = torch.relu(cassi_system.array_x_positions.data) # Prevent the parameters to be negative
        # Optional: Print cost_value every few iterations to monitor progress
        if iteration % 5 == 0:  # Adjust printing frequency as needed
            print(f"Iteration {iteration}, Cost: {cost_value.item()}")
        
        if iteration % 200 == 0:
            plt.imshow(pattern.detach().numpy(), aspect=aspect)
            plt.show()

            plt.imshow(cassi_system.filtering_cube[:, :, 1].detach().numpy(), aspect=aspect)
            plt.show()

            plt.plot(np.sum(cassi_system.filtering_cube[:, :, 1].detach().numpy(),axis=0))
            plt.show()

            if (test=="MAX_CENTER") or (test == "EQUAL_LIGHT"):
                plt.imshow(cassi_system.measurement.detach().numpy())
                plt.show()
                #plt.imshow(cassi_system.panchro.detach().numpy())
                #plt.show()
    print(cassi_system.array_x_positions)
    print(torch.std(cassi_system.measurement.detach()))


    
    # cassi_system.image_acquisition()


