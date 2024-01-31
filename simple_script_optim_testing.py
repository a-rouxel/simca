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

test = "SMILE"

if test=="SMILE":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim.yml")
    aspect = 0.1
elif test=="EQUAL_LIGHT" or test=="MAX_CENTER":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_max_center.yml")
    aspect = 1
elif test == "SMILE_mono":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_smile_mono.yml")
    aspect = 1


if __name__ == '__main__':
    time_start = time.time()
    # Initialize the CASSI system
    cassi_system = CassiSystemOptim(system_config=config_system)

    # time0 = time.time()
        # DATASET : Load the hyperspectral dataset
    cassi_system.load_dataset(dataset_name, config_dataset["datasets directory"])

    # Loop beginning if optics optim.
    cassi_system.update_optical_model(system_config=config_system)
    X_vec_out, Y_vec_out = cassi_system.propagate_coded_aperture_grid()

    num_iterations = 3000  # Define num_iterations as needed

    convergence_counter = 0 # Counter to check convergence
    max_cnt = 100
    min_cost_value = np.inf 

    if test == "EQUAL_LIGHT":
        cassi_system.generate_custom_pattern_parameters_slit_width(nb_slits=2, nb_rows=2, start_width = 1)
    elif (test == "SMILE") or (test == "SMILE_mono"):
        cassi_system.generate_custom_pattern_parameters_slit(position=0.2)
    
    # Ensure array_x_positions is a tensor with gradient tracking
    cassi_system.array_x_positions = cassi_system.array_x_positions.clone().detach().requires_grad_(True)

    # Define the optimizer
    lr = 0.005 # default: 0.005
    optimizer = torch.optim.Adam([cassi_system.array_x_positions], lr=lr)  # Adjust the learning rate as needed

    # Main optimization loop
    for iteration in range(num_iterations):  # Define num_iterations as needed
        optimizer.zero_grad()  # Clear previous gradients
        if test == "EQUAL_LIGHT":
            pattern = cassi_system.generate_custom_slit_pattern_width(start_pattern = "line", start_position = 0)
        else:
            pattern = cassi_system.generate_custom_slit_pattern()
        #print(pattern[:, pattern.shape[1]//2-4:pattern.shape[1]//2+4])
        cassi_system.generate_filtering_cube()
        if (test == "SMILE"):
            cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, sigma = 0.75, pos_slit = 0.385)
        elif (test == "SMILE_mono"):
            cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, sigma = 0.75, pos_slit = 0.385)
            cassi_system.image_acquisition(use_psf = False, chunck_size = 50)
        elif test == "MAX_CENTER":
            cassi_system.image_acquisition(use_psf=False, chunck_size=50)
            cost_value = evaluate_center(cassi_system.measurement)
        elif test == "EQUAL_LIGHT":
            cassi_system.image_acquisition(use_psf=False, chunck_size=50)
            cost_value = evaluate_mean_lighting(cassi_system.measurement)
        
        if cost_value < min_cost_value:
            min_cost_value = cost_value
            convergence_counter = 0
        else:
            convergence_counter+=1
        
        if (iteration >= 50) and (convergence_counter >= max_cnt): # If loss didn't decrease in 25 steps, break
            break
        
        cost_value.backward()  # Perform backpropagation
        # print("Gradients after backward:", cassi_system.array_x_positions.grad)
        optimizer.step()  # Update x positions
        cassi_system.array_x_positions.data = torch.relu(cassi_system.array_x_positions.data) # Prevent the parameters to be negative
        # Optional: Print cost_value every few iterations to monitor progress
        if iteration % 5 == 0:  # Adjust printing frequency as needed
            print(f"Iteration {iteration}, Cost: {cost_value.item()}")
        
        if iteration % 200 == 0:
            print(f"Exec time: {time.time() - time_start}s")
            plt.imshow(pattern.detach().numpy(), aspect=aspect)
            plt.show()

            plt.imshow(cassi_system.filtering_cube[:, :, 0].detach().numpy(), aspect=aspect)
            plt.show()

            plt.plot(np.sum(cassi_system.filtering_cube[:, :, 0].detach().numpy(),axis=0))
            plt.show()

            if (test=="MAX_CENTER") or (test == "EQUAL_LIGHT") or (test == "SMILE_mono"):
                #plt.imshow(cassi_system.panchro.detach().numpy(), zorder=3)
                plt.imshow(cassi_system.measurement.detach().numpy(), zorder=5)
                plt.show()
                #plt.imshow(cassi_system.panchro.detach().numpy())
                #plt.show()
    print(cassi_system.array_x_positions)
    #print(torch.std(cassi_system.measurement.detach()))

    print(f"Min cost: {min_cost_value}")
    print(f"Exec time: {time.time() - time_start}s")
    plt.imshow(pattern.detach().numpy(), aspect=aspect)
    plt.show()

    plt.imshow(cassi_system.filtering_cube[:, :, 0].detach().numpy(), aspect=aspect)
    plt.show()

    plt.plot(np.sum(cassi_system.filtering_cube[:, :, 0].detach().numpy(),axis=0))
    plt.show()

    if (test=="MAX_CENTER") or (test == "EQUAL_LIGHT") or (test == "SMILE_mono"):
        plt.imshow(cassi_system.measurement.detach().numpy())
        plt.show()

