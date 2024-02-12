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
from simca.cost_functions import evaluate_slit_scanning_straightness, evaluate_center, evaluate_mean_lighting, evaluate_max_lighting
from simca.functions_optim import optim_smile, optim_width

#matplotlib.use('Agg')
config_dataset = load_yaml_config("simca/configs/dataset.yml")
config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_max_center.yml")
config_patterns = load_yaml_config("simca/configs/pattern.yml")
config_acquisition = load_yaml_config("simca/configs/acquisition.yml")

dataset_name = "indian_pines"

test = "SMILE"

algo = "LBFGS"

if test=="SMILE":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim.yml")
    aspect = 1
elif test=="EQUAL_LIGHT" or test=="MAX_CENTER":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_max_center.yml")
    aspect = 1
elif test == "MAX_LIGHT":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim.yml")
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
    sigma = 0.75

    num_iterations = 3000  # Define num_iterations as needed

    """ convergence_counter = 0 # Counter to check convergence
    max_cnt = 100
    min_cost_value = np.inf 

    if test == "EQUAL_LIGHT":
        cassi_system.generate_custom_pattern_parameters_slit_width(nb_slits=2, nb_rows=2, start_width = 1)
    elif test == "MAX_LIGHT":
        cassi_system.generate_custom_pattern_parameters_slit_width(nb_slits=1, nb_rows=cassi_system.system_config["detector"]["number of pixels along Y"], start_width = sigma)
    elif (test == "SMILE") or (test == "SMILE_mono"):
        cassi_system.generate_custom_pattern_parameters_slit(position=0.5)
    
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
        elif test == "MAX_LIGHT":
            start_position = torch.tensor([0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.4923,
        0.4859, 0.4915, 0.4934, 0.4972, 0.4996, 0.5003, 0.5009, 0.5013, 0.5033,
        0.5041, 0.5064, 0.5068, 0.5078, 0.5070, 0.5099, 0.5103, 0.5146, 0.5145,
        0.5146, 0.5152, 0.5173, 0.5195, 0.5215, 0.5208, 0.5208, 0.5222, 0.5247,
        0.5272, 0.5287, 0.5285, 0.5240, 0.5283, 0.5282, 0.5288, 0.5288, 0.5291,
        0.5289, 0.5282, 0.5334, 0.5314, 0.5324, 0.5370, 0.5323, 0.5322, 0.5341,
        0.5329, 0.5361, 0.5364, 0.5346, 0.5333, 0.5340, 0.5333, 0.5339, 0.5345,
        0.5359, 0.5349, 0.5364, 0.5344, 0.5341, 0.5346, 0.5353, 0.5345, 0.5347,
        0.5346, 0.5362, 0.5363, 0.5330, 0.5321, 0.5323, 0.5299, 0.5315, 0.5318,
        0.5298, 0.5291, 0.5291, 0.5298, 0.5292, 0.5256, 0.5270, 0.5283, 0.5268,
        0.5255, 0.5245, 0.5200, 0.5205, 0.5207, 0.5190, 0.5188, 0.5144, 0.5122,
        0.5138, 0.5133, 0.5131, 0.5122, 0.5111, 0.5156, 0.5118, 0.5091, 0.5077,
        0.5068, 0.5030, 0.5003, 0.5000, 0.4992, 0.4970, 0.4968, 0.4947, 0.4949,
        0.4935, 0.4983, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000])
            pattern = cassi_system.generate_custom_slit_pattern_width(start_pattern = "corrected", start_position = start_position)
        else:
            pattern = cassi_system.generate_custom_slit_pattern()
        #print(pattern[:, pattern.shape[1]//2-4:pattern.shape[1]//2+4])
        cassi_system.generate_filtering_cube()
        pos_slit = 0.4625
        pos_slit = 0.41
        if (test == "SMILE"):
            cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, sigma = sigma, pos_slit = pos_slit)
        elif (test == "SMILE_mono"):
            cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, sigma = sigma, pos_slit = pos_slit)
            cassi_system.image_acquisition(use_psf = False, chunck_size = 50)
        elif test == "MAX_CENTER":
            cassi_system.image_acquisition(use_psf=False, chunck_size=50)
            cost_value = evaluate_center(cassi_system.measurement)
        elif test == "EQUAL_LIGHT":
            cassi_system.image_acquisition(use_psf=False, chunck_size=50)
            cost_value = evaluate_mean_lighting(cassi_system.measurement)
        elif test == "MAX_LIGHT":
            cassi_system.image_acquisition(use_psf=False, chunck_size=50)
            cost_value = evaluate_max_lighting(cassi_system.measurement, pos_slit)

        if cost_value < min_cost_value:
            min_cost_value = cost_value
            convergence_counter = 0
            best_x = cassi_system.array_x_positions.clone().detach()
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

            if (test=="MAX_CENTER") or (test == "EQUAL_LIGHT") or (test == "SMILE_mono") or (test == "MAX_LIGHT"):
                #plt.imshow(cassi_system.panchro.detach().numpy(), zorder=3)
                plt.imshow(cassi_system.measurement.detach().numpy(), zorder=5)
                plt.show()
                #plt.imshow(cassi_system.panchro.detach().numpy())
                #plt.show()
    cassi_system.array_x_positions.data = torch.relu(best_x.data)

    if test == "EQUAL_LIGHT":
        pattern = cassi_system.generate_custom_slit_pattern_width(start_pattern = "line", start_position = 0)
    elif test == "MAX_LIGHT":
        pattern = cassi_system.generate_custom_slit_pattern_width(start_pattern = "corrected", start_position = start_position)
    else:
        pattern = cassi_system.generate_custom_slit_pattern()
    cassi_system.generate_filtering_cube()
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

    if (test=="MAX_CENTER") or (test == "EQUAL_LIGHT") or (test == "SMILE_mono") or (test == "MAX_LIGHT"):
        plt.imshow(cassi_system.measurement.detach().numpy())
        plt.show()"""
    algo = "ADAM"
    if algo == "LBFGS":
        lr = 0.002 # default: 0.05
    elif algo == "ADAM":
        lr = 0.005 # default: 0.005
    
    max_cnt = 25

    #cassi_system = optim_smile(cassi_system, 0.5, 0.41, sigma, 'cpu', algo, lr, num_iterations, max_cnt, plot_frequency=200)
    start_position = torch.tensor([0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.4923,
        0.4859, 0.4915, 0.4934, 0.4972, 0.4996, 0.5003, 0.5009, 0.5013, 0.5033,
        0.5041, 0.5064, 0.5068, 0.5078, 0.5070, 0.5099, 0.5103, 0.5146, 0.5145,
        0.5146, 0.5152, 0.5173, 0.5195, 0.5215, 0.5208, 0.5208, 0.5222, 0.5247,
        0.5272, 0.5287, 0.5285, 0.5240, 0.5283, 0.5282, 0.5288, 0.5288, 0.5291,
        0.5289, 0.5282, 0.5334, 0.5314, 0.5324, 0.5370, 0.5323, 0.5322, 0.5341,
        0.5329, 0.5361, 0.5364, 0.5346, 0.5333, 0.5340, 0.5333, 0.5339, 0.5345,
        0.5359, 0.5349, 0.5364, 0.5344, 0.5341, 0.5346, 0.5353, 0.5345, 0.5347,
        0.5346, 0.5362, 0.5363, 0.5330, 0.5321, 0.5323, 0.5299, 0.5315, 0.5318,
        0.5298, 0.5291, 0.5291, 0.5298, 0.5292, 0.5256, 0.5270, 0.5283, 0.5268,
        0.5255, 0.5245, 0.5200, 0.5205, 0.5207, 0.5190, 0.5188, 0.5144, 0.5122,
        0.5138, 0.5133, 0.5131, 0.5122, 0.5111, 0.5156, 0.5118, 0.5091, 0.5077,
        0.5068, 0.5030, 0.5003, 0.5000, 0.4992, 0.4970, 0.4968, 0.4947, 0.4949,
        0.4935, 0.4983, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000])
    cassi_system = optim_width(cassi_system, start_position, 0.41, cassi_system.system_config["detector"]["number of pixels along Y"], sigma, 'cpu', algo, lr, num_iterations, max_cnt, plot_frequency = 200)
    
    print(f"Exec time: {time.time() - time_start:.3f}s")
    print(cassi_system.array_x_positions)
    plt.imshow(cassi_system.pattern.detach().numpy(), aspect=aspect)
    plt.show()

    plt.imshow(cassi_system.filtering_cube[:, :, 0].detach().numpy(), aspect=aspect)
    plt.show()

    plt.plot(np.sum(cassi_system.filtering_cube[:, :, 0].detach().numpy(),axis=0))
    plt.show()

    if (test=="MAX_CENTER") or (test == "EQUAL_LIGHT") or (test == "SMILE_mono") or (test == "MAX_LIGHT"):
        plt.imshow(cassi_system.measurement.detach().numpy())
        plt.show()
