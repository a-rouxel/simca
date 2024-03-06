from simca import load_yaml_config
from simca.CassiSystemOptim import CassiSystemOptim
from simca.CassiSystem import CassiSystem
import numpy as np
import snoop
import matplotlib.pyplot as plt
import matplotlib.animation as anim
#import matplotlib
import torch
import time
from pprint import pprint
from simca.cost_functions import evaluate_slit_scanning_straightness, evaluate_center, evaluate_mean_lighting, evaluate_max_lighting

def optim_smile(cassi_system, position, pos_slit_detector, sigma, device,
                algo, lr, num_iterations,  max_iter_cnv, threshold = 0,
                prev_position = None, plot_frequency = None, aspect_plot = 1):
    if prev_position is None:
        cassi_system.generate_custom_pattern_parameters_slit(position=position)
    else:
        cassi_system.generate_custom_pattern_parameters_slit(position=position+prev_position)

    cassi_system.array_x_positions = cassi_system.array_x_positions.to(device)
    # Ensure array_x_positions is a tensor with gradient tracking
    cassi_system.array_x_positions = cassi_system.array_x_positions.clone().detach().requires_grad_(True)
    best_x = cassi_system.array_x_positions.clone().detach()

    convergence_counter = 0 # Counter to check convergence
    min_cost_value = np.inf 

    # Define the optimizer
    if algo == "LBFGS":
        optimizer = torch.optim.LBFGS([cassi_system.array_x_positions], lr=lr)  # Adjust the learning rate as needed
    elif algo == "ADAM":
        optimizer = torch.optim.Adam([cassi_system.array_x_positions], lr=lr)  # Adjust the learning rate as needed
    
    time_start = time.time()
    # Main optimization loop
    for iteration in range(num_iterations):  # Define num_iterations as needed

        if algo == "LBFGS":
            def closure():
                optimizer.zero_grad()
                cassi_system.generate_custom_slit_pattern()

                cassi_system.pattern = cassi_system.pattern.to(device)
                cassi_system.generate_filtering_cube()
                cassi_system.filtering_cube = cassi_system.filtering_cube.to(device)

                cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, device, sigma = sigma, pos_slit = pos_slit_detector)
                cost_value.backward()
                return cost_value
            
            optimizer.step(closure) 

            cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, device, sigma = sigma, pos_slit = pos_slit_detector)

        elif algo == "ADAM":
            optimizer.zero_grad()  # Clear previous gradients
            cassi_system.generate_custom_slit_pattern()

            cassi_system.pattern = cassi_system.pattern.to(device)
            cassi_system.generate_filtering_cube()
            cassi_system.filtering_cube = cassi_system.filtering_cube.to(device)
            cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, device, sigma = sigma, pos_slit = pos_slit_detector)
            cost_value.backward()
            optimizer.step() 
        
        if (cost_value - min_cost_value) < threshold:
            min_cost_value = cost_value
            convergence_counter = 0
            best_x = cassi_system.array_x_positions.clone().detach()
        else:
            convergence_counter+=1
        
        if (iteration >= 50) and (convergence_counter >= max_iter_cnv): # If loss didn't decrease in max_iter_cnv steps, break
            break
        
        cassi_system.array_x_positions.data = torch.relu(cassi_system.array_x_positions.data) # Prevent the parameters to be negative

        # Optional: Print cost_value every few iterations to monitor progress
        if iteration % 5 == 0:  # Adjust printing frequency as needed
            print(f"\nIteration {iteration}, Cost: {cost_value.item()}")

        if plot_frequency is not None:
            if iteration % plot_frequency == 0:
                print(f"Exec time: {time.time() - time_start:.3f}s")
                plt.imshow(cassi_system.pattern.detach().numpy(), aspect=aspect_plot)
                plt.show()

                plt.imshow(cassi_system.filtering_cube[:, :, cassi_system.filtering_cube.shape[2]//2].detach().numpy(), aspect=aspect_plot)
                plt.show()

                plt.plot(np.sum(cassi_system.filtering_cube[:, :, cassi_system.filtering_cube.shape[2]//2].detach().numpy(),axis=0))
                plt.show()
    
    cassi_system.array_x_positions.data = torch.relu(best_x.data)
    cassi_system.generate_custom_slit_pattern()
    cassi_system.pattern = cassi_system.pattern.to(device)
    cassi_system.generate_filtering_cube()
    cassi_system.filtering_cube = cassi_system.filtering_cube.to(device)
    
    return cassi_system

def optim_width(cassi_system, position, target, nb_rows, start_width, device,
                algo, lr, num_iterations,  max_iter_cnv, threshold = 0,
                plot_frequency = None, aspect_plot = 1):
    
    #start_width = torch.rand(size=(1,nb_rows), generator=gen)*1.5-0.75
    #start_width = 0.005
    cassi_system.generate_custom_pattern_parameters_slit_width(nb_slits=1, nb_rows=nb_rows, start_width = start_width) #0.005

    cassi_system.array_x_positions = cassi_system.array_x_positions.to(device)
    # Ensure array_x_positions is a tensor with gradient tracking
    cassi_system.array_x_positions = cassi_system.array_x_positions.clone().detach().requires_grad_(True)

    best_x = cassi_system.array_x_positions.clone().detach()

    convergence_counter = 0 # Counter to check convergence
    min_cost_value = np.inf 

    # Define the optimizer
    if algo == "LBFGS":
        optimizer = torch.optim.LBFGS([cassi_system.array_x_positions], lr=lr)  # Adjust the learning rate as needed
    elif algo == "ADAM":
        optimizer = torch.optim.Adam([cassi_system.array_x_positions], lr=lr)  # Adjust the learning rate as needed
    
    time_start = time.time()
    # Main optimization loop
    for iteration in range(num_iterations):  # Define num_iterations as needed

        if algo == "LBFGS":
            def closure():
                optimizer.zero_grad()
                cassi_system.generate_custom_slit_pattern_width(start_pattern = "corrected", start_position = position)

                cassi_system.pattern = cassi_system.pattern.to(device)
                cassi_system.generate_filtering_cube()
                cassi_system.filtering_cube = cassi_system.filtering_cube.to(device)

                cassi_system.image_acquisition(use_psf=False, chunck_size=cassi_system.system_config["detector"]["number of pixels along Y"])
                cost_value = evaluate_max_lighting(cassi_system.array_x_positions.detach(), cassi_system.measurement, target)
                cost_value.backward()
                return cost_value
            
            optimizer.step(closure) 

            cost_value = evaluate_max_lighting(cassi_system.array_x_positions.detach(), cassi_system.measurement, target)

        elif algo == "ADAM":
            optimizer.zero_grad()  # Clear previous gradients
            cassi_system.generate_custom_slit_pattern_width(start_pattern = "corrected", start_position = position)

            cassi_system.pattern = cassi_system.pattern.to(device)
            cassi_system.generate_filtering_cube()
            cassi_system.filtering_cube = cassi_system.filtering_cube.to(device)
            cassi_system.image_acquisition(use_psf=False, chunck_size=cassi_system.system_config["detector"]["number of pixels along Y"])
            cost_value = evaluate_max_lighting(cassi_system.array_x_positions.detach(), cassi_system.measurement, target)
            cost_value.backward()
            optimizer.step() 
        
        if (cost_value - min_cost_value) < threshold:
            min_cost_value = cost_value
            convergence_counter = 0
            best_x = cassi_system.array_x_positions.clone().detach()
        else:
            convergence_counter+=1
        
        if (iteration >= 50) and (convergence_counter >= max_iter_cnv): # If loss didn't decrease in max_iter_cnv steps, break
            break
        
        cassi_system.array_x_positions.data = torch.relu(cassi_system.array_x_positions.data) # Prevent the parameters to be negative

        # Optional: Print cost_value every few iterations to monitor progress
        if iteration % 5 == 0:  # Adjust printing frequency as needed
            print(f"\nIteration {iteration}, Cost: {cost_value.item()}")

        if plot_frequency is not None:
            if iteration % plot_frequency == 0:
                print(f"Exec time: {time.time() - time_start:.3f}s")
                plt.figure()
                plt.imshow(cassi_system.pattern.detach().numpy(), aspect=aspect_plot)

                plt.figure()
                plt.imshow(cassi_system.filtering_cube[:, :, cassi_system.filtering_cube.shape[2]//2].detach().numpy(), aspect=aspect_plot)

                #plt.figure()
                #plt.plot(np.sum(cassi_system.filtering_cube[:, :, cassi_system.filtering_cube.shape[2]//2].detach().numpy(),axis=0))

                plt.figure()
                plt.imshow(cassi_system.measurement.detach().numpy(), cmap="gray")
                plt.colorbar()
                plt.show()

    cassi_system.array_x_positions.data = torch.relu(best_x.data)
    #cassi_system.array_x_positions.data = best_x.data
    cassi_system.generate_custom_slit_pattern_width(start_pattern = "corrected", start_position = position)
    cassi_system.pattern = cassi_system.pattern.to(device)
    cassi_system.generate_filtering_cube()

    return cassi_system
