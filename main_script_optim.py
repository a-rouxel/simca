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

#matplotlib.use('Agg')
config_dataset = load_yaml_config("simca/configs/dataset.yml")
config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_max_center.yml")
config_patterns = load_yaml_config("simca/configs/pattern.yml")
config_acquisition = load_yaml_config("simca/configs/acquisition.yml")

dataset_name = "indian_pines"

test = "SMILE"

algo = "ADAM"

if test=="SMILE":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_smile.yml")
    aspect = 0.1
elif test=="EQUAL_LIGHT" or test=="MAX_CENTER":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_max_center.yml")
    aspect = 1
elif test == "SMILE_mono":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_smile_mono.yml")
    aspect = 1
elif test == "MAX_LIGHT":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_max_lighting.yml")
    aspect = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    cassi_system.X_coordinates_propagated_coded_aperture = cassi_system.X_coordinates_propagated_coded_aperture.to(device)
    cassi_system.Y_coordinates_propagated_coded_aperture = cassi_system.Y_coordinates_propagated_coded_aperture.to(device)
    cassi_system.X_detector_coordinates_grid = cassi_system.X_detector_coordinates_grid.to(device)
    cassi_system.Y_detector_coordinates_grid = cassi_system.Y_detector_coordinates_grid.to(device)

    num_iterations = 1000  # Define num_iterations as needed

    resulting_image = torch.zeros((cassi_system.system_config["detector"]["number of pixels along Y"], cassi_system.system_config["detector"]["number of pixels along X"]))

    first_pos = 0.2
    last_pos = 0.8
    step_pos = 0.6/1

    image_counter = 0

    patterns1 = []
    patterns2 = []

    #for position in np.linspace(0.4, 0.6, np.round(cassi_system.system_config["coded aperture"]["number of pixels along X"]*0.2).astype('int')):
    for position in np.arange(first_pos, last_pos, step_pos):
        image_counter += 1 
        print(f"===== Start of image acquisition {image_counter} =====")
        convergence_counter = 0 # Counter to check convergence
        max_cnt = 25
        min_cost_value = np.inf 

        if test == "EQUAL_LIGHT":
            cassi_system.generate_custom_pattern_parameters_slit_width(nb_slits=10, nb_rows=3, start_width = sigma)
        elif test == "MAX_LIGHT":
            cassi_system.generate_custom_pattern_parameters_slit_width(nb_slits=1, nb_rows=cassi_system.system_config["detector"]["number of pixels along Y"], start_width = sigma)
        elif (test == "SMILE") or (test == "SMILE_mono"):
            if image_counter==1:
                cassi_system.generate_custom_pattern_parameters_slit(position=position)
            else:
                cassi_system.generate_custom_pattern_parameters_slit(position=position+(cassi_system.array_x_positions-prev_position))

        prev_position = position

        pos_on_detector = round(position*cassi_system.system_config["detector"]["number of pixels along X"])

        cassi_system.array_x_positions = cassi_system.array_x_positions.to(device)
        # Ensure array_x_positions is a tensor with gradient tracking
        cassi_system.array_x_positions = cassi_system.array_x_positions.clone().detach().requires_grad_(True)
        best_x = cassi_system.array_x_positions.clone().detach()

        # Define the optimizer
        if algo == "LBFGS":
            lr = 0.05 # default: 0.05
            optimizer = torch.optim.LBFGS([cassi_system.array_x_positions], lr=lr)  # Adjust the learning rate as needed
        elif algo == "ADAM":
            lr = 0.005 # default: 0.005
            optimizer = torch.optim.Adam([cassi_system.array_x_positions], lr=lr)  # Adjust the learning rate as needed

        # Main optimization loop
        for iteration in range(num_iterations):  # Define num_iterations as needed
            
            #print(pattern[:, pattern.shape[1]//2-4:pattern.shape[1]//2+4])
            if algo == "LBFGS":
                def closure():
                    optimizer.zero_grad()
                    if test == "EQUAL_LIGHT":
                        pattern = cassi_system.generate_custom_slit_pattern_width(start_pattern = "line")
                    elif test == "MAX_LIGHT":
                        pattern = cassi_system.generate_custom_slit_pattern_width(start_pattern = "line", start_position = position)
                    else:
                        pattern = cassi_system.generate_custom_slit_pattern()

                    cassi_system.pattern = cassi_system.pattern.to(device)
                    cassi_system.generate_filtering_cube()
                    cassi_system.filtering_cube = cassi_system.filtering_cube.to(device)
                    if (test == "SMILE"):
                        cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, sigma = sigma, pos_slit = 0.1+position)
                    elif (test == "SMILE_mono"):
                        cassi_system.image_acquisition(use_psf = False, chunck_size = 50)
                        cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, sigma = sigma, pos_slit = position)
                    elif test == "MAX_CENTER":
                        cassi_system.image_acquisition(use_psf=False, chunck_size=50)
                        cost_value = evaluate_center(cassi_system.measurement)
                    elif test == "EQUAL_LIGHT":
                        cassi_system.image_acquisition(use_psf=False, chunck_size=50)
                        cost_value = evaluate_mean_lighting(cassi_system.measurement)
                    elif test == "MAX_LIGHT":
                        cassi_system.image_acquisition(use_psf=False, chunck_size=50)
                        cost_value = evaluate_max_lighting(cassi_system.measurement, pos_on_detector)
                    cost_value.backward()
                    return cost_value
                optimizer.step(closure) 

                if (test == "SMILE"):
                    cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, sigma = sigma, pos_slit = position)
                elif (test == "SMILE_mono"):
                    cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, sigma = sigma, pos_slit = position)
                elif test == "MAX_CENTER":
                    cost_value = evaluate_center(cassi_system.measurement)
                elif test == "EQUAL_LIGHT":
                    cost_value = evaluate_mean_lighting(cassi_system.measurement)
                elif test == "MAX_LIGHT":
                    cost_value = evaluate_max_lighting(cassi_system.measurement, pos_on_detector)

            elif algo == "ADAM":
                optimizer.zero_grad()  # Clear previous gradients
                if test == "EQUAL_LIGHT":
                    pattern = cassi_system.generate_custom_slit_pattern_width(start_pattern = "line") 
                elif test == "MAX_LIGHT":
                    pattern = cassi_system.generate_custom_slit_pattern_width(start_pattern = "line", start_position = position)
                else:
                    pattern = cassi_system.generate_custom_slit_pattern()

                cassi_system.pattern = cassi_system.pattern.to(device)
                #print(pattern[:, pattern.shape[1]//2-4:pattern.shape[1]//2+4])
                cassi_system.generate_filtering_cube()
                cassi_system.filtering_cube = cassi_system.filtering_cube.to(device)
                if (test == "SMILE"):
                    cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, sigma = sigma, pos_slit = position)
                elif (test == "SMILE_mono"):
                    cost_value = evaluate_slit_scanning_straightness(cassi_system.filtering_cube, sigma = sigma, pos_slit = position)
                elif test == "MAX_CENTER":
                    cassi_system.image_acquisition(use_psf=False, chunck_size=50)
                    cost_value = evaluate_center(cassi_system.measurement)
                elif test == "EQUAL_LIGHT":
                    cassi_system.image_acquisition(use_psf=False, chunck_size=50)
                    cost_value = evaluate_mean_lighting(cassi_system.measurement)
                elif test == "MAX_LIGHT":
                    cassi_system.image_acquisition(use_psf=False, chunck_size=50)
                    cost_value = evaluate_max_lighting(cassi_system.measurement, pos_on_detector)
                cost_value.backward()
                optimizer.step() 
            
            if cost_value < min_cost_value:
                min_cost_value = cost_value
                convergence_counter = 0
                best_x = cassi_system.array_x_positions.clone().detach()
            else:
                convergence_counter+=1
            
            if (iteration >= 50) and (convergence_counter >= max_cnt): # If loss didn't decrease in 25 steps, break
                break
            # print("Gradients after backward:", cassi_system.array_x_positions.grad)
            
            cassi_system.array_x_positions.data = torch.relu(cassi_system.array_x_positions.data) # Prevent the parameters to be negative
            # Optional: Print cost_value every few iterations to monitor progress
            if iteration % 5 == 0:  # Adjust printing frequency as needed
                print(f"Iteration {iteration}, Cost: {cost_value.item()}")
            
            if iteration % 200 == -1:
                print(f"Exec time: {time.time() - time_start}s")
                plt.imshow(cassi_system.pattern.detach().numpy(), aspect=aspect)
                plt.show()

                plt.imshow(cassi_system.filtering_cube[:, :, 0].detach().numpy(), aspect=aspect)
                plt.show()

                plt.plot(np.sum(cassi_system.filtering_cube[:, :, 0].detach().numpy(),axis=0))
                plt.show()

                if (test=="MAX_CENTER") or (test == "EQUAL_LIGHT") or (test == "SMILE_mono") or (test == "MAX_LIGHT"):
                    plt.imshow(cassi_system.measurement.detach().numpy())
                    plt.show()
                    #plt.imshow(cassi_system.panchro.detach().numpy())
                    #plt.show()
        cassi_system.array_x_positions.data = torch.relu(best_x.data)

        if test == "SMILE":
            pattern = cassi_system.generate_custom_slit_pattern().detach().numpy()
            patterns1.append(pattern)
            if image_counter == 1:
                patterns2.append(np.zeros_like(cassi_system.generate_custom_slit_pattern().detach().numpy()))
                saved_first_x_positions = cassi_system.array_x_positions.detach().numpy()
            else:
                cassi_system.generate_custom_pattern_parameters_slit(position=position+(saved_first_x_positions-first_pos))
                moved_pattern = cassi_system.generate_custom_slit_pattern().detach().numpy()

                cassi_system.array_x_positions.data = torch.relu(best_x.data)

                patterns2.append(pattern - moved_pattern)
        elif test == "MAX_LIGHT":
            pattern = cassi_system.generate_custom_slit_pattern_width(start_pattern = "line", start_position = position)
            cassi_system.generate_filtering_cube()
            measurement = cassi_system.image_acquisition(use_psf=False, chunck_size=50).detach().numpy()
            patterns1.append(measurement)
            
            patterns2.append(pattern.detach().numpy())
        print(f"Starting position: {position}")
        print(cassi_system.array_x_positions)
    #print(torch.std(cassi_system.measurement.detach()))

    print(f"Min cost: {min_cost_value}")
    print(f"Exec time: {time.time() - time_start}s")

    fig1 = plt.figure()
    im1 = plt.imshow(patterns1[0], animated = True, aspect=aspect)
    plt.colorbar()

    fig2 = plt.figure()
    im2 = plt.imshow(patterns2[0], animated = True, aspect=aspect)
    plt.colorbar()
    def update(i):
        im1.set_array(patterns1[i])
        return im1,
    def update2(i):
        im2.set_array(patterns2[i])
        return im2,

    if test == "SMILE":
        interval = 200
    elif test == "MAX_LIGHT":
        interval = 2000

    animation_fig1 = anim.FuncAnimation(fig1, update, frames=len(patterns1), interval = interval, repeat=True)
    animation_fig2 = anim.FuncAnimation(fig2, update2, frames=len(patterns2), interval = interval, repeat=True)

    
    plt.show()
    animation_fig1.save("test.gif")
    animation_fig2.save("test_diff.gif")
    """ plt.imshow(cassi_system.pattern.detach().numpy(), aspect=aspect)
    plt.show()

    plt.imshow(cassi_system.filtering_cube[:, :, 0].detach().numpy(), aspect=aspect)
    plt.show()

    plt.plot(np.sum(cassi_system.filtering_cube[:, :, 0].detach().numpy(),axis=0))
    plt.show()

    if (test=="MAX_CENTER") or (test == "EQUAL_LIGHT") or (test == "SMILE_mono"):
        plt.imshow(cassi_system.measurement.detach().numpy())
        plt.show()
 """
