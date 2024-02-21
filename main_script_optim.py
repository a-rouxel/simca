from simca import load_yaml_config
from simca.CassiSystemOptim import CassiSystemOptim
from simca.CassiSystem import CassiSystem
import numpy as np
import snoop
import matplotlib.pyplot as plt
import matplotlib.animation as anim
#import matplotlib
import torch
import time, datetime
import os
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

algo = "ADAM"

if test=="SMILE":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim.yml")
    aspect = 1
elif test=="EQUAL_LIGHT" or test=="MAX_CENTER":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_max_center.yml")
    aspect = 1
elif test == "SMILE_mono":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim_smile_mono.yml")
    aspect = 1
elif test == "MAX_LIGHT":
    config_system = load_yaml_config("simca/configs/cassi_system_simple_optim.yml")
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
    sigma = 1.5

    cassi_system.X_coordinates_propagated_coded_aperture = cassi_system.X_coordinates_propagated_coded_aperture.to(device)
    cassi_system.Y_coordinates_propagated_coded_aperture = cassi_system.Y_coordinates_propagated_coded_aperture.to(device)
    cassi_system.X_detector_coordinates_grid = cassi_system.X_detector_coordinates_grid.to(device)
    cassi_system.Y_detector_coordinates_grid = cassi_system.Y_detector_coordinates_grid.to(device)

    num_iterations = 1000  # Define num_iterations as needed

    first_pos = 0.38
    last_pos = 0.8
    step_pos = 0.2/1

    pattern_pos = [0.68, 0.58, 0.48, 0.38]
    pos_slit_detector_list = [20/145, 40/145, 60/145, 80/145]
    image_counter = 0
    
    patterns1 = []
    corrected_patterns1 = []
    smile_positions = []
    corrected_smile_positions = []
    patterns2 = []
    width_values = []
    cubes1 = []
    corrected_cubes1 = []
    cubes2 = []
    acquisitions = []

    prev_position = None

    #for position in np.linspace(0.4, 0.6, np.round(cassi_system.system_config["coded aperture"]["number of pixels along X"]*0.2).astype('int')):
    #for position in np.arange(first_pos, last_pos, step_pos):
    for position_i in range(len(pattern_pos)):
        position = pattern_pos[position_i]
        pos_slit_detector = pos_slit_detector_list[position_i]

        image_counter += 1 
        print(f"===== Start of image acquisition {image_counter} =====")
        max_iter_cnt = 25

        cassi_system = CassiSystemOptim(system_config=config_system)

        # time0 = time.time()
            # DATASET : Load the hyperspectral dataset
        cassi_system.load_dataset(dataset_name, config_dataset["datasets directory"])

        # Loop beginning if optics optim.
        cassi_system.update_optical_model(system_config=config_system)
        X_vec_out, Y_vec_out = cassi_system.propagate_coded_aperture_grid()
        sigma = 1.5

        cassi_system.X_coordinates_propagated_coded_aperture = cassi_system.X_coordinates_propagated_coded_aperture.to(device)
        cassi_system.Y_coordinates_propagated_coded_aperture = cassi_system.Y_coordinates_propagated_coded_aperture.to(device)
        cassi_system.X_detector_coordinates_grid = cassi_system.X_detector_coordinates_grid.to(device)
        cassi_system.Y_detector_coordinates_grid = cassi_system.Y_detector_coordinates_grid.to(device)
        
        # Adjust the learning rate
        if algo == "LBFGS":
            lr = 0.002 # default: 0.05
        elif algo == "ADAM":
            lr = 0.005 # default: 0.005

        """ if position == 0.5:
            pos_slit_detector = 0.41
        elif position == 0.7:
            pos_slit_detector = 0.124 """
        cassi_system = optim_smile(cassi_system, position, pos_slit_detector, sigma, device, algo, lr, num_iterations, max_iter_cnt, prev_position = prev_position, plot_frequency=None)

        pattern = cassi_system.pattern.detach().numpy()
        cube = cassi_system.filtering_cube.detach().numpy()[:,:,0]

        patterns1.append(pattern)
        cubes1.append(cube)
        start_position = cassi_system.array_x_positions.detach().numpy()
        smile_positions.append(start_position)

        diffs = np.diff(start_position)
        diffs_ind = np.nonzero(diffs)[0]
        pos_middle = start_position[diffs_ind.min()+1:diffs_ind.max()+1]
        poly_coeffs = np.polyfit(np.linspace(1,2, len(pos_middle)), pos_middle, deg = 4)
        poly = np.poly1d(poly_coeffs)
        start_position[diffs_ind.min()+1:diffs_ind.max()+1] = poly(np.linspace(1,2, len(pos_middle)))

        corrected_smile_positions.append(start_position)

        start_position = torch.tensor(start_position)

        cassi_system.array_x_positions.data = start_position
        cassi_system.generate_custom_slit_pattern()
        cassi_system.generate_filtering_cube()

        pattern = cassi_system.pattern.detach().numpy()
        cube = cassi_system.filtering_cube.detach().numpy()[:,:,0]
        corrected_patterns1.append(pattern)
        corrected_cubes1.append(cube)

        prev_position = (cassi_system.array_x_positions.detach()-position)

        # Adjust the learning rate
        if algo == "LBFGS":
            lr = 0.002 # default: 0.05
        elif algo == "ADAM":
            lr = 0.01 # default: 0.005

        cassi_system = optim_width(cassi_system, start_position, pos_slit_detector, cassi_system.system_config["detector"]["number of pixels along Y"], sigma, device, algo, lr, num_iterations, max_iter_cnt, plot_frequency = None)

        pattern = cassi_system.pattern.detach().numpy()
        cube = cassi_system.filtering_cube.detach().numpy()[:,:,0]
        acquisition = cassi_system.measurement.detach().numpy()

        patterns2.append(pattern)
        cubes2.append(cube)
        acquisitions.append(acquisition)
        width_values.append(cassi_system.array_x_positions.detach().numpy())

        
            
    #print(torch.std(cassi_system.measurement.detach()))

    print(f"Exec time: {time.time() - time_start}s")

    fig1 = plt.figure()
    im1 = plt.imshow(patterns1[0], animated = True, aspect=aspect)
    plt.colorbar()

    fig1bis = plt.figure()
    im1bis = plt.imshow(corrected_patterns1[0], animated = True, aspect=aspect)
    plt.colorbar()

    fig2 = plt.figure()
    im2 = plt.imshow(cubes1[0], animated = True, aspect=aspect)
    plt.colorbar()

    fig2bis = plt.figure()
    im2bis = plt.imshow(corrected_cubes1[0], animated = True, aspect=aspect)
    plt.colorbar()

    fig3 = plt.figure()
    im3 = plt.imshow(patterns2[0], animated = True, aspect=aspect)
    plt.colorbar()

    fig4 = plt.figure()
    im4 = plt.imshow(cubes2[0], animated = True, aspect=aspect)
    plt.colorbar()

    fig5 = plt.figure()
    im5 = plt.imshow(acquisitions[0], animated = True, aspect=aspect)
    plt.colorbar()

    def update1(i):
        im1.set_array(patterns1[i])
        return im1,
    def update1bis(i):
        im1bis.set_array(corrected_patterns1[i])
        return im1bis,
    def update2(i):
        im2.set_array(cubes1[i])
        return im2,
    def update2bis(i):
        im2bis.set_array(corrected_cubes1[i])
        return im2bis,
    def update3(i):
        im3.set_array(patterns2[i])
        return im3,
    def update4(i):
        im4.set_array(cubes2[i])
        return im4,
    def update5(i):
        im5.set_array(acquisitions[i])
        return im5,

    animation_fig1 = anim.FuncAnimation(fig1, update1, frames=len(patterns1), interval = 1000, repeat=True)
    animation_fig1bis = anim.FuncAnimation(fig1bis, update1bis, frames=len(corrected_patterns1), interval = 1000, repeat=True)
    animation_fig2 = anim.FuncAnimation(fig2, update2, frames=len(cubes1), interval = 1000, repeat=True)
    animation_fig2bis = anim.FuncAnimation(fig2bis, update2bis, frames=len(corrected_cubes1), interval = 1000, repeat=True)
    animation_fig3 = anim.FuncAnimation(fig3, update3, frames=len(patterns2), interval = 1000, repeat=True)
    animation_fig4 = anim.FuncAnimation(fig4, update4, frames=len(cubes2), interval = 1000, repeat=True)
    animation_fig5 = anim.FuncAnimation(fig5, update5, frames=len(acquisitions), interval = 1000, repeat=True)
    
    plt.show()

    folder = datetime.datetime.now().strftime('%y-%m-%d_%Hh%M')
    os.makedirs(f"./results/{folder}") 

    animation_fig1.save(f"./results/{folder}/patterns_smile.gif")
    animation_fig1bis.save(f"./results/{folder}/corrected_patterns_smile.gif")
    animation_fig2.save(f"./results/{folder}/cubes_smile.gif")
    animation_fig2bis.save(f"./results/{folder}/corrected_cubes_smile.gif")
    animation_fig3.save(f"./results/{folder}/patterns_width.gif")
    animation_fig4.save(f"./results/{folder}/cubes_width.gif")
    animation_fig5.save(f"./results/{folder}/acquisitions.gif")

    np.savez(f"./results/{folder}/results.npz", smile_positions=np.stack(smile_positions, axis=0), patterns_smile=np.stack(patterns1, axis=0), cubes_smile = np.stack(cubes1, axis=0),
                            corrected_smile_positions=np.stack(corrected_smile_positions, axis=0), corrected_patterns_smile=np.stack(corrected_patterns1, axis=0), corrected_cubes_smile=np.stack(corrected_cubes1, axis=0),
                            width_values=np.stack(width_values, axis=0), patterns_width=np.stack(patterns2, axis=0), cubes_width = np.stack(cubes2, axis=0),
                            acquisitions=np.stack(acquisitions, axis=0))
