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
    aspect = 0.1
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
    sigma = 0.75

    cassi_system.X_coordinates_propagated_coded_aperture = cassi_system.X_coordinates_propagated_coded_aperture.to(device)
    cassi_system.Y_coordinates_propagated_coded_aperture = cassi_system.Y_coordinates_propagated_coded_aperture.to(device)
    cassi_system.X_detector_coordinates_grid = cassi_system.X_detector_coordinates_grid.to(device)
    cassi_system.Y_detector_coordinates_grid = cassi_system.Y_detector_coordinates_grid.to(device)

    num_iterations = 1000  # Define num_iterations as needed

    first_pos = 0.5
    last_pos = 0.8
    step_pos = 0.2/1

    image_counter = 0

    patterns1 = []
    patterns2 = []
    acquisitions = []

    prev_position = None

    #for position in np.linspace(0.4, 0.6, np.round(cassi_system.system_config["coded aperture"]["number of pixels along X"]*0.2).astype('int')):
    for position in np.arange(first_pos, last_pos, step_pos):
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
        sigma = 0.75

        cassi_system.X_coordinates_propagated_coded_aperture = cassi_system.X_coordinates_propagated_coded_aperture.to(device)
        cassi_system.Y_coordinates_propagated_coded_aperture = cassi_system.Y_coordinates_propagated_coded_aperture.to(device)
        cassi_system.X_detector_coordinates_grid = cassi_system.X_detector_coordinates_grid.to(device)
        cassi_system.Y_detector_coordinates_grid = cassi_system.Y_detector_coordinates_grid.to(device)
        
        # Adjust the learning rate
        if algo == "LBFGS":
            lr = 0.002 # default: 0.05
        elif algo == "ADAM":
            lr = 0.005 # default: 0.005

        if position == 0.5:
            pos_slit_detector = 0.41
        elif position == 0.7:
            pos_slit_detector = 0.124
        cassi_system = optim_smile(cassi_system, position, pos_slit_detector, sigma, device, algo, lr, num_iterations, max_iter_cnt, prev_position = prev_position, plot_frequency=200)

        prev_position = (cassi_system.array_x_positions.detach()-position)

        pattern = cassi_system.pattern.detach().numpy()

        patterns1.append(pattern)

        start_position = cassi_system.array_x_positions.detach()

        cassi_system = optim_width(cassi_system, start_position, 0.41, cassi_system.system_config["detector"]["number of pixels along Y"], sigma, device, algo, lr, num_iterations, max_iter_cnt, plot_frequency = 200)

        pattern = cassi_system.pattern.detach().numpy()
        acquisition = cassi_system.measurement.detach().numpy()

        patterns2.append(pattern)
        acquisitions.append(acquisition)

        
            
    #print(torch.std(cassi_system.measurement.detach()))

    print(f"Exec time: {time.time() - time_start}s")

    fig1 = plt.figure()
    im1 = plt.imshow(patterns1[0], animated = True, aspect=aspect)
    plt.colorbar()

    fig2 = plt.figure()
    im2 = plt.imshow(patterns2[0], animated = True, aspect=aspect)
    plt.colorbar()

    fig3 = plt.figure()
    im3 = plt.imshow(acquisitions[0], animated = True, aspect=aspect)
    plt.colorbar()
    def update1(i):
        im1.set_array(patterns1[i])
        return im1,
    def update2(i):
        im2.set_array(patterns2[i])
        return im2,
    def update3(i):
        im3.set_array(acquisitions[i])
        return im3,

    animation_fig1 = anim.FuncAnimation(fig1, update1, frames=len(patterns1), interval = 1000, repeat=True)
    animation_fig2 = anim.FuncAnimation(fig2, update2, frames=len(patterns2), interval = 1000, repeat=True)
    animation_fig3 = anim.FuncAnimation(fig3, update2, frames=len(acquisitions), interval = 1000, repeat=True)
    
    plt.show()
    animation_fig1.save("patterns_smile.gif")
    animation_fig2.save("patterns_width.gif")
    animation_fig3.save("acquisitions.gif")
