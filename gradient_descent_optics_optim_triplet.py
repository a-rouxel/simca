from simca.cost_functions_optics import evaluate_spectral_dispersion_values
from simca import load_yaml_config
from simca.CassiSystemOptim import CassiSystemOptim
from torch.optim import Adam, LBFGS
from opticalglass.glassfactory import get_glass_catalog
from simca.cost_functions_optics import *
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import datetime
import json
# from torchviz import make_dot

def format_score_details(**kwargs):
    """
    Format the score details into a dictionary.

    Args:
        **kwargs: Keyword arguments of score components and their values.

    Returns:
        dict: A dictionary containing the score formula and component values.
    """
    score_details = {
        'formula': "score = cost_dispersion + cost_distance_glasses + cost_deviation + cost_distorsion + cost_thickness + cost_beam_compression",
        'components': kwargs  # This will include all the passed score components and their values.
    }
    return score_details

target_dispersion = 2200

import math
#matplotlib.use('Agg')
config_system = load_yaml_config("simca/configs/cassi_system_optim_optics_full_triplet_starting_point.yml")

hoya_pd = get_glass_catalog('Schott')
list_of_glasses1 = hoya_pd.df.iloc[:, 0].index.tolist()
# Access the 'nd' values where the first level is NaN
idx = pd.IndexSlice

nd_values = hoya_pd.df.loc[:, idx[pd.NA, 'nd']]
vd_values = hoya_pd.df.loc[:, idx["abbe number", 'vd']]

nd_values = torch.tensor(nd_values.values)
vd_values = torch.tensor(vd_values.values)

cassi_system = CassiSystemOptim(system_config=config_system)

# Assuming 'device' is defined (e.g., 'cuda' or 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cassi_system.to(device)  # Move your model to the specified device

optim_params = [cassi_system.optical_model.lba_c,
                cassi_system.optical_model.alpha_c,
                cassi_system.optical_model.A1,
                cassi_system.optical_model.A2,
                # cassi_system.optical_model.A3,
                cassi_system.optical_model.nd1,
                cassi_system.optical_model.nd2,
                # cassi_system.optical_model.nd3,
                cassi_system.optical_model.vd1,
                cassi_system.optical_model.vd2,
                # cassi_system.optical_model.vd3,
                ]

optimizer = Adam(optim_params, lr=0.005)
iterations = 10000


# Initialize tracking variables
best_loss = float('inf')  # Set the initial best loss to infinity
patience = 500  # How many iterations to wait after last improvement
non_improvement_count = 0  # Counter for iterations without improvement


### create a new dir for the results

if not os.path.exists('results'):
    os.makedirs('results')

# list_scores = []
# numpy_alpha_c = np.linspace(1, 90, 80)
# for alpha_c in numpy_alpha_c:
#     config_system["system architecture"]["dispersive element"]["alpha_c"] = alpha_c
#     cassi_system = CassiSystemOptim(system_config=config_system)
#
#     score = testing_model(cassi_system,nd_values, vd_values)
#     is_nan = torch.isnan(score).item()
#     list_scores.append(score.detach().numpy())
#     print(score)
#
# # idx_min = list_scores.index(min(list_scores))
# min_value = np.nanmin(list_scores)  # Returns the min of the array, ignoring any NaNs
# idx_min = list_scores.index(min_value)
# # print("min score", min(idx_min), min_value)
# config_system["system architecture"]["dispersive element"]["alpha_c"] = numpy_alpha_c[idx_min]
# cassi_system = CassiSystemOptim(system_config=config_system)

# create a new dir for the results

run_id = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
results_dir = os.path.join('results', run_id)
os.makedirs(results_dir, exist_ok=True)


for i in range(iterations):
    # Zero the parameter gradients
    optimizer.zero_grad()

    cassi_system.optical_model.A3 = cassi_system.optical_model.A1
    cassi_system.optical_model.nd3 = cassi_system.optical_model.nd1
    cassi_system.optical_model.vd3 = cassi_system.optical_model.vd1

    alpha_c_out = cassi_system.optical_model.rerun_central_dispersion()
    # Forward pass
    x_vec_out, y_vec_out = cassi_system.propagate_coded_aperture_grid()

    alpha_c = cassi_system.optical_model.alpha_c
    alpha_c_out = cassi_system.optical_model.alpha_c_transmis
    list_apex_angles = [cassi_system.optical_model.A1, cassi_system.optical_model.A2, cassi_system.optical_model.A3]
    list_theta_in = cassi_system.optical_model.list_theta_in
    list_theta_out = cassi_system.optical_model.list_theta_out

    # evaluate spectral dispersion
    dispersion, central_coordinates_X = evaluate_spectral_dispersion_values(x_vec_out, y_vec_out)

    y_second_derivative = calculate_second_derivative(cassi_system.wavelengths, central_coordinates_X)

    # evaluate direct_view
    deviation = evaluate_direct_view(alpha_c, alpha_c_out, list_apex_angles)
    # print("deviation in degrees", deviation * 180 / np.pi)

    # evaluate beam compression
    beam_compression = evaluate_beam_compression(list_theta_in, list_theta_out)
    # print("beam compression", beam_compression)

    # evaluate thickness
    thickness = evaluate_thickness(list_apex_angles)
    # print("thickness", thickness)

    # evaluate distortions
    x_vec_no_dist, y_vec_no_dist = get_cassi_system_no_spectial_distorsions(cassi_system.X_coded_aper_coordinates,
                                                                            cassi_system.Y_coded_aper_coordinates,
                                                                            central_coordinates_X)

    # plt.scatter(cassi_system.X_coordinates_propagated_coded_aperture.detach().numpy()[..., 0],
    #             cassi_system.Y_coordinates_propagated_coded_aperture.detach().numpy()[..., 0], color='blue', label="distor")
    # plt.scatter(cassi_system.X_coordinates_propagated_coded_aperture.detach().numpy()[..., -1],
    #             cassi_system.Y_coordinates_propagated_coded_aperture.detach().numpy()[..., -1], color='red', label="distor")
    # plt.scatter(x_vec_no_dist.detach().numpy()[..., 0], y_vec_no_dist.detach().numpy()[..., 0], label="no dist",
    #             color='green')
    # plt.scatter(x_vec_no_dist.detach().numpy()[..., -1], y_vec_no_dist.detach().numpy()[..., -1], label="no dist",
    #             color='green')
    # plt.show()
    distortion_metric = evaluate_distortions(x_vec_out, y_vec_out, x_vec_no_dist, y_vec_no_dist)

    # print("distortion metric", distortion_metric)

    # distance from closest point
    current_value1 = [cassi_system.optical_model.nd1, cassi_system.optical_model.vd1]
    current_value2 = [cassi_system.optical_model.nd2, cassi_system.optical_model.vd2]
    current_value3 = [cassi_system.optical_model.nd3, cassi_system.optical_model.vd3]

    distance_closest_point1, min_idx1 = evaluate_distance(current_value1, nd_values, vd_values)
    distance_closest_point2, min_idx2 = evaluate_distance(current_value2, nd_values, vd_values)
    distance_closest_point3, min_idx3 = evaluate_distance(current_value3, nd_values, vd_values)

    # # cost functions
    cost_dispersion = (target_dispersion/5- dispersion/5)**2
    cost_distance_glasses = (10**10*(distance_closest_point1**2 + distance_closest_point2**2 + distance_closest_point3**2))**2
    cost_deviation = (deviation*10)**2
    cost_distorsion = (distortion_metric/1000)**2
    cost_thickness = thickness*10
    cost_beam_compression = torch.abs(beam_compression - 1)*10
    cost_non_linearity = 0.001* y_second_derivative**2
    # cost_distance_total_intern_reflection = 0.1*softplus(cassi_system.optical_model.min_distance_from_total_intern_reflection)**2
    cost_block = ((-1*list_apex_angles[0] + list_apex_angles[1] - list_apex_angles[2])*100)**2


    score =  cost_dispersion + cost_distance_glasses + cost_deviation  + cost_thickness + cost_block
    # score =  (alpha_c_out - 10)**2
    # Assume `score` is your loss metric that you want to minimize

    current_loss = score.item()

    is_nan = math.isnan(current_loss)
    if is_nan:
        print("nan value")
        break


    latest_optical_params = {"alpha_c":cassi_system.optical_model.alpha_c.clone(),
                             "lba_c":cassi_system.optical_model.lba_c.clone(),
                            "A1":cassi_system.optical_model.A1.clone(),
                            "A2":cassi_system.optical_model.A2.clone(),
                            "A3":cassi_system.optical_model.A3.clone(),
                            "nd1":cassi_system.optical_model.nd1.clone(),
                            "nd2":cassi_system.optical_model.nd2.clone(),
                            "nd3":cassi_system.optical_model.nd3.clone(),
                            "vd1":cassi_system.optical_model.vd1.clone(),
                            "vd2":cassi_system.optical_model.vd2.clone(),
                            "vd3":cassi_system.optical_model.vd3.clone(),
    }

    # Backward pass
    score.backward()
    optimizer.step()



    if i %100 == 0:
        print(f'Iteration {i}, Loss: {current_loss}')
        print("dispersion in microns", dispersion)
        print("cost_dispersion", cost_dispersion)
        print("cost_distance_glasses", cost_distance_glasses)
        print("cost_deviation", cost_deviation)
        print("cost_distorsion", cost_distorsion)
        print("cost_thickness", cost_thickness)
        # print("cost_beam_compression", cost_beam_compression)
        print("cost_block", cost_block)
        print(cost_block)
        # print("cost_block", cost_block)
        # print("cost_distance_total_intern_reflection", cost_distance_total_intern_reflection)

        # Saving optimization details at the end of optimization
        optimization_details = {
            'reason_for_stopping': 'no improvement' if non_improvement_count >= patience else 'completed',
            'iterations': i + 1,
            'end_parameters': {param_name: param.item() for param_name, param in latest_optical_params.items()},
            'current_loss': current_loss,
            # Add other details as needed
        }

        details_path = os.path.join(results_dir, 'optimization_details.json')
        with open(details_path, 'a') as f:
            json.dump(optimization_details, f, indent=4)

        score_details = format_score_details(
            cost_dispersion=cost_dispersion.item(),
            cost_distance_glasses=cost_distance_glasses.item(),
            cost_deviation=cost_deviation.item(),
            cost_distorsion=cost_distorsion.item(),
            cost_thickness=cost_thickness.item(),
            cost_beam_compression=cost_beam_compression.item(),
        )

        # Save the score details to a file
        details_path = os.path.join(results_dir, f'score_details_iteration.json')
        with open(details_path, 'a') as f:
            json.dump(score_details, f, indent=4)

    # Check for improvement
    if current_loss < best_loss:
        best_loss = current_loss
        non_improvement_count = 0  # Reset counter
    else:
        non_improvement_count += 1

    # Check if it's time to stop
    if non_improvement_count >= patience:
        print(f'Stopping early at iteration {i} due to no improvement.')
        break  # Exit the loop

    # input("Press Enter to continue...")



# Testing with closest points

current_glass_values_1 = [latest_optical_params["nd1"], latest_optical_params["vd1"]]
current_glass_values_2 = [latest_optical_params["nd2"], latest_optical_params["vd2"]]
current_glass_values_3 = [latest_optical_params["nd3"], latest_optical_params["vd3"]]


distance_closest_point1, min_idx_1 = evaluate_distance(current_glass_values_1, nd_values, vd_values)
distance_closest_point2, min_idx_2 = evaluate_distance(current_glass_values_2, nd_values, vd_values)
distance_closest_point3, min_idx_3 = evaluate_distance(current_glass_values_3, nd_values, vd_values)

plt.scatter(nd_values, vd_values, c='b', label='all glasses')
plt.scatter(current_glass_values_1[0].detach().numpy(), current_glass_values_1[1].detach().numpy(), c='r', label='current glass 1')
plt.scatter(current_glass_values_2[0].detach().numpy(), current_glass_values_2[1].detach().numpy(), c='g', label='current glass 2')
plt.scatter(current_glass_values_3[0].detach().numpy(), current_glass_values_3[1].detach().numpy(), c='y', label='current glass 3')
plt.scatter(nd_values[min_idx_1], vd_values[min_idx_1], c='r', label='closest glass 1',marker='x')
plt.scatter(nd_values[min_idx_2], vd_values[min_idx_2], c='g', label='closest glass 2',marker='x')
plt.scatter(nd_values[min_idx_3], vd_values[min_idx_3], c='y', label='closest glass 3',marker='x')
plot_path = os.path.join(results_dir, 'glasses_plot.png')
plt.savefig(plot_path)
# plt.close()
plt.show()

glass_1 = list_of_glasses1[min_idx_1]
glass_2 = list_of_glasses1[min_idx_2]
glass_3 = list_of_glasses1[min_idx_3]

# print(glass_1, glass_2, glass_3)



print(latest_optical_params)

cassi_system_renewal = CassiSystemOptim(system_config=cassi_system.system_config)


print(cassi_system_renewal.system_config)
# set the closest glasses
cassi_system_renewal.system_config["system architecture"]["dispersive element"]["wavelength center"] = latest_optical_params["lba_c"]
cassi_system_renewal.system_config["system architecture"]["dispersive element"]["alpha_c"] = latest_optical_params["alpha_c"] * 180 / np.pi
cassi_system_renewal.system_config["system architecture"]["dispersive element"]["A1"] = latest_optical_params["A1"] * 180 / np.pi
cassi_system_renewal.system_config["system architecture"]["dispersive element"]["A2"] = latest_optical_params["A2"]* 180 / np.pi
cassi_system_renewal.system_config["system architecture"]["dispersive element"]["A3"] = latest_optical_params["A3"] * 180 / np.pi
cassi_system_renewal.system_config["system architecture"]["dispersive element"]['continuous glass materials 1'] = False
cassi_system_renewal.system_config["system architecture"]["dispersive element"]['continuous glass materials 2']= False
cassi_system_renewal.system_config["system architecture"]["dispersive element"]['continuous glass materials 3'] = False
cassi_system_renewal.system_config["system architecture"]["dispersive element"]['glass1'] = glass_1
cassi_system_renewal.system_config["system architecture"]["dispersive element"]['glass2'] = glass_2
cassi_system_renewal.system_config["system architecture"]["dispersive element"]['glass3'] = glass_3

cassi_system_renewal.update_optical_model(cassi_system.system_config)

# pattern : Generate the coded aperture pattern
# cassi_system_renewal.generate_2D_pattern(config_patterns)

X_coordinates_propagated_coded_aperture, Y_coordinates_propagated_coded_aperture = cassi_system_renewal.propagate_coded_aperture_grid()

# evaluate spectral dispersion
dispersion, central_coordinates_X = evaluate_spectral_dispersion_values(X_coordinates_propagated_coded_aperture, Y_coordinates_propagated_coded_aperture)

x_vec_no_dist, y_vec_no_dist = get_cassi_system_no_spectial_distorsions(cassi_system_renewal.X_coded_aper_coordinates,
                                                                        cassi_system_renewal.Y_coded_aper_coordinates,
                                                                        central_coordinates_X)
print(cassi_system_renewal.system_config)
print("dispersion in microns", dispersion)

plt.scatter(X_coordinates_propagated_coded_aperture.detach().numpy()[..., 0], Y_coordinates_propagated_coded_aperture.detach().numpy()[..., 0], color='blue', label="distor")
plt.scatter(X_coordinates_propagated_coded_aperture.detach().numpy()[..., -1], Y_coordinates_propagated_coded_aperture.detach().numpy()[..., -1], color='red', label="distor")
#
plt.scatter(x_vec_no_dist.detach().numpy()[..., 0], y_vec_no_dist.detach().numpy()[..., 0], label="no dist",
            color='green')
plt.scatter(x_vec_no_dist.detach().numpy()[..., -1], y_vec_no_dist.detach().numpy()[..., -1], label="no dist",
            color='green')

plot_path = os.path.join(results_dir, 'grids.png')
plt.savefig(plot_path)
plt.show()




