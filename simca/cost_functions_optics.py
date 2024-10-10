import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import math
import json
from torch.optim import Adam
from opticalglass.glassfactory import get_glass_catalog

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from simca.helper import *
import matplotlib

from simca import load_yaml_config
from simca.CassiSystem import CassiSystem
from opticalglass.glassfactory import get_glass_catalog


from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pprint import pprint

def get_catalog_glass_infos(catalog, device="cpu"):
    glass_pd = get_glass_catalog(catalog)
    list_of_glasses = glass_pd.df.iloc[:, 0].index.tolist()

    tuple_glass_data = glass_pd.glass_map_data("d")

    nd = tuple_glass_data[0][:]
    vd = tuple_glass_data[1][:]
    nd_values = torch.tensor(nd, device=device)
    vd_values = torch.tensor(vd, device=device)

    return list_of_glasses, nd_values, vd_values


def plot_grids_coordinates(cassi_system,test_name='test',save_fig_dir=None,save_fig=False):

    X_coordinates_propagated_coded_aperture, Y_coordinates_propagated_coded_aperture = cassi_system.propagate_coded_aperture_grid()


    dispersion, central_coordinates_X = evaluate_spectral_dispersion_values(X_coordinates_propagated_coded_aperture.unsqueeze(0), Y_coordinates_propagated_coded_aperture.unsqueeze(0))
    x_vec_no_dist, y_vec_no_dist = get_cassi_system_no_spectial_distorsions(
        cassi_system.X_coded_aper_coordinates,
        cassi_system.Y_coded_aper_coordinates,
        central_coordinates_X
    )


    plt.figure()

    # plot distorsion grids
    plt.scatter(x_vec_no_dist.detach().cpu().numpy()[..., 0], y_vec_no_dist.detach().cpu().numpy()[..., 0], label="no dist", color='grey',s=70)
    plt.scatter(x_vec_no_dist.detach().cpu().numpy()[..., 3], y_vec_no_dist.detach().cpu().numpy()[..., 3], label="no dist", color='grey',s=70)
    plt.scatter(x_vec_no_dist.detach().cpu().numpy()[..., -1], y_vec_no_dist.detach().cpu().numpy()[..., -1], label="no dist", color='grey',s=70)
    plt.scatter(X_coordinates_propagated_coded_aperture.detach().cpu().numpy()[..., 0], Y_coordinates_propagated_coded_aperture.detach().cpu().numpy()[..., 0], color='blue', label="distor")
    plt.scatter(X_coordinates_propagated_coded_aperture.detach().cpu().numpy()[..., 3], Y_coordinates_propagated_coded_aperture.detach().cpu().numpy()[..., 3], color='green', label="distor")
    plt.scatter(X_coordinates_propagated_coded_aperture.detach().cpu().numpy()[..., -1], Y_coordinates_propagated_coded_aperture.detach().cpu().numpy()[..., -1], color='red', label="distor")

    if save_fig:
        plt.savefig(save_fig_dir +f"grids_coordinates_{test_name}.svg")
    
    plt.show()

    if save_fig:
        np.save(save_fig_dir + f"/x_coordinates_{test_name}",X_coordinates_propagated_coded_aperture.detach().cpu().numpy())
        np.save(save_fig_dir + f"/y_coordinates_{test_name}",Y_coordinates_propagated_coded_aperture.detach().cpu().numpy())
        np.save(save_fig_dir + f"/wavelengths_{test_name}",cassi_system.wavelengths.detach().cpu().numpy())
        



def format_score_details(cost_details,weighted_cost_components,cost_weights):
    score_details = {
        'components non weights': {key: value for key, value in cost_weights.items()},
        'components with weights': {key : float(value) for key, value in cost_details.items()},
        'weighted components': {key : float(value) for key, value in weighted_cost_components.items()},
    }
    return score_details
    


def initialize_optimizer(cassi_system, params_to_optimize, num_iterations):
    base_params = []
    glass_params = []
    lba_c_param = []
    
    for name, param in cassi_system.optical_model.named_parameters():
        if name in params_to_optimize:
            if name in ['nd1', 'vd1', 'nd2', "vd2", "nd3", "vd3"]:
                glass_params.append(param)
            elif name =='lba_c':
                lba_c_param.append(param)
            else:
                base_params.append(param)

    # Define parameter groups with different initial learning rates
    param_groups = [
        {'params': base_params, 'lr': 0.01},  # Base initial learning rate
        {'params': glass_params, 'lr': 0.01},  # Higher initial learning rate for glass parameters
        {'params': lba_c_param, 'lr': 0.01}  # Higher initial learning rate for lba_c parameter
    ]

    optimizer = Adam(param_groups)
    
    # Create a cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=1e-6)

    return optimizer, scheduler


def setup_results_directory(output_dir, step_name):
    # Create the optimization_results directory within the output_dir
    optimization_results_dir = os.path.join(output_dir, "optimization_results")
    os.makedirs(optimization_results_dir, exist_ok=True)
    
    # Create a subdirectory for the current step
    step_dir = os.path.join(optimization_results_dir, step_name)
    os.makedirs(step_dir, exist_ok=True)
    
    return step_dir

def evaluate_optical_performances(cassi_system,save_fig_dir=None):

    
    cassi_system.optical_model.rerun_central_dispersion()
    x_vec_out, y_vec_out = cassi_system.propagate_coded_aperture_grid()

    alpha_c = cassi_system.optical_model.alpha_c
    alpha_c_out = cassi_system.optical_model.alpha_c_transmis
    list_apex_angles = [cassi_system.optical_model.A1, cassi_system.optical_model.A2, cassi_system.optical_model.A3]
    list_theta_in = cassi_system.optical_model.list_theta_in
    list_theta_out = cassi_system.optical_model.list_theta_out

    dispersion, central_coordinates_X = evaluate_spectral_dispersion_values(x_vec_out, y_vec_out)
    y_second_derivative = calculate_second_derivative(cassi_system.wavelengths, central_coordinates_X)
    deviation = evaluate_direct_view(alpha_c, alpha_c_out, list_apex_angles)
    beam_compression = evaluate_beam_compression(list_theta_in, list_theta_out)
    thickness = evaluate_thickness(list_apex_angles)
    x_vec_no_dist, y_vec_no_dist = get_cassi_system_no_spectial_distorsions(
        cassi_system.X_coded_aper_coordinates,
        cassi_system.Y_coded_aper_coordinates,
        central_coordinates_X
    )
    distortion_metric, _,mean_distortion = evaluate_distortions(x_vec_out, y_vec_out, x_vec_no_dist, y_vec_no_dist,save_fig_dir)

    performances = {
        'dispersion [um]': int(dispersion),
        'deviation [deg]': float(deviation)*180/np.pi,
        'max distortion [um]': float(distortion_metric),
        'mean distortion [um]': float(mean_distortion),
        'beam_compression [no units]': float(beam_compression),}


    return performances


def evaluate_cost_functions(cassi_system, cost_weights, target_dispersion, nd_values, vd_values, iteration_number, list_of_glasses):
    cassi_system.optical_model.rerun_central_dispersion()
    x_vec_out, y_vec_out = cassi_system.propagate_coded_aperture_grid()


    alpha_c = cassi_system.optical_model.alpha_c
    alpha_c_out = cassi_system.optical_model.alpha_c_transmis
    list_apex_angles = [cassi_system.optical_model.A1, cassi_system.optical_model.A2, cassi_system.optical_model.A3]
    list_theta_in = cassi_system.optical_model.list_theta_in
    list_theta_out = cassi_system.optical_model.list_theta_out

    dispersion, central_coordinates_X = evaluate_spectral_dispersion_values(x_vec_out, y_vec_out)
    y_second_derivative = calculate_second_derivative(cassi_system.wavelengths, central_coordinates_X)
    deviation = evaluate_direct_view(alpha_c, alpha_c_out, list_apex_angles)
    beam_compression = evaluate_beam_compression(list_theta_in, list_theta_out)
    thickness = evaluate_thickness(list_apex_angles)
    x_vec_no_dist, y_vec_no_dist = get_cassi_system_no_spectial_distorsions(
        cassi_system.X_coded_aper_coordinates,
        cassi_system.Y_coded_aper_coordinates,
        central_coordinates_X
    )
    distortion_metric, _,mean_distortion = evaluate_distortions(x_vec_out, y_vec_out, x_vec_no_dist, y_vec_no_dist)
    if cassi_system.optical_model.index_estimation_method == "cauchy":
        distance_glasses_metric = evaluate_distance_glasses(cassi_system, nd_values, vd_values, list_of_glasses)
    else : 
        distance_glasses_metric = 0

    

    cost_components = {
        'cost_dispersion': (target_dispersion  - dispersion) ** 2,
        'cost_distance_glasses': (10 ** 10 * distance_glasses_metric) * iteration_number,
        'cost_deviation': (deviation * 5000) ** 2,
        'cost_distorsion': distortion_metric ** 2,
        'cost_thickness': thickness * 5000,
        'cost_beam_compression': torch.abs(beam_compression - 1) * 10,
        'cost_non_linearity': 0.001 * y_second_derivative ** 2,
        'cost_distance_total_intern_reflection': 2 * softplus(cassi_system.optical_model.min_distance_from_total_intern_reflection) ** 2
    }
    interesting_values = {
        'dispersion [um]': dispersion,
        'deviation [deg]': deviation,
        'max distortion [um]': distortion_metric,
        'mean distortion [um]': mean_distortion,
        'beam_compression [no units]': beam_compression,}
    
    weighted_cost_components = {key: cost_weights[key] * value for key, value in cost_components.items()}   
    
    
    score = sum(cost_weights.get(key, 0) * value for key, value in cost_components.items())


    return score, cost_components,weighted_cost_components,interesting_values


def save_optimization_details(results_dir, non_improvement_count, patience, i, latest_optical_params, current_loss,interesting_values):
    optimization_details = {
        'reason_for_stopping': 'no improvement' if non_improvement_count >= patience else 'completed',
        'iterations': i,
        'end_parameters': {
            param_name: (param.item() if 'A' not in param_name and 'alpha_c' not in param_name and 'delta' not in param_name else math.degrees(param.item()))
            for param_name, param in latest_optical_params.items()
        },
        'optical system values': {param_name: (param.item() if 'deviation' not in param_name else math.degrees(param.item())) for param_name, param in interesting_values.items()},
        'current_loss': current_loss,
    }

    details_path = os.path.join(results_dir, 'optimization_details.json')
    with open(details_path, 'a') as f:
        json.dump(optimization_details, f, indent=4)


def plot_glass_selection(nd_values, vd_values, current_glass_values_1, current_glass_values_2, min_idx_1, min_idx_2):
    plt.scatter(nd_values.detach().cpu().numpy(), vd_values.detach().cpu().numpy(), c='b', label='all glasses')
    plt.scatter(current_glass_values_1[0].detach().cpu().numpy(), current_glass_values_1[1].detach().cpu().numpy(), c='r', label='current glass 1')
    plt.scatter(current_glass_values_2[0].detach().cpu().numpy(), current_glass_values_2[1].detach().cpu().numpy(), c='g', label='current glass 2')
    plt.scatter(nd_values[min_idx_1].detach().cpu().numpy(), vd_values[min_idx_1].detach().cpu().numpy(), c='r', label='closest glass 1', marker='x')
    plt.scatter(nd_values[min_idx_2].detach().cpu().numpy(), vd_values[min_idx_2].detach().cpu().numpy(), c='g', label='closest glass 2', marker='x')
    plt.show()


def calculate_second_derivative(x, y):
    """
    Calculate the second derivative of y with respect to x using central differences.

    Parameters:
    - x: 1D torch tensor of independent variable values.
    - y: 1D torch tensor of dependent variable values (e.g., spectral_dispersion).

    Returns:
    - Second derivative of y with respect to x.
    """

    # plt.figure()
    # plt.plot(x.detach().cpu().numpy(),y.detach().cpu().numpy())
    # plt.show()

    # Calculate spacing between points
    h = torch.diff(x)[0]  # Assuming uniform spacing

    # Calculate second derivative using central differences
    y_second_derivative = (y[:-2] - 2 * y[1:-1] + y[2:]) / (h**2)

    # Return the mean of the absolute values of the second derivative

    return torch.mean(torch.abs(y_second_derivative))

def get_cassi_system_no_spectial_distorsions(X, Y, central_coordinates_X):

    central_coordinates_X = [coord for coord in list(central_coordinates_X)]

    X_vec_out = torch.zeros((1,X.shape[0],X.shape[1],len(central_coordinates_X)),device=X.device)
    Y_vec_out = torch.zeros((1,X.shape[0],X.shape[1],len(central_coordinates_X)),device=Y.device)


    for i in range(X_vec_out.shape[-1]):
        X_vec_out[0,:,:,i] = -1*X + central_coordinates_X[i]
        Y_vec_out[0,:,:,i] = -1*Y

    return X_vec_out, Y_vec_out

def evaluate_spectral_dispersion_values(x_vec_out, y_vec_out):

    # calculate spectral dispersion in um
    
    central_coordinates_X = x_vec_out[0,x_vec_out.shape[1]//2, x_vec_out.shape[2]//2,:]
    spectral_dispersion = torch.abs(central_coordinates_X[0] - central_coordinates_X[-1])



    return spectral_dispersion, central_coordinates_X


def evaluate_direct_view(alpha_c,alpha_c_out,list_apex_angles):

    total_dev = alpha_c + alpha_c_out
    for idx,angle in enumerate(list_apex_angles):
            total_dev += ((-1)**(idx+1))*angle
    return total_dev
def softplus(x):
    return torch.log(1 + torch.exp(-x+5))**3
def evaluate_beam_compression(list_theta_in, list_theta_out):

    product = 1
    for theta_in, theta_out in zip(list_theta_in, list_theta_out):
        # print("theta_in", theta_in*180/np.pi)
        # print("theta_out", theta_out*180/np.pi)
        product *= torch.abs(torch.cos(theta_in))/torch.abs(torch.cos(theta_out))

    return product

def evaluate_thickness(list_angles):
    list_angles = torch.tensor([angle**2 for angle in list_angles])
    sum_angles = torch.sum(list_angles)
    return sum_angles


def evaluate_distance_glasses(cassi_system, nd_values, vd_values, list_of_glasses):

    if cassi_system.system_config["system architecture"]["dispersive element"]["type"] == 'prism':
        current_value1 = [cassi_system.optical_model.nd1, cassi_system.optical_model.vd1]
        distance_closest_point1, min_idx_1 = evaluate_distance(current_value1, nd_values, vd_values)
        cost_distance_glass = distance_closest_point1**2

    elif cassi_system.system_config["system architecture"]["dispersive element"]["type"] == 'doubleprism' or cassi_system.system_config["system architecture"]["dispersive element"]["type"] == 'amici':
        current_value1 = [cassi_system.optical_model.nd1, cassi_system.optical_model.vd1]
        current_value2 = [cassi_system.optical_model.nd2, cassi_system.optical_model.vd2]
        distance_closest_point1, min_idx_1 = evaluate_distance(current_value1, nd_values, vd_values)
        distance_closest_point2, min_idx_2 = evaluate_distance(current_value2, nd_values, vd_values)
        cost_distance_glass = distance_closest_point1**2 + distance_closest_point2**2

    elif cassi_system.system_config["system architecture"]["dispersive element"]["type"] == 'tripleprism':
        current_value1 = [cassi_system.optical_model.nd1, cassi_system.optical_model.vd1]
        current_value2 = [cassi_system.optical_model.nd2, cassi_system.optical_model.vd2]
        current_value3 = [cassi_system.optical_model.nd3, cassi_system.optical_model.vd3]
        distance_closest_point1, min_idx_1 = evaluate_distance(current_value1, nd_values, vd_values)
        distance_closest_point2, min_idx_2 = evaluate_distance(current_value2, nd_values, vd_values)
        distance_closest_point3, min_idx_3 = evaluate_distance(current_value3, nd_values, vd_values)
        cost_distance_glass = distance_closest_point1**2 + distance_closest_point2**2 + distance_closest_point3**2

    return cost_distance_glass

def evaluate_distortions(X_vec_out_distor, Y_vec_out_distors, X_vec_out, Y_vec_out,save_fig_dir=None):

    #print gradient value
    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    distance_map = torch.sqrt((X_vec_out_distor - X_vec_out)**2 + (Y_vec_out_distors - Y_vec_out)**2)
    
    # mean distance
    distortion_metric = torch.max(torch.max(torch.max(distance_map)))
    mean_distortion = torch.mean(torch.mean(torch.mean(distance_map)))
    # distortion_metric = np.sum(np.sum(distance_map))
    # distortion_metric = torch.sum(torch.sum(torch.sum(distance_map)))

    if save_fig_dir is not None:

        for i in [0,5,-1]:

            max_x = X_vec_out[0,0,0,i].detach().cpu().numpy()
            min_x = X_vec_out[0,0,-1,i].detach().cpu().numpy()
            max_y = Y_vec_out[0,0,0,i].detach().cpu().numpy()
            min_y = Y_vec_out[0,-1,0,i].detach().cpu().numpy()
            # print(min_x, max_x, min_y, max_y)

            plt.figure()
            norm = matplotlib.colors.LogNorm(vmin=0.1, vmax=125)
            plt.imshow(distance_map[0,:,:,i].detach().cpu().numpy(),cmap=parula_map,extent=[min_x, max_x, min_y, max_y],norm=norm)
            # plt.imshow(distance_map[0,:,:,0].detach().cpu().numpy(),cmap=parula_map)
            plt.xlabel("x [$\mu$m]",fontsize=18)
            plt.ylabel("y [$\mu$m]",fontsize=18)
            plt.title(f"Distortion map at $\lambda$ = {i}",fontsize=14)
            plt.colorbar()
            plt.savefig(save_fig_dir + f"distor_map_{i}.svg")
            # plt.show()
    return distortion_metric, distance_map,mean_distortion

def evaluate_distance(current_value_pair,nd_values,vd_values):

    etendue_nd = torch.max(nd_values) - torch.min(nd_values)
    etendue_vd = torch.max(vd_values) - torch.min(vd_values)

    distance = ((current_value_pair[0] - nd_values)/etendue_nd)**2 + ((current_value_pair[1] - vd_values)/etendue_vd)**2
    min_distance = torch.min(distance)
    min_distance_idx = torch.argmin(distance)

    return min_distance,min_distance_idx

def evaluate_linearity(values):
    """
    Calculate the R-squared value for a 1D numpy array of values.

    Parameters:
    - values: A 1D numpy array of values.

    Returns:
    - R2 value indicating the fit to a straight line.
    """
    # Reshape data for sklearn
    X = np.arange(len(values)).reshape(-1, 1)  # Independent variable (e.g., time or index)
    y = values.reshape(-1, 1)  # Dependent variable (e.g., observed values)

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict y values
    y_pred = model.predict(X)

    # Calculate R2 score
    r2 = r2_score(y, y_pred)

    return r2, y_pred


def evaluate_approximation_thickness(list_angles):
    sum_angles = np.sum(list_angles)
    return sum_angles

def testing_model(cassi_system,nd_values, vd_values):
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
    distortion_metric,_ = evaluate_distortions(x_vec_out, y_vec_out, x_vec_no_dist, y_vec_no_dist)

    # print("distortion metric", distortion_metric)

    # distance from closest point
    current_value1 = [cassi_system.optical_model.nd1, cassi_system.optical_model.vd1]
    current_value2 = [cassi_system.optical_model.nd2, cassi_system.optical_model.vd2]
    current_value3 = [cassi_system.optical_model.nd3, cassi_system.optical_model.vd3]

    distance_closest_point1, min_idx1 = evaluate_distance(current_value1, nd_values, vd_values)
    distance_closest_point2, min_idx2 = evaluate_distance(current_value2, nd_values, vd_values)
    distance_closest_point3, min_idx3 = evaluate_distance(current_value3, nd_values, vd_values)

    # # cost functions
    cost_distance_glasses = (10**4*(distance_closest_point1**2 + distance_closest_point2**2 + distance_closest_point3**2))**2
    cost_deviation = deviation*10
    cost_distorsion = distortion_metric**2
    cost_thickness = thickness*10
    cost_beam_compression = torch.abs(beam_compression - 1)*10
    cost_non_linearity = 0.001* y_second_derivative**2
    # cost_distance_total_intern_reflection = 0.1*softplus(cassi_system.optical_model.min_distance_from_total_intern_reflection)**2


    score =  cost_distance_glasses + cost_deviation + cost_distorsion + cost_thickness + cost_beam_compression + cost_non_linearity

    return score


def optimize_cassi_system(params_to_optimize, target_dispersion, cost_weights, config_path, iterations, patience, output_dir, step_name, index_estimation_method="cauchy", device='cpu'):
    results_dir = setup_results_directory(output_dir, step_name)
    
    config_system = load_yaml_config(config_path)
    cassi_system = CassiSystem(system_config=config_system, device=device, index_estimation_method=index_estimation_method)
    device = cassi_system.device
    catalog = config_system["system architecture"]["dispersive element"]["catalog"]

    list_of_glasses, nd_values, vd_values = get_catalog_glass_infos(catalog=catalog, device=device)

    optimizer, scheduler = initialize_optimizer(cassi_system, params_to_optimize, 300)
    non_improvement_count = 0
    best_loss = float('inf')

    optimization_details = []

    for i in range(iterations):

        optimizer.zero_grad()



        if cassi_system.system_config['system architecture']['dispersive element']['type'] == 'amici':

            if cassi_system.optical_model.index_estimation_method == "cauchy":
                cassi_system.optical_model.nd3 = cassi_system.optical_model.nd1
                cassi_system.optical_model.vd3 = cassi_system.optical_model.vd1
            if cassi_system.optical_model.index_estimation_method == "sellmeier":
                cassi_system.optical_model.glass3 = cassi_system.optical_model.glass1

        score, cost_details, weighted_cost_components, interesting_values = evaluate_cost_functions(cassi_system, cost_weights, target_dispersion, nd_values, vd_values, i, list_of_glasses)
        current_loss = score.item()

        latest_optical_params = {param_name: param.clone().detach() for param_name, param in cassi_system.optical_model.named_parameters()}

        if cassi_system.system_config['system architecture']['dispersive element']['type'] == 'amici':
            latest_optical_params['A3'] = latest_optical_params['A1']
            if cassi_system.optical_model.index_estimation_method == "cauchy":
                latest_optical_params['nd3'] = latest_optical_params['nd1']
                latest_optical_params['vd3'] = latest_optical_params['vd1']


        if i % 10 == 0:
            print(f'Iteration {i}, Loss: {current_loss}')


            details = {
                'reason_for_stopping': 'no improvement' if non_improvement_count >= patience else 'completed',
                'iterations': i,
                'end_parameters': {
                    param_name: (param.item() if 'A' not in param_name and 'alpha_c' not in param_name and 'delta' not in param_name else math.degrees(param.item()))
                    for param_name, param in latest_optical_params.items()
                },
                'optical system values': {param_name: (param.item() if 'deviation' not in param_name else math.degrees(param.item())) for param_name, param in interesting_values.items()},
                'current_loss': current_loss,
            }

            if math.isnan(current_loss):
                print("nan value")
                break

            optimization_details.append(details)

            # Save optical characteristics corresponding to the various iterations
            details_path = os.path.join(results_dir, 'optimization_details.json')
            with open(details_path, 'w') as f:
                json.dump(optimization_details, f, indent=4)

            score_details = format_score_details(cost_details, weighted_cost_components, cost_weights)
            # Save the cost functions score details
            details_path = os.path.join(results_dir, f'score_details_iteration_{i}.json')
            with open(details_path, 'a') as f:
                json.dump(score_details, f, indent=4)



        score.backward()

        # for param in cassi_system.optical_model._parameters:
        #     print(f"gradient {param}: {cassi_system.optical_model.lba_c.grad}")

        optimizer.step()
        scheduler.step()  # Step the scheduler

        if current_loss < best_loss:
            best_loss = current_loss
            non_improvement_count = 0
        else:
            non_improvement_count += 1

        if non_improvement_count >= patience:
            print(f'Stopping early at iteration {i} due to no improvement.')
            break


    # After the optimization loop
    final_config = cassi_system.system_config
    return final_config, latest_optical_params

def test_cassi_system(config_path, index_estimation_method="cauchy", save_fig_dir=None, save_fig=False):

    config_system = load_yaml_config(config_path)
    cassi_system = CassiSystem(system_config=config_system, index_estimation_method=index_estimation_method, device="cpu")

    plot_grids_coordinates(cassi_system, save_fig_dir=save_fig_dir, save_fig=save_fig)
    performances = evaluate_optical_performances(cassi_system, save_fig_dir=save_fig_dir)

    return cassi_system.system_config, performances

import json
import os
import matplotlib.pyplot as plt

def plot_optimization_process(output_dir, step_name):
    step_dir = os.path.join(output_dir, "optimization_results", step_name)
    iterations = []
    weighted_costs = {}
    total_costs = []

    def parse_json_objects(content):
        decoder = json.JSONDecoder()
        pos = 0
        while True:
            try:
                obj, pos = decoder.raw_decode(content, pos)
                yield obj
            except json.JSONDecodeError:
                if pos == len(content):
                    break
                pos += 1

    # Read all score details files
    for filename in os.listdir(step_dir):
        if filename.startswith("score_details_iteration_") and filename.endswith(".json"):
            file_path = os.path.join(step_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Parse multiple JSON objects in the file
                    for data in parse_json_objects(content):
                        iteration = int(filename.split('_')[-1].split('.')[0])
                        iterations.append(iteration)
                        total_cost = 0
                        for cost_name, cost_value in data['weighted components'].items():
                            if cost_name not in weighted_costs:
                                weighted_costs[cost_name] = []
                            weighted_costs[cost_name].append(cost_value)
                            total_cost += cost_value
                        total_costs.append(total_cost)
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue  # Skip this file and continue with the next one

    if not iterations:
        print(f"No valid data found for step {step_name}. Skipping plot creation.")
        return

    # Sort data by iteration
    sorted_indices = sorted(range(len(iterations)), key=lambda k: iterations[k])
    iterations = [iterations[i] for i in sorted_indices]
    for cost_name in weighted_costs:
        weighted_costs[cost_name] = [weighted_costs[cost_name][i] for i in sorted_indices]
    total_costs = [total_costs[i] for i in sorted_indices]

    # Plot individual weighted costs
    plt.figure(figsize=(12, 8))
    for cost_name, cost_values in weighted_costs.items():
        plt.plot(iterations, cost_values, label=cost_name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Weighted Cost')
    plt.title(f'Individual Weighted Costs over Iterations - {step_name}')
    plt.legend()
    plt.yscale('log')  # Use log scale for y-axis
    plt.grid(True)
    
    # Save the individual weighted costs plot
    individual_plot_path = os.path.join(output_dir, f"optimization_plot_individual_{step_name}.png")
    plt.savefig(individual_plot_path)
    plt.close()

    # Plot total cost function
    plt.figure(figsize=(12, 8))
    plt.plot(iterations, total_costs, label='Total Cost')
    
    plt.xlabel('Iteration')
    plt.ylabel('Total Cost')
    plt.title(f'Total Cost Function over Iterations - {step_name}')
    plt.legend()
    plt.yscale('log')  # Use log scale for y-axis
    plt.grid(True)
    
    # Save the total cost plot
    total_plot_path = os.path.join(output_dir, f"optimization_plot_total_{step_name}.png")
    plt.savefig(total_plot_path)
    plt.close()

    print(f"Individual costs plot saved as {individual_plot_path}")
    print(f"Total cost plot saved as {total_plot_path}")