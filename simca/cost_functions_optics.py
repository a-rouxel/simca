import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
    cost_distance_glasses = (10**4*(distance_closest_point1**2 + distance_closest_point2**2 + distance_closest_point3**2))**2
    cost_deviation = deviation*10
    cost_distorsion = distortion_metric**2
    cost_thickness = thickness*10
    cost_beam_compression = torch.abs(beam_compression - 1)*10
    cost_non_linearity = 0.001* y_second_derivative**2
    # cost_distance_total_intern_reflection = 0.1*softplus(cassi_system.optical_model.min_distance_from_total_intern_reflection)**2


    score =  cost_distance_glasses + cost_deviation + cost_distorsion + cost_thickness + cost_beam_compression + cost_non_linearity

    return score
def calculate_second_derivative(x, y):
    """
    Calculate the second derivative of y with respect to x using central differences.

    Parameters:
    - x: 1D numpy array of independent variable values.
    - y: 1D numpy array of dependent variable values (e.g., spectral_dispersion).

    Returns:
    - Second derivative of y with respect to x.
    """
    # # Ensure x and y are numpy arrays for element-wise operations
    # x = np.asarray(x)
    # y = np.asarray(y)

    # Calculate spacings between points
    h = torch.diff(x)


    h = h[0]  # Spacing

    # Calculate second derivative using central differences
    y_second_derivative = (y[:-2] - 2 * y[1:-1] + y[2:]) / h**2

    sum_y = torch.sum(y)
    return sum_y

def get_cassi_system_no_spectial_distorsions(X, Y, central_coordinates_X):

    central_coordinates_X = [coord for coord in list(central_coordinates_X)]

    X_vec_out = torch.zeros((1,X.shape[0],X.shape[1],len(central_coordinates_X)))
    Y_vec_out = torch.zeros((1,X.shape[0],X.shape[1],len(central_coordinates_X)))


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


def evaluate_distortions(X_vec_out_distor, Y_vec_out_distors, X_vec_out, Y_vec_out):

    distance_map = torch.sqrt((X_vec_out_distor - X_vec_out)**2 + (Y_vec_out_distors - Y_vec_out)**2)

    # mean distance
    distortion_metric = torch.max(torch.max(torch.max(distance_map)))
    # distortion_metric = np.sum(np.sum(distance_map))
    # distortion_metric = torch.sum(torch.sum(torch.sum(distance_map)))
    return distortion_metric

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


