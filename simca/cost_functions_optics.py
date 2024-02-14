import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def calculate_second_derivative(x, y):
    """
    Calculate the second derivative of y with respect to x using central differences.

    Parameters:
    - x: 1D numpy array of independent variable values.
    - y: 1D numpy array of dependent variable values (e.g., spectral_dispersion).

    Returns:
    - Second derivative of y with respect to x.
    """
    # Ensure x and y are numpy arrays for element-wise operations
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate spacings between points
    h = np.diff(x)

    # Ensure uniform spacing
    if not np.allclose(h, h[0]):
        raise ValueError("x values must be uniformly spaced to use this method.")

    h = h[0]  # Spacing

    # Calculate second derivative using central differences
    y_second_derivative = (y[:-2] - 2 * y[1:-1] + y[2:]) / h**2

    return y_second_derivative

def get_cassi_system_no_spectial_distorsions(X, Y, central_coordinates_X):

    central_coordinates_X = [float(coord) for coord in list(central_coordinates_X)]


    X_vec_out = np.zeros((X.shape[0], X.shape[1], len(central_coordinates_X)))
    Y_vec_out = np.zeros((Y.shape[0], Y.shape[1], len(central_coordinates_X)))

    for i in range(len(central_coordinates_X)):
        X_vec_out[:,:,i] = -1*X + central_coordinates_X[i]
        Y_vec_out[:,:,i] = -1*Y

    return X_vec_out, Y_vec_out

def evaluate_spectral_dispersion_values(x_vec_out, y_vec_out):

    # calculate spectral dispersion in um

    central_coordinates_X = x_vec_out[x_vec_out.shape[0]//2, x_vec_out.shape[1]//2,:]
    spectral_dispersion = np.abs(central_coordinates_X[0] - central_coordinates_X[-1])
    return float(spectral_dispersion), central_coordinates_X


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


