import math
import numpy as np
from utils.helpers import *

def find_cam_position_curve_fit_new_model_no_A_oneinput(X_cam,Y_cam,n, A,F, alpha_c,delta_alpha_c, delta_beta_c):
    """
    Mapping function between the DMD and the scene (ray tracing type function)


    Parameters
    ----------
    x_dmd : float -- in um
        X position, on the DMD, of the pixel for a given lambda
    y_dmd : float -- in um
        Y position, on the DMD, of the pixel for a given lambda
    n : float
        refractive index of the prism of a given lambda.
    F : float -- in um
        focal length of the lens L3.
    A : float -- in rad
        apex angle of the BK7 prism.
    alpha_c : float -- in rad
        minimum angle of deviation for the central wavelength.

    Returns
    -------
    x_scene : float -- in um
        X position, on the scene, of the pixel for a given lambda.
    y_scene : float -- in um
        Y position, on the scene, of the pixel for a given lambda.

    """

    print("X_cam, Y_cam",X_cam[0],Y_cam[0])
    print("n", n[0])
    print(A,F, alpha_c,delta_alpha_c, delta_beta_c)
    angle_with_P1 = alpha_c - A / 2 + delta_alpha_c
    angle_with_P2 = alpha_c - A / 2 - delta_alpha_c

    k = l_p_a_V2(X_cam, Y_cam, F)
    # Rotation in relation to P1 around the Y axis

    k_1 = rotation_y(angle_with_P1) @ k[:,0,:]
    # Rotation in relation to P1 around the X axis
    k_2 = rotation_x(delta_beta_c) @ k_1
    # Rotation of P1 in relation to frame_in along the new Y axis
    k_3 = rotation_y(A / 2) @ k_2

    norm_k = np.sqrt(k_3[0] ** 2 + k_3[1] ** 2 + k_3[2] ** 2)
    k_3 /= norm_k

    # Output vector of the prism in frame_out
    k_out_p = prism_in_to_prism_out_parallel_dmd_scene(k_3, n, A)

    k_out_p = k_out_p * norm_k

    # Rotation of P2 in relation to frame_in along the new Y axis
    k_3_bis = np.dot(rotation_y(A / 2), k_out_p)
    # Rotation in relation to P2 around the X axis
    k_2_bis = np.dot(rotation_x(-delta_beta_c), k_3_bis)
    # Rotation in relation to P2 around the Y axis
    k_1_bis = np.dot(rotation_y(angle_with_P2), k_2_bis)

    theta_out = np.arctan(k_1_bis[0] / k_1_bis[2])
    phi_out = np.arctan(k_1_bis[1] / k_1_bis[2])

    [X_dmd, Y_dmd] = l_a_p_V2(theta_out,  phi_out, F)


    return X_dmd, Y_dmd

def l_a_p_V2(alpha, beta, F):

    x = F*np.tan(alpha)
    y  = F*np.tan(beta)

    return [x,y]

def prism_in_to_prism_out_parallel_dmd_scene(k0, n, A):
    """
    Ray tracing through the prism

    Parameters
    ----------
    k0 : numpy array -- no units
        input vector
    n : float -- no units
        refractive index of the considered wavelength
    A : float -- rad
        Apex angle of the prism
    Returns
    -------
    kout : numpy array -- no units
        output vector

    """

    kp = np.array([k0[0], k0[1], np.sqrt(n ** 2 - k0[0] ** 2 - k0[1] ** 2)])

    kp_r = np.matmul(rotation_y(-A), kp)

    kout = [kp_r[0], kp_r[1], np.sqrt(1 - kp_r[0] ** 2 - kp_r[1] ** 2)]


    return kout


def l_p_a_V2(x_obj, y_obj, F):

    alpha = -1*np.arctan(x_obj / F)
    beta  = -1*np.arctan(y_obj / F)

    k0 = np.array([[np.sin(alpha) * np.cos(beta)], [np.sin(beta)*np.cos(alpha)],
                   [np.cos(alpha) * np.cos(beta)]])

    return k0



def compute_refractive_index(x_cam, y_cam, x_dmd, y_dmd, F, A, alpha_c):


    """
    Inversion of the mapping function. Solving it in n(lambda).

    Parameters
    ----------
    x_cam : float
        X position of the pixel on the camera   -- in um
    y_cam : float
        Y position of the pixel on the camera   -- in um.
    x_dmd : float
        X position of the pixel on the dmd      -- in um
    y_dmd : float
        Y position of the pixel on the dmd      -- in um.
    polynomial_coef : np.array
        polynomial coefficients of the quadratic approximation of the y dependencies -- no units
    F : float -- in um
        focal length of the lens L3.
    A : float -- in rad
        apex angle of the BK7 prism.
    alpha_c : float -- in rad
        minimum angle of deviation for the central wavelength.

    Returns
    -------
    n : float
        refractive index of the prism that corresponds to the (x_cam,x_dmd) pair -- no units.
    """
    (k_in, alpha_in, beta_in) = scene_to_prism(x_cam, y_cam, F, alpha_c)
    (k_out, alpha_out, beta_out) = DMD_to_prism(x_dmd, y_dmd, F, alpha_c)

    n = math.sqrt(((math.sin(alpha_in) * math.cos(beta_in) * math.cos(A) - k_out[0]) / math.sin(A)) ** 2 + math.sin(
        alpha_in) ** 2 * math.cos(beta_in) ** 2 + math.sin(beta_in) ** 2)

    return n