import numpy as np
import math
from cassi_systems.functions_general_purpose import *
def propagate_through_arm_vector(dispersive_element_type,X_mask,Y_mask,n, lba,A,G,m,F, alpha_c,alpha_c_transmis,delta_alpha_c, delta_beta_c):

    if dispersive_element_type == "prism":

        angle_with_P1 = alpha_c - A / 2 + delta_alpha_c
        angle_with_P2 = alpha_c_transmis - A / 2 - delta_alpha_c

        k = model_Lens_pos_to_angle(X_mask, Y_mask, F)
        # Rotation in relation to P1 around the Y axis

        k_1 = rotation_y(angle_with_P1) @ k[:, 0, :]
        # Rotation in relation to P1 around the X axis
        k_2 = rotation_x(delta_beta_c) @ k_1
        # Rotation of P1 in relation to frame_in along the new Y axis
        k_3 = rotation_y(A / 2) @ k_2

        norm_k = np.sqrt(k_3[0] ** 2 + k_3[1] ** 2 + k_3[2] ** 2)
        k_3 /= norm_k

        k_out_p = model_Prism_angle_to_angle(k_3, n, A)
        k_out_p = k_out_p * norm_k

        k_3_bis = np.dot(rotation_y(A / 2), k_out_p)

        # Rotation in relation to P2 around the X axis
        k_2_bis = np.dot(rotation_x(-delta_beta_c), k_3_bis)
        # Rotation in relation to P2 around the Y axis
        k_1_bis = np.dot(rotation_y(angle_with_P2), k_2_bis)

        theta_out = np.arctan(k_1_bis[0] / k_1_bis[2])
        phi_out = np.arctan(k_1_bis[1] / k_1_bis[2])

        [X_dmd, Y_dmd] = model_Lens_angle_to_position(theta_out, phi_out, F)



    elif dispersive_element_type == "grating":

        angle_with_P1 = alpha_c - delta_alpha_c
        angle_with_P2 = alpha_c_transmis + delta_alpha_c

        k = model_Lens_pos_to_angle(X_mask, Y_mask, F)
        # Rotation in relation to P1 around the Y axis

        k_1 = rotation_y(angle_with_P1) @ k[:,0,:]
        # Rotation in relation to P1 around the X axis
        k_2 = rotation_x(delta_beta_c) @ k_1

        k_3 = rotation_y(0) @ k_2
        norm_k = np.sqrt(k_3[0] ** 2 + k_3[1] ** 2 + k_3[2] ** 2)
        k_3 /= norm_k

        k_out_p = model_Grating_angle_to_angle(k_3, lba, m, G)
        k_out_p = k_out_p * norm_k

        k_3_bis = np.dot(rotation_y(0), k_out_p)

        # Rotation in relation to P2 around the X axis
        k_2_bis = np.dot(rotation_x(-delta_beta_c), k_3_bis)
        # Rotation in relation to P2 around the Y axis
        k_1_bis = np.dot(rotation_y(angle_with_P2), k_2_bis)

        theta_out = np.arctan(k_1_bis[0] / k_1_bis[2])
        phi_out = np.arctan(k_1_bis[1] / k_1_bis[2])

        [X_dmd, Y_dmd] = model_Lens_angle_to_position(theta_out, phi_out, F)

    return X_dmd, Y_dmd

def model_Grating_angle_to_angle(k_in, lba, m, G):


    alpha_in = np.arctan(k_in[0]) * np.sqrt(1 + np.tan(k_in[0])**2 + np.tan(k_in[1])**2)
    beta_in = np.arctan(k_in[1]) * np.sqrt(1 + np.tan(k_in[0])**2 + np.tan(k_in[1])**2)

    alpha_out = -1*np.arcsin(m * lba*10**-9  * G * 10**3 - np.sin(alpha_in))
    beta_out = beta_in


    k_out = [np.sin(alpha_out) * np.cos(beta_out), np.sin(beta_out)*np.cos(alpha_out),
                   np.cos(alpha_out) * np.cos(beta_out)]

    return k_out

def simplified_grating_in_out(alpha,lba,m,G):

    alpha_out = np.arcsin(m * lba * 10 ** -9 * G * 10 ** 3 - np.sin(alpha))

    return alpha_out

def model_Lens_angle_to_position(alpha, beta, F):

    x = F*np.tan(alpha)
    y  = F*np.tan(beta)

    return [x,y]

def model_Prism_angle_to_angle(k0, n, A):
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


def model_Lens_pos_to_angle(x_obj, y_obj, F):

    alpha = -1*np.arctan(x_obj / F)
    beta  = -1*np.arctan(y_obj / F)

    k0 = np.array([[np.sin(alpha) * np.cos(beta)], [np.sin(beta)*np.cos(alpha)],
                   [np.cos(alpha) * np.cos(beta)]])

    return k0


def D_m(n, A):

    # A should be given in radians
    return 2 * np.arcsin(n * np.sin(A / 2)) - A

def alpha_c(A, D_m):
    return (A + D_m) / 2


def sellmeier(lambda_):
    """
    Evaluating the refractive index value of a BK7 prism for a given lambda based on Sellmeier equation

    Parameters
    ----------
    lambda_ : float -- in nm
        input wavelength on the prism.

    Returns
    -------
    n : float
        index value corresponding to the input wavelength

    """

    B1 = 1.03961212;
    B2 = 0.231792344;
    B3 = 1.01046945;
    C1 = 6.00069867 * (10 ** -3);
    C2 = 2.00179144 * (10 ** -2);
    C3 = 1.03560653 * (10 ** 2);

    lambda_in_mm = lambda_ / 1000

    n = math.sqrt(1 + B1 * lambda_in_mm ** 2 / (lambda_in_mm ** 2 - C1) + B2 * lambda_in_mm ** 2 / (
                lambda_in_mm ** 2 - C2) + B3 * lambda_in_mm ** 2 / (lambda_in_mm ** 2 - C3));

    return n



