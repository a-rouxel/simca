from utils.helpers import *




def propagate_through_arm_vector(dispersive_element_type,X_mask,Y_mask,n, lba,A,G,m,F, alpha_c,alpha_c_transmis,delta_alpha_c, delta_beta_c):
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


    if dispersive_element_type == "prism":
        angle_with_P1 = alpha_c - A / 2 + delta_alpha_c
        angle_with_P2 = alpha_c_transmis - A / 2 - delta_alpha_c
    elif dispersive_element_type == "grating":
        angle_with_P1 = alpha_c - delta_alpha_c
        angle_with_P2 = alpha_c_transmis + delta_alpha_c

    k = l_p_a_V2(X_mask, Y_mask, F)
    # Rotation in relation to P1 around the Y axis

    k_1 = rotation_y(angle_with_P1) @ k[:,0,:]
    # Rotation in relation to P1 around the X axis
    k_2 = rotation_x(delta_beta_c) @ k_1
    # Rotation of P1 in relation to frame_in along the new Y axis
    if dispersive_element_type == 'prism':
        k_3 = rotation_y(A / 2) @ k_2
    elif dispersive_element_type == 'grating':
        k_3 = rotation_y(0) @ k_2

    norm_k = np.sqrt(k_3[0] ** 2 + k_3[1] ** 2 + k_3[2] ** 2)
    k_3 /= norm_k

    # Output vector of the prism in frame_out
    if dispersive_element_type == 'prism':
        k_out_p = prism_in_to_prism_out_parallel_dmd_scene(k_3, n, A)
    elif dispersive_element_type == 'grating':
        k_out_p = grating_in_to_grating_out(k_3, lba, m, G)

    k_out_p = k_out_p * norm_k

    if dispersive_element_type == 'prism':
        k_3_bis = np.dot(rotation_y(A / 2), k_out_p)
    elif dispersive_element_type == 'grating':
        k_3_bis = np.dot(rotation_y(0), k_out_p)

    # Rotation in relation to P2 around the X axis
    k_2_bis = np.dot(rotation_x(-delta_beta_c), k_3_bis)
    # Rotation in relation to P2 around the Y axis
    k_1_bis = np.dot(rotation_y(angle_with_P2), k_2_bis)

    theta_out = np.arctan(k_1_bis[0] / k_1_bis[2])
    phi_out = np.arctan(k_1_bis[1] / k_1_bis[2])

    [X_dmd, Y_dmd] = l_a_p_V2(theta_out,  phi_out, F)

    print(angle_with_P1, angle_with_P2)


    return X_dmd, Y_dmd

def grating_in_to_grating_out(k_in, lba, m, G):


    alpha_in = np.arctan(k_in[0]) * np.sqrt(1 + np.tan(k_in[0])**2 + np.tan(k_in[1])**2)
    beta_in = np.arctan(k_in[1]) * np.sqrt(1 + np.tan(k_in[0])**2 + np.tan(k_in[1])**2)

    # print(m)
    # print(lba)
    # print(G)
    # print(np.sin(alpha_in))

    alpha_out = -1*np.arcsin(m * lba*10**-9  * G * 10**3 - np.sin(alpha_in))
    beta_out = beta_in

    # print(alpha_in, alpha_out)

    k_out = [np.sin(alpha_out) * np.cos(beta_out), np.sin(beta_out)*np.cos(alpha_out),
                   np.cos(alpha_out) * np.cos(beta_out)]

    return k_out

def simplified_grating_in_out(alpha,lba,m,G):
    alpha_out = np.arcsin(m * lba * 10 ** -9 * G * 10 ** 3 - np.sin(alpha))

    return alpha_out



def calib_grating_beta_cam_dmd(alpha, lba, m, G):
    """
    According to the grating equation, calculating the diffracted angle for given parameters

    Parameters
    ----------
    alpha : float64
        incident angle on the grating for the central spatial position -- in rad
    lba : float64
        wavelength of the diffracted beam   -- in nm
    m : integer
        order of diffraction. -- no units
    G : integer
        grating density  -- in lines/mm

    Returns
    -------
    beta_ : float
            angle of the diffracted beam

    """

    beta_ = np.arcsin(m * lba * 10 ** -9 * G * 1000 - np.sin(alpha));

    return beta_

def propagate_through_arm_scalar(X_mask,Y_mask,n, A,F, alpha_c,delta_alpha_c, delta_beta_c):
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

    angle_with_P1 = alpha_c - A / 2 + delta_alpha_c
    angle_with_P2 = alpha_c - A / 2 - delta_alpha_c

    k = l_p_a_V2(X_mask, Y_mask, F)
    # Rotation in relation to P1 around the Y axis

    k_1 = rotation_y(angle_with_P1) @ k

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


def D_m(n, A):
    # A should be given in radians
    print(2 * np.arcsin(n * np.sin(A / 2)) - A)
    return 2 * np.arcsin(n * np.sin(A / 2)) - A

def alpha_c(A, D_m):
    return (A + D_m) / 2

def rotation_z(theta):
    """
    Rotate 3D matrix around the z axis

    Parameters
    ----------
    theta : float -- in rad
        Input angle.

    Returns
    -------
    r : 2D numpy array
        The rotation matrix.

    """

    r = np.array(((np.cos(theta), -np.sin(theta), 0),
                  (np.sin(theta), np.cos(theta), 0),
                  (0, 0, 1)));

    return r


def rotation_y(theta):
    """
    Rotate 3D matrix around the y axis

    Parameters
    ----------
    theta : float -- in rad
        Input angle.

    Returns
    -------
    r : 2D numpy array
        The rotation matrix.
    """

    r = np.array(((np.cos(theta), 0, np.sin(theta)),
                  (0, 1, 0),
                  (-np.sin(theta), 0, np.cos(theta))));

    return r


def rotation_x(theta):
    """
    Rotate 3D matrix around the x axis

    Parameters
    ----------
    theta : float -- in rad
        Input angle.

    Returns
    -------
    r : 2D numpy array
        The rotation matrix.
    """

    r = np.array(((1, 0, 0),
                  (0, math.cos(theta), -math.sin(theta)),
                  (0, math.sin(theta), math.cos(theta))));

    return r

def sellmeier(lambda_):
    """
    Evaluating the refractive index value of a BK7 prism for a given lambda

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



