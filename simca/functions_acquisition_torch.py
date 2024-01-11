import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.nn.unpool import knn_interpolate

# TODO: sd measurement torch
def generate_sd_measurement_cube(filtered_scene,X_input, Y_input, X_target, Y_target,grid_type,interp_method):
    """
    Generate SD measurement cube from the coded aperture and the scene.
    For Single-Disperser CASSI systems, the scene is filtered then propagated in the detector plane.

    Args:
        filtered_scene (numpy.ndarray): filtered scene (shape = R x C x W)

    Returns:
        numpy.ndarray: SD measurement cube (shape = R x C x W)
    """

    print("--- Generating SD measurement cube ---- ")

    measurement_sd = interpolate_data_on_grid_positions(filtered_scene,
                                                             X_input,
                                                             Y_input,
                                                             X_target,
                                                             Y_target,
                                                             grid_type=grid_type,
                                                             interp_method=interp_method)
    return measurement_sd


def generate_dd_measurement_torch(scene, filtering_cube,chunk_size):
    """
    Generate DD-CASSI type system measurement from a scene and a filtering cube. ref : "Single-shot compressive spectral imaging with a dual-disperser architecture", M.Gehm et al., Optics Express, 2007

    Args:
        scene (numpy.ndarray): observed scene (shape = R  x C x W)
        filtering_cube (numpy.ndarray):   filtering cube of the instrument for a given pattern (shape = R x C x W)
        chunk_size (int) : size of the spatial chunks in which the Hadamard product is performed

    Returns:
        numpy.ndarray: filtered scene (shape = R x C x W)
    """

    # Initialize an empty array for the result
    filtered_scene = torch.empty_like(filtering_cube)

    # Calculate total iterations for tqdm
    total_iterations = (filtering_cube.shape[0] // chunk_size + 1) * (filtering_cube.shape[1] // chunk_size + 1)

    with tqdm(total=total_iterations) as pbar:
        # Perform the multiplication in chunks
        for i in range(0, filtering_cube.shape[0], chunk_size):
            for j in range(0, filtering_cube.shape[1], chunk_size):
                filtered_scene[i:i + chunk_size, j:j + chunk_size, :] = filtering_cube[i:i + chunk_size,
                                                                           j:j + chunk_size, :] * scene[
                                                                                                  i:i + chunk_size,
                                                                                                  j:j + chunk_size,
                                                                                                  :]
                pbar.update()

    filtered_scene = torch.nan_to_num(filtered_scene)

    return filtered_scene

def interpolate_data_on_grid_positions_torch(data, X_init, Y_init, X_target, Y_target, grid_type="unstructured", interp_method="linear"):
    """
    Interpolate data on a single 2D grid defined by X_target and Y_target

    Args:
        data (numpy.ndarray): data to interpolate (3D or 2D)
        X_init (numpy.ndarray): X coordinates of the initial grid (3D)
        Y_init (numpy.ndarray): Y coordinates of the initial grid (3D)
        X_target (numpy.ndarray): X coordinates of the target grid (2D)
        Y_target (numpy.ndarray): Y coordinates of the target grid (2D)
        grid_type (str): type of the target grid (default = "unstructured", other option = "regular")
        interp_method (str): interpolation method (default = "linear")

    Returns:
        numpy.ndarray: 3D data interpolated on the target grid
    """

    X_init = torch.from_numpy(X_init).double() if isinstance(X_init, np.ndarray) else X_init
    Y_init = torch.from_numpy(Y_init).double() if isinstance(Y_init, np.ndarray) else Y_init
    X_target = torch.from_numpy(X_target).double() if isinstance(X_target, np.ndarray) else X_target
    Y_target = torch.from_numpy(Y_target).double() if isinstance(Y_target, np.ndarray) else Y_target
    data = torch.from_numpy(data).double() if isinstance(data, np.ndarray) else data

    interpolated_data = torch.zeros((X_target.shape[0],X_target.shape[1],X_init.shape[2]))
    nb_of_grids = X_init.shape[2]

    if grid_type == "unstructured":
        worker = worker_unstructured_torch
    elif grid_type == "regular":
        worker = worker_regulargrid_torch
    
    if data.ndim == 2:
        data = data[:, :, None]
        data = torch.repeat_interleave(data, nb_of_grids, dim=2)

    tasks = [(X_init[:, :, i], Y_init[:, :, i], data[:, :, i], X_target, Y_target, interp_method) for i in
                range(nb_of_grids)]

    for index, zi in tqdm(enumerate(tasks), total=nb_of_grids,
                            desc='Interpolate 3D data on grid positions'):
        interpolated_data[:, :, index] = worker(zi)
    

    """X_target = X_target[..., None]
    X_target = torch.repeat_interleave(X_target, nb_of_grids, axis=2)
    Y_target = Y_target[..., None]
    Y_target = torch.repeat_interleave(Y_target, nb_of_grids, axis=2)
    interpolated_data = worker([X_init, Y_init, data, X_target, Y_target, interp_method])"""

    interpolated_data = torch.nan_to_num(interpolated_data)

    return interpolated_data

def worker_unstructured_torch(args):
    """
    Process to parallellize the unstructured griddata interpolation between the propagated grid (mask and the detector grid

    Args:
        args (tuple): containing the following elements: X_init_2D, Y_init_2D, data_2D, X_target_2D, Y_target_2D

    Returns:
        torch.tensor: 2D array of the data interpolated on the target grid
    """
    X_init_2D, Y_init_2D, data_2D, X_target_2D, Y_target_2D, interp_method = args
    data = data_2D.flatten()[:, None]
    init = torch.stack((X_init_2D.flatten(), Y_init_2D.flatten()), dim=-1)
    target = torch.stack((X_target_2D.flatten(), Y_target_2D.flatten()), dim=-1)
    """Essai 3D
    data = data_2D.flatten(0, 1)
    init = torch.cat((X_init_2D.flatten(0,1), Y_init_2D.flatten(0,1)), dim = 1)
    init = torch.cat([torch.stack((X_init_2D.flatten(0,1)[...,i], Y_init_2D.flatten(0,1)[...,i]), dim=-1) for i in range(3)], dim=-1)
    target = torch.cat((X_target_2D.flatten(0, 1), Y_target_2D.flatten(0, 1)), dim=1)
    target = torch.cat([torch.stack((X_target_2D.flatten(0,1)[...,i], Y_target_2D.flatten(0,1)[...,i]), dim=-1) for i in range(3)], dim=-1)"""
    interpolated_data = knn_interpolate(data,
                                 init,
                                 target,
                                 k=3)
    return interpolated_data.reshape(X_target_2D.shape)

def worker_regulargrid_torch(args):
    """
    Process to parallellize the structured griddata interpolation between the propagated grid (mask and the detector grid
    Note : For now it is identical to the unstructured method but it could be faster ...

    Args:
        args (tuple): containing the following elements: X_init_2D, Y_init_2D, data_2D, X_target_2D, Y_target_2D

    Returns:
        torch.tensor: 2D array of the data interpolated on the target grid
    """
    X_init_2D, Y_init_2D, data_2D, X_target_2D, Y_target_2D, interp_method = args
    data = data_2D.flatten()[:, None]
    init = torch.stack((X_init_2D.flatten(), Y_init_2D.flatten()), dim=-1)
    target = torch.stack((X_target_2D.flatten(), Y_target_2D.flatten()), dim=-1)
    
    interpolated_data = knn_interpolate(data,
                                 init,
                                 target,
                                 k=3)
    return interpolated_data.reshape(X_target_2D.shape)





