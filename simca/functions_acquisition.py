import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool


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

def generate_dd_measurement(scene, filtering_cube,chunk_size):
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
    filtered_scene = np.empty_like(filtering_cube)

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

    filtered_scene = np.nan_to_num(filtered_scene)

    return filtered_scene

def match_dataset_to_instrument(dataset, filtering_cube):
    """
    Match the size of the dataset to the size of the filtering cube. Either by padding or by cropping

    Args:
        dataset (numpy.ndarray): dataset
        filtering_cube (numpy.ndarray):  filtering cube of the instrument

    Returns:
        numpy.ndarray: observed scene (shape = R  x C x W)
    """



    if filtering_cube.shape[0] != dataset.shape[0] or filtering_cube.shape[1] != dataset.shape[1]:
        if dataset.shape[0] < filtering_cube.shape[0]:
            dataset = np.pad(dataset, ((np.floor((filtering_cube.shape[0] - dataset.shape[0])/2).astype('int'), np.ceil((filtering_cube.shape[0] - dataset.shape[0])/2).astype('int')), (0, 0), (0, 0)), mode="constant")
        if dataset.shape[1] < filtering_cube.shape[1]:
            dataset = np.pad(dataset, ((0, 0), (np.floor((filtering_cube.shape[1] - dataset.shape[1])/2).astype('int'), np.ceil((filtering_cube.shape[1] - dataset.shape[1])/2).astype('int')), (0, 0)), mode="constant")
        scene = dataset[0:filtering_cube.shape[0], 0:filtering_cube.shape[1], :]
        print("Dataset Spatial Cropping : Filtering cube and scene must have the same number of lines and columns")
        
    else:
        scene = dataset

    if len(filtering_cube.shape) == 3 and filtering_cube.shape[2] != dataset.shape[2]:
        scene = dataset[:, :, 0:filtering_cube.shape[2]]
        print("Dataset Spectral Cropping : Filtering cube and scene must have the same number of wavelengths")	
    else:
        scene = dataset
        
    try:
        scene
    except:
        print("Please load a scene first")

    return scene

def match_dataset_labels_to_instrument(dataset_labels, filtering_cube):
    """
    Match the size of the dataset labels to the size of the filtering cube. Either by padding or by cropping

    Args:
        dataset_labels (numpy.ndarray): dataset labels (shape = R_dts  x C_dts)
        filtering_cube (numpy.ndarray): filtering cube of the instrument

    Returns:
        numpy.ndarray: scene labels (shape = R  x C)
    """

    if filtering_cube.shape[0] != dataset_labels.shape[0] or filtering_cube.shape[1] != dataset_labels.shape[1]:
        if dataset_labels.shape[0] < filtering_cube.shape[0]:
            dataset_labels = np.pad(dataset_labels, ((0, filtering_cube.shape[0] - dataset_labels.shape[0])), mode="constant")
        if dataset_labels.shape[1] < filtering_cube.shape[1]:
            dataset_labels = np.pad(dataset_labels, ((0, 0), (0, filtering_cube.shape[1] - dataset_labels.shape[1])), mode="constant")
        dataset_labels = dataset_labels[0:filtering_cube.shape[0], 0:filtering_cube.shape[1]]
        print("Filtering cube and scene must have the same lines and columns")

    return dataset_labels

def crop_center(array, nb_of_pixels_along_x, nb_of_pixels_along_y):
    """
    Crop the given array to the given size, centered on the array

    Args:
        array (numpy.ndarray): 2D array to be cropped
        nb_of_pixels_along_x (int): number of samples to keep along the X axis
        nb_of_pixels_along_y (int): number of samples to keep along the Y axis

    Returns:
        numpy.ndarray: cropped array
    """

    y_len, x_len = array.shape

    x_start = x_len//2 - nb_of_pixels_along_x//2
    x_end = x_start + nb_of_pixels_along_x

    y_start = y_len//2 - nb_of_pixels_along_y//2
    y_end = y_start + nb_of_pixels_along_y

    if nb_of_pixels_along_x<array.shape[1]:
        array = array[:, x_start:x_end]

    if nb_of_pixels_along_y<array.shape[0]:
        array = array[y_start:y_end, :]

    return array

def interpolate_data_on_grid_positions(data, X_init, Y_init, X_target, Y_target, grid_type="unstructured", interp_method="linear"):
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
    interpolated_data = np.zeros((X_target.shape[0],X_target.shape[1],X_init.shape[2]))
    nb_of_grids = X_init.shape[2]

    if grid_type == "unstructured":
        worker = worker_unstructured
    elif grid_type == "regular":
        worker = worker_regulargrid

    if data.ndim == 2:
        data = data[:, :, np.newaxis]
        data = np.repeat(data, nb_of_grids, axis=2)

    with Pool(mp.cpu_count()) as p:

        tasks = [(X_init[:, :, i], Y_init[:, :, i], data[:, :, i], X_target, Y_target, interp_method) for i in
                 range(nb_of_grids)]

        for index, zi in tqdm(enumerate(p.imap(worker, tasks)), total=nb_of_grids,
                              desc='Interpolate 3D data on grid positions'):
            interpolated_data[:, :, index] = zi

    interpolated_data = np.nan_to_num(interpolated_data)

    return interpolated_data

def worker_unstructured(args):
    """
    Process to parallellize the unstructured griddata interpolation between the propagated grid (mask and the detector grid

    Args:
        args (tuple): containing the following elements: X_init_2D, Y_init_2D, data_2D, X_target_2D, Y_target_2D

    Returns:
        numpy.ndarray: 2D array of the data interpolated on the target grid
    """
    X_init_2D, Y_init_2D, data_2D, X_target_2D, Y_target_2D, interp_method = args
    interpolated_data = griddata((X_init_2D.flatten(),
                                  Y_init_2D.flatten()),
                                 data_2D.flatten(),
                                 (X_target_2D, Y_target_2D),
                                 method=interp_method)
    return interpolated_data

def worker_regulargrid(args):
    """
    Process to parallellize the structured griddata interpolation between the propagated grid (mask and the detector grid
    Note : For now it is identical to the unstructured method but it could be faster ...

    Args:
        args (tuple): containing the following elements: X_init_2D, Y_init_2D, data_2D, X_target_2D, Y_target_2D

    Returns:
        numpy.ndarray: 2D array of the data interpolated on the target grid
    """
    X_init_2D, Y_init_2D, data_2D, X_target_2D, Y_target_2D, interp_method = args

    interpolated_data = griddata((X_init_2D.flatten(),
                                  Y_init_2D.flatten()),
                                 data_2D.flatten(),
                                 (X_target_2D, Y_target_2D),
                                 method=interp_method)
    return interpolated_data






