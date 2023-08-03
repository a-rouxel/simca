import numpy as np
from tqdm import tqdm

def generate_dd_measurement(scene, filtering_cube,chunk_size):
    """
    Generate DD-CASSI type system measurement from a scene and a filtering cube. ref : "Single-shot compressive spectral imaging with a dual-disperser architecture", M.Gehm et al., Optics Express, 2007

    Args:
        scene (numpy array): 3D array of the scene to be measured
        filtering_cube (numpy array): 3D array corresponding to the spatio-spectral filtering cube of the instrument
        chunk_size (int) : size of the spatial chunks in which the Hadamard product is performed

    Returns:
        filtered_scene (numpy array): 3D array of the filtered scene
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


def match_scene_to_instrument(scene, filtering_cube):
    """
    Match the size of the scene to the size of the filtering cube. Either by padding or by cropping

    Args:
        scene (numpy array): 3D array of the scene to be measured
        filtering_cube (numpy array): 3D array corresponding to the spatio-spectral filtering cube of the instrument

    Returns:
        scene (numpy array): 3D array of the scene to be measured, matched to the size of the filtering cube
    """

    if filtering_cube.shape[0] != scene.shape[0] or filtering_cube.shape[1] != scene.shape[1]:
        if scene.shape[0] < filtering_cube.shape[0]:
            scene = np.pad(scene, ((0, filtering_cube.shape[0] - scene.shape[0]), (0, 0), (0, 0)), mode="constant")
        if scene.shape[1] < filtering_cube.shape[1]:
            scene = np.pad(scene, ((0, 0), (0, filtering_cube.shape[1] - scene.shape[1]), (0, 0)), mode="constant")
        scene = scene[0:filtering_cube.shape[0], 0:filtering_cube.shape[1], :]
        print("Filtering cube and scene must have the same lines and columns")

    if len(filtering_cube.shape) == 3:
        if filtering_cube.shape[2] != scene.shape[2]:
            scene = scene[:, :, 0:filtering_cube.shape[2]]
            print("Filtering cube and scene must have the same number of wavelengths")

    return scene

def match_scene_labels_to_instrument(dataset_labels, filtering_cube):
    """
    Match the size of the dataset labels to the size of the filtering cube. Either by padding or by cropping

    Args:
        dataset_labels (numpy array): 2D array of the scene to be measured
        filtering_cube (numpy array): 3D array corresponding to the spatio-spectral filtering cube of the instrument

    Returns:
        cropped dataset_labels (numpy array): 2D array of the scene labels, matched to the size of the filtering cube
    """

    if filtering_cube.shape[0] != dataset_labels.shape[0] or filtering_cube.shape[1] != dataset_labels.shape[1]:
        if dataset_labels.shape[0] < filtering_cube.shape[0]:
            dataset_labels = np.pad(dataset_labels, ((0, filtering_cube.shape[0] - dataset_labels.shape[0])), mode="constant")
        if dataset_labels.shape[1] < filtering_cube.shape[1]:
            dataset_labels = np.pad(dataset_labels, ((0, 0), (0, filtering_cube.shape[1] - dataset_labels.shape[1])), mode="constant")
        dataset_labels = dataset_labels[0:filtering_cube.shape[0], 0:filtering_cube.shape[1]]
        print("Filtering cube and scene must have the same lines and columns")

    return dataset_labels

def crop_center(array, nb_of_samples_along_x, nb_of_samples_along_y):
    """
    Crop the given array to the given size, centered on the array

    Args:
        array (numpy array): 2D array to be cropped
        nb_of_samples_along_x (int): number of samples to keep along the x axis
        nb_of_samples_along_y (int): number of samples to keep along the y axis

    Returns:
        array (numpy array): 2D array cropped
    """

    y_len, x_len = array.shape

    x_start = x_len//2 - nb_of_samples_along_x//2
    x_end = x_start + nb_of_samples_along_x

    y_start = y_len//2 - nb_of_samples_along_y//2
    y_end = y_start + nb_of_samples_along_y

    if nb_of_samples_along_x<array.shape[1]:
        array = array[:, x_start:x_end]

    if nb_of_samples_along_y<array.shape[0]:
        array = array[y_start:y_end, :]

    return array


def generate_2D_gaussian(radius, sample_size_x,sample_size_y, nb_of_samples):
    """
    Generate a 2D Gaussian of a given radius

    Args:
        radius (float): radius of the Gaussian
        sample_size_x (float): size of each sample along the x axis
        sample_size_y (float): size of each sample along the y axis
        nb_of_samples (int): number of samples along each axis

    Returns:
        X (numpy array): 2D array of the x coordinates of the grid
        Y (numpy array): 2D array of the y coordinates of the grid
        gaussian_2d (numpy array): array of the 2D Gaussian
    """

    # Define the grid
    grid_size_x = sample_size_x * (nb_of_samples - 1)
    grid_size_y = sample_size_y * (nb_of_samples - 1)
    x = np.linspace(-grid_size_x / 2, grid_size_x / 2, nb_of_samples)
    y = np.linspace(-grid_size_y / 2, grid_size_y / 2, nb_of_samples)
    X, Y = np.meshgrid(x, y)

    # Compute the 2D Gaussian function
    gaussian_2d = np.exp(-(X**2 + Y**2) / (2 * radius**2))

    return X, Y, gaussian_2d



