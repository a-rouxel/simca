import numpy as np
from tqdm import tqdm

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


def match_dataset_to_instrument(scene, filtering_cube):
    """
    Match the size of the dataset to the size of the filtering cube. Either by padding or by cropping

    Args:
        dataset (numpy.ndarray): dataset
        filtering_cube (numpy.ndarray):  filtering cube of the instrument

    Returns:
        numpy.ndarray: observed scene (shape = R  x C x W)
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






