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

def crop_center(array_x, array_y, nb_of_samples_along_x, nb_of_samples_along_y):
    

    y_len, x_len = array_x.shape

    x_start = x_len//2 - nb_of_samples_along_x//2
    x_end = x_start + nb_of_samples_along_x

    y_start = y_len//2 - nb_of_samples_along_y//2
    y_end = y_start + nb_of_samples_along_y

    if nb_of_samples_along_x<array_x.shape[1]:
        array_x_crop = array_x[:, x_start:x_end]
    if nb_of_samples_along_y<array_x.shape[0]:
        array_y_crop = array_y[y_start:y_end, :]
    if nb_of_samples_along_y>=array_x.shape[0] and nb_of_samples_along_x>=array_x.shape[1]:
        array_x_crop = array_x
        array_y_crop = array_y

    return array_x_crop, array_y_crop


def generate_2D_gaussian(radius, sample_size_x,sample_size_y, nb_of_samples):
    # Define the grid
    grid_size_x = sample_size_x * (nb_of_samples - 1)
    grid_size_y = sample_size_y * (nb_of_samples - 1)
    x = np.linspace(-grid_size_x / 2, grid_size_x / 2, nb_of_samples)
    y = np.linspace(-grid_size_y / 2, grid_size_y / 2, nb_of_samples)
    X, Y = np.meshgrid(x, y)

    # Compute the 2D Gaussian function
    gaussian_2d = np.exp(-(X**2 + Y**2) / (2 * radius**2))

    return X, Y, gaussian_2d



