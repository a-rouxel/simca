import numpy as np
from tqdm import tqdm
def get_measurement_in_3D(scene, filtering_cube,chunk_size):

    # Initialize an empty array for the result
    measurement_in_3D = np.empty_like(filtering_cube)

    # Calculate total iterations for tqdm
    total_iterations = (filtering_cube.shape[0] // chunk_size + 1) * (filtering_cube.shape[1] // chunk_size + 1)

    with tqdm(total=total_iterations) as pbar:
        # Perform the multiplication in chunks
        for i in range(0, filtering_cube.shape[0], chunk_size):
            for j in range(0, filtering_cube.shape[1], chunk_size):
                measurement_in_3D[i:i + chunk_size, j:j + chunk_size, :] = filtering_cube[i:i + chunk_size,
                                                                           j:j + chunk_size, :] * scene[
                                                                                                  i:i + chunk_size,
                                                                                                  j:j + chunk_size,
                                                                                                  :]
                pbar.update()

    measurement_in_3D = np.nan_to_num(measurement_in_3D)

    return measurement_in_3D


def match_scene_to_instrument(scene, filtering_cube):
    if filtering_cube.shape[0] != scene.shape[0] or filtering_cube.shape[1] != scene.shape[1]:
        if scene.shape[0] < filtering_cube.shape[0]:
            scene = np.pad(scene, ((0, filtering_cube.shape[0] - scene.shape[0]), (0, 0), (0, 0)), mode="constant")
        if scene.shape[1] < filtering_cube.shape[1]:
            scene = np.pad(scene, ((0, 0), (0, filtering_cube.shape[1] - scene.shape[1]), (0, 0)), mode="constant")
        scene = scene[0:filtering_cube.shape[0], 0:filtering_cube.shape[1], :]
        print("Filtering cube and scene must have the same lines and columns")


        if filtering_cube.shape[2] != scene.shape[2]:
            scene = scene[:, :, 0:filtering_cube.shape[2]]
            print("Filtering cube and scene must have the same number of wavelengths")

    return scene