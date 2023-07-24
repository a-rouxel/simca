import numpy as np


def generate_spectral_angular_map_optimized(recons_scene, interpolated_scene):
    # Calculate norms for each pair of corresponding vectors in recons_scene and interpolated_scene
    recons_norms = np.linalg.norm(recons_scene, axis=-1)
    interpolated_norms = np.linalg.norm(interpolated_scene, axis=-1)

    # Calculate dot product for each pair of corresponding vectors in recons_scene and interpolated_scene
    dot_products = np.einsum('ijk,ijk->ij', recons_scene, interpolated_scene)

    # Calculate cosines of angles between corresponding vectors in recons_scene and interpolated_scene
    cosines = dot_products / (recons_norms * interpolated_norms)

    # Calculate Spectral Angular Mapper (SAM)
    sam = (2 / np.pi) * np.abs(np.arccos(cosines))

    return sam

