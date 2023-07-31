import os
from scipy import io
from imageio import imread
import spectral
import pickle as pkl
import numpy as np
import matplotlib.image as mpimg
import re
import seaborn as sns
import h5py
from sklearn.decomposition import PCA

def get_dataset(dataset_name, folder="./datasets/"):
    """Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to "./datasets/"
    Returns:
        img: 3D hyperspectral image (WxHxB)
        labels: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
    """

    folder = folder + dataset_name + "/"
    h5_file = h5py.File(folder + dataset_name + ".h5", "r")

    scene = np.array(h5_file["scene"],dtype=np.float32)

    # For matlab generated h5
    list_wavelengths = np.array(h5_file["wavelengths"]).tolist()[0]
    # for python generated h5
    if type(list_wavelengths) == float:
        list_wavelengths = np.array(h5_file["wavelengths"]).tolist()

    # No NaN accepted
    nan_mask = np.isnan(scene.sum(axis=-1))
    scene[nan_mask] = 0

    try:
        labels = np.array(h5_file["labels"], dtype=np.int8)
        label_names = [l[0] for l in h5_file['label_names'].asstr()[...]]
        ignored_labels = list(h5_file['ignored_labels'][...][0])
        labels[nan_mask] = 0
    except:
        labels = None
        label_names = None
        ignored_labels = None



    return scene, list_wavelengths, labels, label_names, ignored_labels


def palette_init(label_values):
    """Creates a palette for the classes
    """
    palette = {0: (0, 0, 0)}
    if label_values is None:
        return None
    else:
        for k, color in enumerate(sns.color_palette("hls", len(label_values) - 1)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
    return palette

from scipy.interpolate import interpn

def explore_spectrums(img, complete_gt, class_names,
                      ignored_labels=None, delta_lambda=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        vis : Visdom display
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """

    stats_per_class = {"mean_spectrums": {}, 'std_spectrums': {}, 'non_degenerate_mean_spectrums': {},
                       'non_degenerate_covariance': {}}
    n_samples_per_class = np.array([np.sum([complete_gt == i]) for i in np.unique(complete_gt) if i not in ignored_labels])

    """if delta_lambda is not None:
        n_dim_pca = int(np.min([0.25 * img.shape[-1] * np.log10(delta_lambda), 0.75 * np.min(n_samples_per_class)]))
    else:
        n_dim_pca = int(np.min([0.25*img.shape[-1], 0.75*np.min(n_samples_per_class)]))"""
    # n_dim_pca = int(np.min([0. * img.shape[-1], 0.5 * np.min(n_samples_per_class)]))
    n_dim_pca = 5
    mask = complete_gt > 0
    pca = PCA(n_dim_pca)
    pca.fit(img[mask])

    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask]
        # pca = PCA(n_dim_pca)
        class_spectrums_pca = pca.transform(class_spectrums)

        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)

        stats_per_class['mean_spectrums'][class_names[c]] = mean_spectrum
        stats_per_class['std_spectrums'][class_names[c]] = std_spectrum
        stats_per_class['non_degenerate_mean_spectrums'][class_names[c]] = np.mean(class_spectrums_pca, axis=0)
        stats_per_class['non_degenerate_covariance'][class_names[c]] = np.cov(np.transpose(class_spectrums_pca))

    return stats_per_class



def interpolate_dataset_cube_along_wavelength(scene, scene_wavelengths, new_wavelengths_sampling, chunk_size=50):
    # Generate the coordinates for the original grid
    x = np.arange(scene.shape[0])
    y = np.arange(scene.shape[1])
    z = scene_wavelengths

    # Generate the coordinates for the new grid
    new_z = new_wavelengths_sampling

    # Initialize an empty array for the result
    interpolated_scene = np.empty((scene.shape[0], scene.shape[1], len(new_z)))

    # Perform the interpolation in chunks
    for i in range(0, scene.shape[0], chunk_size):
        for j in range(0, scene.shape[1], chunk_size):
            new_coordinates = np.meshgrid(x[i:i+chunk_size], y[j:j+chunk_size], new_z, indexing='ij')
            interpolated_scene[i:i+chunk_size, j:j+chunk_size, :] = interpn((x[i:i+chunk_size], y[j:j+chunk_size], z), scene[i:i+chunk_size, j:j+chunk_size, :], tuple(new_coordinates))

    print("interpolated_scene.shape", interpolated_scene.shape)

    return interpolated_scene