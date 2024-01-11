import numpy as np
import seaborn as sns
import h5py
from sklearn.decomposition import PCA

def get_dataset(dataset_name, folder="./datasets/"):
    """Gets the dataset specified by name and return the related components.
    Args:
        dataset_name (str): the name of the dataset
        folder (str): folder where the datasets are stored, defaults to "./datasets/"
    Returns
        numpy.ndarray: 3D hyperspectral image (WxHxB)
        numpy.ndarray: 2D array of labels (integers)
        list: list of class names
        ignored_labels: list of int classes to ignore
    """

    folder = folder + dataset_name + "/"
    h5_file = h5py.File(folder + dataset_name + ".h5", "r")

    scene = np.array(h5_file["scene"],dtype=np.float32)

    # For matlab generated h5
    wavelengths_vec = np.array(h5_file["wavelengths"])[0]
    # for python generated h5
    if wavelengths_vec.shape[0] == 1:
        wavelengths_vec = np.array(h5_file["wavelengths"])

    # No NaN accepted
    nan_mask = np.isnan(scene.sum(axis=-1))
    scene[nan_mask] = 0

    try:
        labels = np.array(h5_file["labels"], dtype=np.int8)
        label_names = [l[0] for l in h5_file['label_names'].asstr()[...]]

        try :
            ignored_labels = list(h5_file['ignored_labels'][...][0])
        except:
            ignored_labels = list(h5_file['ignored_labels'][...])


        labels[nan_mask] = 0


    except:
        labels = None
        label_names = None
        ignored_labels = None



    return scene, wavelengths_vec, labels, label_names, ignored_labels


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
                      ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
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



def interpolate_data_along_wavelength(data, current_sampling, new_sampling, chunk_size=50):
    """Interpolate the input 3D data along a new sampling in the third axis.

    Args:
        data (numpy.ndarray): 3D data to interpolate
        current_sampling (numpy.ndarray): current sampling for the 3rd axis
        new_sampling (numpy.ndarray): new sampling for the 3rd axis
        chunk_size (int): size of the chunks to use for the interpolation
    """

    # Generate the coordinates for the original grid
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = current_sampling

    # Initialize an empty array for the result
    interpolated_data = np.empty((data.shape[0], data.shape[1], len(new_sampling)))

    # Perform the interpolation in chunks
    for i in range(0, data.shape[0], chunk_size):
        for j in range(0, data.shape[1], chunk_size):
            new_coordinates = np.meshgrid(x[i:i+chunk_size], y[j:j+chunk_size], new_sampling, indexing='ij')
            interpolated_data[i:i+chunk_size, j:j+chunk_size, :] = interpn((x[i:i+chunk_size], y[j:j+chunk_size], z), data[i:i+chunk_size, j:j+chunk_size, :], tuple(new_coordinates))

    return interpolated_data