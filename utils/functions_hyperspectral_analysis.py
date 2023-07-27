import numpy as np
from sklearn.decomposition import PCA

def explore_spectrums(img, complete_gt, class_names,
                      ignored_labels=None):
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