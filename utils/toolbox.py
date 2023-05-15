import math


import numpy as np
from scipy.ndimage import sobel
from scipy.stats import entropy
from sklearn.decomposition import PCA



def compute_sam(mean_spectrums, label_values, ignored_labels):
    sam_matrix = np.zeros((len(label_values), len(label_values)))
    for i in range(len(mean_spectrums) + len(ignored_labels)):
        if i not in ignored_labels:
            for j in range(i):
                if j not in ignored_labels:
                    sam = SAM(mean_spectrums[label_values[i]], mean_spectrums[label_values[j]])
                    sam_matrix[i, j] = sam
                    sam_matrix[j, i] = sam
    return sam_matrix


def compute_SNR(mean_spectrums, std_spectrums, label_values, ignored_labels):
    snr_tab = np.zeros(len(label_values))
    for i in range(len(mean_spectrums) + len(ignored_labels)):
        if i not in ignored_labels:
            mask_non_zero = std_spectrums[label_values[i]] > 1e-4
            snr = np.mean(mean_spectrums[label_values[i]][mask_non_zero] / std_spectrums[label_values[i]][mask_non_zero])
            snr_tab[i] = snr
    return snr_tab



def compute_bdist(mean_spectrums, cov_spectrums, label_values, ignored_labels):
    """
    Compute Bhattacharrya distances for each paris of classes
    
    :param mean_spectrums: dictionary (class_name -> mean_spectrum)
    :param cov_spectrums: dictionary (class_name -> spectrum_cov_matrix)
    :param label_values: list, class_names[i] = name of the ith class
    :param ignored_labels: list

    :return: bdist_matrix, np.array, shape=(n_class, n_class)
        bdist_matrix[i,j] = bdist_matrix[j,i] = battacharyya distance between class i and j
    """

    def log_det(x):
        eigv = np.linalg.eigvalsh(x)
        # eigv = eigv[eigv > 1e-20]
        logdet = np.sum(np.log(eigv))
        return logdet

    bdist_matrix = np.zeros((len(label_values), len(label_values)))

    for i in range(len(mean_spectrums) + len(ignored_labels)):
        if i not in ignored_labels:
            for j in range(i):
                if j not in ignored_labels:
                    m1 = mean_spectrums[label_values[i]]
                    m2 = mean_spectrums[label_values[j]]
                    s1 = cov_spectrums[label_values[i]]
                    s2 = cov_spectrums[label_values[j]]

                    m = m1 - m2
                    s = 0.5*(s1 + s2)
                    inv_s = np.linalg.pinv(s, hermitian=True)
                    logdet_s = log_det(s)
                    logdet_s1 = log_det(s1)
                    logdet_s2 = log_det(s2)

                    lin_term = (1 / 8.) * np.dot(np.transpose(m), np.dot(inv_s, m))
                    quad_term = 0.5 * (logdet_s - 0.5 * logdet_s1 - 0.5 * logdet_s2)

                    bdist_matrix[i, j] = lin_term + quad_term
                    bdist_matrix[j, i] = bdist_matrix[i, j]
    return bdist_matrix


def compute_sobel_img(img):
    """
    Compute the sobel image (= first derivative img = contours of the image) from the panchromatic version of img

    :param img: np.array, input cube

    :return: sobel_img, panchromatic_image
        2d np.array s
    """
    panchro = np.sum(img, axis=-1)

    # panchro = (panchro - np.min(panchro))/(np.std(panchro))
    panchro = (panchro - np.min(panchro))
    panchro = panchro / np.mean(panchro)
    sobel_img = sobel(panchro)
    sobel_img2 = sobel(panchro, axis=0)
    sobel_img3 = np.sqrt(sobel_img**2 + sobel_img2**2)
    return sobel_img3, panchro


def compute_entropy(panchro, sobel_img):
    """
    Compute shannon entropy of the panchromatic image and of its associated sobel_image (=image with contours)

    :param panchro: np.array
        Panchromatic version of the input cube = dataset
    :param sobel_img: np.array
        Image obtained by applying sobel filter to the panchromatic image, to extract contours

    :return: entropy of the panchro, entropy of the sobel image, (auxliary numbers)
    """
    simple_histo = np.histogram(panchro.flatten(), bins='auto')[0]
    first_order_histo = np.histogram(sobel_img.flatten(), bins='auto')[0]
    simple_entropy = entropy(simple_histo/len(simple_histo), base=2)
    first_order_entropy = entropy(first_order_histo/len(first_order_histo), base=2)
    return simple_entropy, first_order_entropy, (len(simple_histo), len(first_order_histo))


def write_entropy_results(viz, ent_tab):
    """
    Write in visdom the entropy results (= shannon entropy of the panchro and the sobel image)

    :param viz: Visdom vizualiser
    :param ent_tab: (entropy of the panchro, entropy of the sobel image, (auxliary numbers))

    :return: None
    """
    simple_ent = ent_tab[0]
    first_order_ent = ent_tab[1]
    s = "--- ENTROPY CARACTERISTICS --- \n \n"
    s += f"Shannon entropy of the image: {simple_ent:.2f}\n"
    s += f"Shannon entropy of the derivative image: {first_order_ent:.2f}\n"
    s += "\n\n"
    s += "PS : Derivative Image = Sobel image = gradient de l'image \n" \
         "A considérer : entropy de la derivative image. Plus elle est élevée, plus les motifs de l'image sont complexes."

    viz.text(s.replace('\n', '<br/>'), opts={'width': 300, 'height': 500})


def write_recap(params, shape_img, viz, b_score=None, sam_score=None, snr_score=None,sobel_carac=None, ent_results=None, label_values=None,
                ignored_labels=None, dataset_name=None):

    """
    Write in visdom a summary of everything that has been computed

    :param params: dictionary, input params
    :param shape_img: shape of the input cube
    :param viz: visdom visualiser
    :param b_score: [bdist_mean, bdist_std, bdist_quantile] (see plot_bdist)
    :param sobel_carac: [sobel_mean, sobel_std, sobel_quantile] (see plot_sobel)
    :param ent_results: [entropy of img, entropy of sobel img (img with contours), (...)] (see compute_entropy)
    :param label_values: list, class_names[i] = name of the ith class
    :param ignored_labels: list
    :param dataset_name: string

    :return: None
    """

    s = '\n\n--- RECAP FILE ---\n'
    s += '\n'
    s += 'CUBE PROPERTIES : \n'
    s += f'Name : {dataset_name} \n'
    s += f'Location : {params["dataset_folder"]}\n'
    s += f'Spatial Size : {shape_img[0]} x {shape_img[1]} \n'
    s += f'Number of spectral bands : {shape_img[2]} \n'
    s += '\n'

    if params['compute_sobel'] or params['compute_entropy']:
        s += 'CUBE COMPLEXITY METRICS :\n'
    if params['compute_sobel']:
        s += f'Sobel mean : {sobel_carac[0]:0.2f} \n'
        s += f"Sobel std : {sobel_carac[1]:0.2f} \n"
        s += f"Sobel 95% quantile : {sobel_carac[2]:0.2f} \n"
        s += '\n'
    if params['compute_entropy']:
        s += f'Shannon entropy of sobel image : {ent_results[0]:0.2f} \n'
        s += f'Shannon entropy of image : {ent_results[1]:0.2f} \n'
        s += '\n'

    if label_values is not None:
        s += 'CLASSIFICATION PROPERTIES : \n'
        s += f'Number of labels : {len(label_values)}\n'
        s += f'Ignored labels : {ignored_labels}\n'
        if params['compute_bdist']:
            s += f"B mean : {b_score[0]:0.2f} \n"
            s += f"B std : {b_score[1]:0.2f} \n"
            s += f"B 75% quantile : {b_score[2]:0.2f} \n"
        if params['compute_sam']:
            s += f"sam mean : {sam_score[0]:0.2f} \n"
            s += f"sam std : {sam_score[1]:0.2f} \n"
            s += f"sam 75% quantile : {sam_score[2]:0.2f} \n"
        if params['compute_snr']:
            s += f"snr mean : {snr_score[0]:0.2f} \n"
            s += f"snr std : {snr_score[1]:0.2f} \n"
            s += f"snr 75% quantile : {snr_score[2]:0.2f} \n"

    s += '\n\n'
    viz.text(s.replace('\n', '<br/>'), opts={'width': 300, 'height': 500})
    print(s)


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


def SAM(s1, s2):
    """
    Computes the spectral angle mapper between two vectors (in radians).

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            The angle between vectors s1 and s2 in radians.
    """
    try:
        s1_norm = math.sqrt(np.dot(s1, s1))
        s2_norm = math.sqrt(np.dot(s2, s2))
        sum_s1_s2 = np.dot(s1, s2)
        angle = math.acos(sum_s1_s2 / (s1_norm * s2_norm))
    except ValueError:
        # python math don't like when acos is called with
        # a value very near to 1
        return 0.0
    return angle

import plotly.graph_objects as go
def plot_all_spectrums(mean_spectrums, std_spectrums):
    """
    Display mean and intracovariance for each class, one graph per class

    :param mean_spectrums: dictionary, mean_spectrums[c] = mean spectrum of class c (c = string )
    :param std_spectrums: dictionary, std_spectrums[c] = mean spectrum of class c (c = string )

    :return: list of plotly figures
    """
    figures = []

    for c in mean_spectrums:
        spec = mean_spectrums[c]
        up_spec = spec + std_spectrums[c]
        low_spec = spec - std_spectrums[c]

        x = np.arange(len(spec))
        x_rev = x[::-1]
        low_spec = low_spec[::-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.concatenate([x, x_rev]), y=np.concatenate([up_spec, low_spec]), name=c,
                                 showlegend=False, fill='toself', fillcolor='violet', line_color='rgba(255,255,255,0)'))
        fig.add_trace(go.Scatter(x=x, y=spec, fillcolor='purple', showlegend=False, name=c))
        fig.update_layout(title=c)

        figures.append(fig)

    return figures


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_spectrums(spectrums, std_spectrums, palette, label_values):
    """Plot mean and intra-class covariance spectrum for each class on a single graph

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
        std_spectrums: dictionary (name -> std of spectrum) of spectrums to plot
        palette: dictionary, color palette
        label_values: list, label_values[i] = name of the ith class
    """
    fig, ax = plt.subplots()

    for k, v in spectrums.items():
        std_spec = std_spectrums[k]
        up_spec = v + std_spec
        low_spec = v - std_spec
        x = np.arange(len(v))
        i = label_values.index(k)
        ax.fill_between(x, up_spec, low_spec, color=sns.color_palette(palette)[i], alpha=0.3)

    for k, v in spectrums.items():
        x = np.arange(len(v))
        i = label_values.index(k)
        sns.lineplot(x=x, y=v, ax=ax, color=sns.color_palette(palette)[i], label=k)

    ax.set_title("Mean spectrum per class")
    plt.legend()
    plt.show()