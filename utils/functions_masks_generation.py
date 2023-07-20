import numpy as np


def generate_blue_noise(size):
    """
    Generate blue noise (high frequency pseudo-random) type mask
    Args:
        size:

    Returns:
        binary mask (numpy array): binary blue noise type mask

    """
    shape = (size[0], size[1])
    N = shape[0] * shape[1]
    rng = np.random.default_rng()
    noise = rng.standard_normal(N)
    noise = np.reshape(noise, shape)

    f_x = fftpack.fftfreq(shape[1])
    f_y = fftpack.fftfreq(shape[0])
    f_x_shift = fftpack.fftshift(f_x)
    f_y_shift = fftpack.fftshift(f_y)
    f_matrix = np.sqrt(f_x_shift[None, :] ** 2 + f_y_shift[:, None] ** 2)

    spectrum = fftpack.fftshift(fftpack.fft2(noise))
    filtered_spectrum = spectrum * f_matrix
    filtered_noise = fftpack.ifft2(fftpack.ifftshift(filtered_spectrum)).real

    # Make the mask binary
    threshold = np.median(filtered_noise)
    binary_mask = np.where(filtered_noise > threshold, 1, 0)

    return binary_mask