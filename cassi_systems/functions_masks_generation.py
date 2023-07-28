import numpy as np
from scipy import fftpack

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

def generate_orthogonal_mask(size, W, N):
    """
    Generate an orthogonal mask according to https://hal.laas.fr/hal-02993037
    Args:
        - size (list of int): size of the mask
        - W (int): number of wavelengths in the scene
        - N (int): number of acquisitions
    Returns:
        - mask (numpy array of shape size[0] x (size[1]+W-1) x N): orthogonal mask
    """

    C, R = size[0], size[1] # Number of columns, number of rows

    K = C + W - 1 # Number of columns in H
    M = int(np.ceil(W/N)) # Number of open mirrors
    H_model = np.zeros((R, W, N)) # Periodic model
    for line in range(R):
        available_pos = list(range(W)) # Positions of possible mirrors to open
        for n in range(N):
            if available_pos:
                if (len(available_pos)>=M):
                    ind = np.random.choice(len(available_pos), M, replace=False) # Indices of the N positions among the available ones
                    pos = np.array(available_pos)[ind] # N mirrors to open
                else:
                    ind = list(range(len(available_pos)))
                    pos = np.array(available_pos)
                H_model[line, pos, n] = 1 # Open the mirrors
                for i in sorted(ind, reverse=True):
                    available_pos.pop(i) # Remove the positions of already opened mirrors

    mask = np.tile(H_model, [1, int(np.ceil(K/W)), 1])[:, :K, :] # Repeat the model periodically

    return mask

def generate_ln_orthogonal_mask(size, W, N):
    """
    Generate a Length-N orthogonal mask according to https://hal.laas.fr/hal-02993037
    Args:
        - size (list of int): size of the mask
        - W (int): number of wavelengths in the scene
        - N (int): number of acquisitions
    Returns:
        - mask (numpy array of shape size[0] x (size[1]+W-1) x N): length-N orthogonal mask
    """

    C, R = size[0], size[1] # Number of columns, number of rows

    K = C + W - 1 # Number of columns in H
    M = int(np.floor(W/N)) # Number of open mirrors
    H_model = np.zeros((R, W, N)) # Periodic model
    for line in range(R):
        for m in range(M):
            available = list(range(m*N, m*N+N)) # Positions of possible mirrors to open
            for n in range(N):
                ind = np.random.randint(len(available)) # Randomly choose a mirror among the possible ones
                H_model[line, available[ind], n] = 1 # Open the mirror
                available.pop(ind) # Remove the mirror from the list of possible ones
        available = list(range(M*N, W)) # List of positions where we can't apply the Length-N method (if W%N != 0)
        for n in range(W-(M*N)):
            ind = np.random.randint(len(available)) # Randomly open those mirrors among the remaining positions
            H_model[line, available[ind], n] = 1
            available.pop(ind)
    mask = np.tile(H_model, [1, int(np.ceil(K/W)), 1])[:, :K, :]

    return mask