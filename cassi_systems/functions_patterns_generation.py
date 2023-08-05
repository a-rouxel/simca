import numpy as np
from scipy import fftpack
from scipy import ndimage
import h5py
def generate_blue_noise_type_1_pattern(shape):
    """
    Generate blue noise (high frequency pseudo-random) type pattern

    Args:
        shape (tuple of int): shape of the pattern

    Returns:
        numpy.ndarray: binary blue noise type pattern

    """

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

    # Make the pattern binary
    threshold = np.median(filtered_noise)
    binary_pattern = np.where(filtered_noise > threshold, 1, 0)

    return binary_pattern

def generate_orthogonal_pattern(size, W, N):
    """
    Generate an orthogonal pattern according to https://hal.laas.fr/hal-02993037

    Args:
        size (list of int): size of the pattern
        W (int): number of wavelengths in the scene
        N (int): number of acquisitions

    Returns:
        numpy.ndarray: orthogonal pattern :  shape =  size[0] x (size[1]+W-1) x N):
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

    pattern = np.tile(H_model, [1, int(np.ceil(K/W)), 1])[:, :K, :] # Repeat the model periodically

    return pattern

def generate_ln_orthogonal_pattern(size, W, N):
    """
    Generate a Length-N orthogonal pattern according to https://hal.laas.fr/hal-02993037

    Args:
        size (list of int): size of the pattern
        W (int): number of wavelengths in the scene
        N (int): number of acquisitions

    Returns:
        numpy.ndarray : length-N orthogonal pattern of shape = size[0] x (size[1]+W-1) x N):
    """

    C, R = size[1], size[0] # Number of columns, number of rows

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
    pattern = np.tile(H_model, [1, int(np.ceil(K/W)), 1])[:, :K, :]

    list_of_patterns = []
    for i in range(pattern.shape[2]):
        m = pattern[:, :-(W-1), i]
        list_of_patterns.append(m)

    return list_of_patterns

def generate_random_pattern(shape, ROM):
    """
    Generate a random pattern with a given rate of open/close mirrors

    Args:
        shape (tuple of int): shape of the pattern
        ROM (float): ratio of open mirrors

    Returns:
        numpy.ndarray: random pattern
    """

    pattern = np.random.choice([0, 1], size=shape, p=[1 - ROM, ROM])

    return pattern

def generate_slit_pattern(shape, slit_position,slit_width):
    """
    Generate a slit pattern that starts at the center of the image and goes to the right as slit position increases.

    Args:
        shape (tuple): shape of the pattern
        slit_position (int): position of the slit in relation to the central column
        slit_width (int): width of the slit in pixels

    Returns:
        numpy.ndarray: slit pattern

    """
    size_y, size_x = shape[0], shape[1]
    slit_position = size_x // 2 + slit_position
    slit_width = slit_width
    pattern = np.zeros((size_y,size_x))

    pattern[:, slit_position - slit_width // 2:slit_position + slit_width] = 1

    return pattern

# Source of blue noise codes: https://momentsingraphics.de/BlueNoise.html
def FindLargestVoid(BinaryPattern,StandardDeviation):
    """This function returns the indices of the largest void in the given binary
       pattern as defined by Ulichney.
      \param BinaryPattern A boolean array (should be two-dimensional although the
             implementation works in arbitrary dimensions).
      \param StandardDeviation The standard deviation used for the Gaussian filter
             in pixels. This can be a single float for an isotropic Gaussian or a
             tuple with one float per dimension for an anisotropic Gaussian.
      \return A flat index i such that BinaryPattern.flat[i] corresponds to the
              largest void. By definition this is a majority pixel.
      \sa GetVoidAndClusterBlueNoise"""
    # The minority value is always True for convenience
    if(np.count_nonzero(BinaryPattern)*2>=np.size(BinaryPattern)):
        BinaryPattern=np.logical_not(BinaryPattern)
    # Apply the Gaussian. We do not want to cut off the Gaussian at all because even
    # the tiniest difference can change the ranking. Therefore we apply the Gaussian
    # through a fast Fourier transform by means of the convolution theorem.
    FilteredArray=np.fft.ifftn(ndimage.fourier_gaussian(np.fft.fftn(np.where(BinaryPattern,1.0,0.0)),StandardDeviation)).real
    # Find the largest void
    return np.argmin(np.where(BinaryPattern,2.0,FilteredArray))

# Source of blue noise codes: https://momentsingraphics.de/BlueNoise.html
def FindTightestCluster(BinaryPattern,StandardDeviation):
    """Like FindLargestVoid() but finds the tightest cluster which is a minority
       pixel by definition.
      \sa GetVoidAndClusterBlueNoise"""
    if(np.count_nonzero(BinaryPattern)*2>=np.size(BinaryPattern)):
        BinaryPattern=np.logical_not(BinaryPattern)
    FilteredArray=np.fft.ifftn(ndimage.fourier_gaussian(np.fft.fftn(np.where(BinaryPattern,1.0,0.0)),StandardDeviation)).real
    return np.argmax(np.where(BinaryPattern,FilteredArray,-1.0))

# Source of blue noise codes: https://momentsingraphics.de/BlueNoise.html
def GetVoidAndClusterBlueNoise(OutputShape,StandardDeviation=1.5,InitialSeedFraction=0.1):
    """Generates a blue noise dither array of the given shape using the method
       proposed by Ulichney [1993] in "The void-and-cluster method for dither array
       generation" published in Proc. SPIE 1913.
      \param OutputShape The shape of the output array. This function works in
             arbitrary dimension, i.e. OutputShape can have arbitrary length. Though
             it is only tested for the 2D case where you should pass a tuple
             (Height,Width).
      \param StandardDeviation The standard deviation in pixels used for the
             Gaussian filter defining largest voids and tightest clusters. Larger
             values lead to more low-frequency content but better isotropy. Small
             values lead to more ordered patterns with less low-frequency content.
             Ulichney proposes to use a value of 1.5. If you want an anisotropic
             Gaussian, you can pass a tuple of length len(OutputShape) with one
             standard deviation per dimension.
      \param InitialSeedFraction The only non-deterministic step in the algorithm
             marks a small number of pixels in the grid randomly. This parameter
             defines the fraction of such points. It has to be positive but less
             than 0.5. Very small values lead to ordered patterns, beyond that there
             is little change.
      \return An integer array of shape OutputShape containing each integer from 0
              to np.prod(OutputShape)-1 exactly once."""
    nRank=np.prod(OutputShape)
    # Generate the initial binary pattern with a prescribed number of ones
    nInitialOne=max(1,min(int((nRank-1)/2),int(nRank*InitialSeedFraction)))
    # Start from white noise (this is the only randomized step)
    InitialBinaryPattern=np.zeros(OutputShape,dtype=bool)
    InitialBinaryPattern.flat=np.random.permutation(np.arange(nRank))<nInitialOne
    # Swap ones from tightest clusters to largest voids iteratively until convergence
    while(True):
        iTightestCluster=FindTightestCluster(InitialBinaryPattern,StandardDeviation)
        InitialBinaryPattern.flat[iTightestCluster]=False
        iLargestVoid=FindLargestVoid(InitialBinaryPattern,StandardDeviation)
        if(iLargestVoid==iTightestCluster):
            InitialBinaryPattern.flat[iTightestCluster]=True
            # Nothing has changed, so we have converged
            break;
        else:
            InitialBinaryPattern.flat[iLargestVoid]=True
    # Rank all pixels
    DitherArray=np.zeros(OutputShape,dtype=int)
    # Phase 1: Rank minority pixels in the initial binary pattern
    BinaryPattern=np.copy(InitialBinaryPattern)
    for Rank in range(nInitialOne-1,-1,-1):
        iTightestCluster=FindTightestCluster(BinaryPattern,StandardDeviation)
        BinaryPattern.flat[iTightestCluster]=False
        DitherArray.flat[iTightestCluster]=Rank
    # Phase 2: Rank the remainder of the first half of all pixels
    BinaryPattern=InitialBinaryPattern
    for Rank in range(nInitialOne,int((nRank+1)/2)):
        iLargestVoid=FindLargestVoid(BinaryPattern,StandardDeviation)
        BinaryPattern.flat[iLargestVoid]=True
        DitherArray.flat[iLargestVoid]=Rank
    # Phase 3: Rank the last half of pixels
    for Rank in range(int((nRank+1)/2),nRank):
        iTightestCluster=FindTightestCluster(BinaryPattern,StandardDeviation)
        BinaryPattern.flat[iTightestCluster]=True
        DitherArray.flat[iTightestCluster]=Rank
    return DitherArray

def generate_blue_noise_type_2_pattern(shape, std=1.5, initial_seed_fraction=0.1):
    """
    Generate blue noise pattern according to the void-and-cluster method proposed by Ulichney [1993] in "The void-and-cluster method for dither array generation" published in Proc. SPIE 1913.

    Args:
        shape (tuple): size of the pattern
        std (float): standard deviation in pixels used for the Gaussian filter
        initial_seed_fraction (float): Initial fraction of marked pixels in the grid. Has to be less than 0.5.
                                         Very small values lead to ordered patterns
    Returns:
        numpy.ndarray: float blue noise pattern
    """
    texture=GetVoidAndClusterBlueNoise(shape,std, initial_seed_fraction)
    pattern = (texture/np.max(texture)) # Float value between 0 and 1

    return pattern

def load_custom_pattern(shape,pattern_path):
    """
    Load custom pattern from h5 file. If the pattern is not the same size as the coded aperture, crop from the center of the loaded pattern.

    Args:
        shape (tuple): size of the pattern
        pattern_path (str): path to the h5 file containing the pattern

    Returns:
        numpy.ndarray: float blue noise pattern

    """
    size_y, size_x = shape[0], shape[1]

    if pattern_path is None:
        raise ValueError("Please provide h5 file path for custom pattern.")
    else:
        with h5py.File(pattern_path, 'r') as f:
            pattern = f['pattern'][:]

        coded_aperture_sampling_y = size_y
        coded_aperture_sampling_x = size_x

        if pattern.shape[0] != coded_aperture_sampling_y or pattern.shape[1] != coded_aperture_sampling_x:
            # Find center point of the pattern
            center_y, center_x = pattern.shape[0] // 2, pattern.shape[1] // 2

            # Determine starting and ending indices for the crop
            start_y = center_y - coded_aperture_sampling_y // 2
            end_y = start_y + coded_aperture_sampling_y
            start_x = center_x - coded_aperture_sampling_x // 2
            end_x = start_x + coded_aperture_sampling_x

            # Crop the pattern
            pattern = pattern[start_y:end_y, start_x:end_x]

            # Confirm the pattern is the correct shape
            if pattern.shape[0] != coded_aperture_sampling_y or pattern.shape[1] != coded_aperture_sampling_x:
                raise ValueError("Error cropping the pattern, its shape does not match the coded aperture sampling.")
        return pattern
def load_custom_list_of_patterns(shape,patterns_path):
    """
    Load custom list of patterns from h5 file. If the patterns are not the same size as the coded aperture, crop from the center of the loaded patterns.

    Args:
        shape (tuple): size of the pattern
        patterns_path (str): path to the h5 file containing the patterns

    Returns:
        list: list of patterns
    """


    list_of_coded_aperture_masks = []
    size_y, size_x = shape[0], shape[1]

    if patterns_path is None:
        raise ValueError("Please provide h5 file path for custom mask.")
    else:
        with h5py.File(patterns_path, 'r') as f:
            list_of_masks = f['list_of_masks'][:]

        for mask in list_of_masks:


            if mask.shape[0] != size_y or mask.shape[1] != size_x:
                # Find center point of the mask
                center_y, center_x = mask.shape[0] // 2, mask.shape[1] // 2

                # Determine starting and ending indices for the crop
                start_y = center_y - size_y // 2
                end_y = start_y + size_y
                start_x = center_x - size_x // 2
                end_x = start_x + size_x

                # Crop the mask
                mask = mask[start_y:end_y, start_x:end_x]

                # Confirm the mask is the correct shape
                if mask.shape[0] != size_y or mask.shape[1] != size_x:
                    raise ValueError("Error cropping the mask, its shape does not match the coded aperture sampling.")
            list_of_coded_aperture_masks.append(mask)

        return list_of_coded_aperture_masks