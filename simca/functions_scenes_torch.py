import torch
import snoop

def interpolate_data_along_wavelength_torch(data, current_sampling, new_sampling, chunk_size=50):
    """Interpolate the input 3D data along a new sampling in the third axis.

    Args:
        data (numpy.ndarray): 3D data to interpolate
        current_sampling (numpy.ndarray): current sampling for the 3rd axis
        new_sampling (numpy.ndarray): new sampling for the 3rd axis
        chunk_size (int): size of the chunks to use for the interpolation
    """

    # Generate the coordinates for the original grid
    x = torch.arange(data.shape[0]).float()
    y = torch.arange(data.shape[1]).float()
    z = current_sampling

    # Initialize an empty array for the result
    interpolated_data = torch.empty((data.shape[0], data.shape[1], len(new_sampling)))

    # Perform the interpolation in chunks
    for i in range(0, data.shape[0], chunk_size):
        for j in range(0, data.shape[1], chunk_size):
            new_coordinates = torch.meshgrid(x[i:i+chunk_size], y[j:j+chunk_size], new_sampling, indexing='ij')
            new_grid = torch.stack(new_coordinates, axis=-1)

            min_bound = torch.tensor([torch.min(x[i:i+chunk_size]), torch.min(y[j:j+chunk_size]), torch.min(z)])
            max_bound = torch.tensor([torch.max(x[i:i+chunk_size]), torch.max(y[j:j+chunk_size]), torch.max(z)])

            new_grid = (2*((new_grid-min_bound)/(max_bound - min_bound))-1).unsqueeze(0).flip(-1).float() # Normalize between -1 and 1

            """new_coordinates = np.meshgrid(x[i:i+chunk_size], y[j:j+chunk_size], new_sampling, indexing='ij')
            new_grid = np.stack(new_coordinates, axis=-1)
            
            new_grid = torch.from_numpy(2*((new_grid-min_bound)/(max_bound - min_bound))-1).unsqueeze(0).flip(-1).double()"""
            interpolated_data[i:i+chunk_size, j:j+chunk_size, :] = torch.nn.functional.grid_sample(input = data[i:i+chunk_size, j:j+chunk_size, :][None, None, ...],
                                                                                                   grid = new_grid,
                                                                                                   padding_mode = "zeros",
                                                                                                   mode = "bilinear",
                                                                                                   align_corners = True).squeeze(0,1)

    return interpolated_data