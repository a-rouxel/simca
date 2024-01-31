import torch
from matplotlib import pyplot as plt

def evaluate_slit_scanning_straightness(filtering_cube, sigma = 0.75, pos_slit=0.5):
    """
    Evaluate the straightness of the slit scanning.
    working cost function for up to focal >100000
    """

    pos_cube = round(filtering_cube.shape[1]*pos_slit)
    cost_value = torch.tensor(0.0, requires_grad=True)

    gaussian = torch.arange(filtering_cube.shape[1]) - pos_cube
    gaussian = torch.exp(-torch.square(gaussian)/(2*sigma**2)).unsqueeze(0)
    for i in range(filtering_cube.shape[2]):
        """ vertical_binning = torch.sum(filtering_cube[:, :, i], axis=0)
        #max_value = torch.max(vertical_binning)
        std_deviation_vertical = torch.std(vertical_binning)
        #std_deviation_horizontal = torch.std(torch.sum(filtering_cube[:,:,i], axis=1))
        # Reward the max value and penalize based on the standard deviation

        # Calculate the differences between consecutive rows (vectorized)
        row_diffs = filtering_cube[1:, :, i] - filtering_cube[:-1, :, i]
        #cost_value = cost_value + max_value / std_deviation - torch.sum(torch.sum(torch.abs(row_diffs)))
        cost_value = cost_value + std_deviation_vertical  - 0.2*torch.sum(torch.sum(torch.square(row_diffs))) #- 0.2*torch.sum(torch.sum(torch.abs(row_diffs)))"""
        row_diffs = filtering_cube[1:, :, 0] - filtering_cube[:-1, :, 0]

        #cost_value = cost_value - torch.sum((torch.square(filtering_cube[:, :, 0] - gaussian)+1e-8)) - 5*torch.sum(torch.var(filtering_cube[:, :, 0], dim=0)) - 0.5*torch.sum(torch.sum(torch.abs(row_diffs)))
        cost_value = cost_value - (2**2)*torch.sum((torch.sqrt(1+((filtering_cube[:, :, 0] - gaussian)/2)**2)-1))
    # Minimizing the negative of cost_value to maximize the original objective
    return -cost_value

def evaluate_center(acquisition):
    """
    Evaluate the value of the central pixel of each line
    """
    cost_value = torch.sum(acquisition[:, acquisition.shape[1]//2], axis=0)
    
    return -cost_value

def evaluate_mean_lighting(acquisition):
    """
    Evaluate the mean and std values of the acquisition, maximizing mean and minimizing std
    """
    #cost_value = torch.mean(acquisition)/(torch.std(acquisition)+1e-9)
    cost_value = 2*torch.mean(acquisition) - 8*torch.std(acquisition)

    return -cost_value

def evaluate_max_lighting(acquisition, pos_check):
    cost_value = 0

    col = acquisition[:, pos_check]

    cost_value = 2*torch.mean(col)**2 - 8*torch.var(col)

    return -cost_value

# def evalute_slit_scanning_straightness(filtering_cube,threshold):
#     """
#     Evaluate the straightness of the slit scanning.
#     working cost function for up to focal >100000
#     """
#     cost_value = torch.tensor(0.0, requires_grad=True)

#     for i in range(filtering_cube.shape[2]):
#         vertical_binning = torch.sum(filtering_cube[:, :, i], axis=0)
#         max_value = torch.max(vertical_binning)
#         values_above_threshold = vertical_binning > threshold
#         number_of_values_above_threshold = torch.sum(values_above_threshold)
#         cost_value = cost_value + max_value / number_of_values_above_threshold
#     return -cost_value


# def evalute_slit_scanning_straightness(filtering_cube,threshold):
#     """
#     Evaluate the straightness of the slit scanning.
#     """
#     cost_value  =0

#     for i in range(filtering_cube.shape[2]):
#         vertical_binning = torch.sum(filtering_cube[:,:,i],axis=0)
#         max_value = torch.max(vertical_binning)
#         # Differentiable way to count values above threshold
#         values_above_threshold = vertical_binning > threshold
#         number_of_values_above_threshold = torch.sum(values_above_threshold)
#         cost_value += 1 / (number_of_values_above_threshold ** 2)

#     cost_value = -1*(cost_value)
#     return cost_value