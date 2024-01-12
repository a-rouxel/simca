import torch

def evalute_slit_scanning_straightness(filtering_cube,threshold):
    """
    Evaluate the straightness of the slit scanning.
    working cost functio for up for focal >100000
    """

    epsilon = 1e-6  # Small constant for numerical stability
    cost_value = torch.tensor(0.0, requires_grad=True)

    for i in range(filtering_cube.shape[2]):
        vertical_binning = torch.sum(filtering_cube[:, :, i], axis=0)
        max_value = torch.max(vertical_binning)
        std_deviation = torch.std(vertical_binning)
        # Reward the max value and penalize based on the standard deviation

        # Calculate the differences between consecutive rows (vectorized)
        row_diffs = filtering_cube[1:, :, i] - filtering_cube[:-1, :, i]
        cost_value = cost_value + max_value / std_deviation - torch.sum(torch.sum(torch.abs(row_diffs)))

    # Minimizing the negative of cost_value to maximize the original objective
    return -cost_value

# def evalute_slit_scanning_straightness(filtering_cube,threshold):
#     """
#     Evaluate the straightness of the slit scanning.
#     working cost functio for up for focal >100000
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