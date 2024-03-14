import torch
from matplotlib import pyplot as plt
import math

def evaluate_slit_scanning_straightness(filtering_cube, device, sigma = 0.75, pos_slit=0.5):
    """
    Evaluate the straightness of the slit scanning.
    working cost function for up to focal >100000
    """

    pos_cube = round(filtering_cube.shape[1]*pos_slit)
    cost_value = torch.tensor(0.0, requires_grad=True).to(device)

    gaussian = (torch.arange(filtering_cube.shape[1]) - pos_cube).to(device)
    gaussian = torch.exp(-torch.square(gaussian)/(2*sigma**2)).unsqueeze(0).to(device)
    gaussian = gaussian
    w = filtering_cube.shape[2]//2

    # Minimize the smile distorsion at the central wavelength
    row_diffs = filtering_cube[1:, :, w] - filtering_cube[:-1, :, w]
    cost_value = cost_value - torch.sum((torch.abs(filtering_cube[:, :, w] - gaussian)+1e-8)**0.4) + 0.6*torch.sum(filtering_cube[:, pos_cube,w]) - 0.8*torch.sum(torch.sum(torch.abs(row_diffs))) #- 2*torch.sum(torch.var(filtering_cube[:, :, 0], dim=0)) 

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
    cost_value = 2*torch.mean(acquisition) - 8*torch.std(acquisition)

    return -cost_value

def evaluate_max_lighting(widths, acquisition, target):
    cost_value = 0

    acq = acquisition[acquisition>100].flatten()    
    
    # Squared loss on the left, log-barrier on the right
    def bowl(scene, target_, saturation=None):
        if saturation is None:
            saturation = target_*1.2
        s = saturation
        t = target_
        cost = 0
        for x in scene:
            if x <= t:
                cost += ((t-x)/t)**2
            elif x < s:
                A = -1/((s-t)**2) + 2/(t**2)
                B = -1/(s-t) + t/((s-t)**2) - 2/t
                C = - 1/2*A*t - B*t
                cost += - math.log((s-x)/(s-t)) + 1/2*A*x**2 + B*x + C
            else:
                cost += 1e18*x**2
        return cost
    
    # Squared loss on the left, inverse on the right
    def bowl_inverse(scene, target_, saturation=None):
        if saturation is None:
            saturation = target_*1.2
        s = saturation
        t = target_
        cost = 0
        for x in scene:
            if x <= t:
                cost += ((t-x)/t)**2
            elif x < s:
                A = -2/((s-t)**3) + 2/(t**2)
                B = -1/((s-t)**2) - A*t
                C = -1/(s-t) - 1/2*A*t - B*t
                cost += - 1/(s-x) + 1/2*A*x**2 + B*x + C
            else:
                cost += 1e18*x**2
        return cost

    print("Var: ", torch.var(acq))
    print("Min: ", torch.min(acq))
    print("Mean: ", torch.mean(acq))
    print("Max: ", torch.max(acq))

    cost_value = - bowl(acq, target, saturation=2.2e6)
    print("Cost: ", - cost_value)
    return - cost_value  
