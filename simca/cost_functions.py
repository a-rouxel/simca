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
    """for i in range(filtering_cube.shape[2]):
         vertical_binning = torch.sum(filtering_cube[:, :, i], axis=0)
        #max_value = torch.max(vertical_binning)
        std_deviation_vertical = torch.std(vertical_binning)
        #std_deviation_horizontal = torch.std(torch.sum(filtering_cube[:,:,i], axis=1))
        # Reward the max value and penalize based on the standard deviation

        # Calculate the differences between consecutive rows (vectorized)
        row_diffs = filtering_cube[1:, :, i] - filtering_cube[:-1, :, i]
        #cost_value = cost_value + max_value / std_deviation - torch.sum(torch.sum(torch.abs(row_diffs)))
        cost_value = cost_value + std_deviation_vertical  - 0.2*torch.sum(torch.sum(torch.square(row_diffs))) #- 0.2*torch.sum(torch.sum(torch.abs(row_diffs)))"""
    row_diffs = filtering_cube[1:, :, w] - filtering_cube[:-1, :, w]
    cost_value = cost_value - torch.sum((torch.abs(filtering_cube[:, :, w] - gaussian)+1e-8)**0.4) + 0.6*torch.sum(filtering_cube[:, pos_cube,w]) - 0.8*torch.sum(torch.sum(torch.abs(row_diffs))) #- 2*torch.sum(torch.var(filtering_cube[:, :, 0], dim=0)) 
        #delta = 2
        #cost_value = cost_value - (delta**2)*torch.sum((torch.sqrt(1+((filtering_cube[:, :, 0] - gaussian)/delta)**2)-1)) # pseudo-huber loss
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

def evaluate_max_lighting(widths, acquisition, target):
    cost_value = 0

    #col = acquisition[:, pos_cube]
    #col = acquisition[:, pos_cube-2:pos_cube+2]
    col = acquisition[acquisition>100].flatten()
    #col = acquisition[93:208, 30].unsqueeze(1)
    """for i in range(1, 5):
        col = torch.cat((col, acquisition[93:208, 30+i*15].unsqueeze(1)), 1) """

    #cost_value = 2*torch.mean(col)**2 - 25*torch.var(col)
    #cost_value = - torch.var(col)
    """ cost_value = 15*torch.mean(col)**2 - 25*torch.var(col)
    cost_value = 8000*torch.mean(col)**2 - torch.sum((col-10000)**2)
    cost_value = 0.75*torch.mean(col) - torch.mean(torch.abs(col-6000)) 
    cost_value = torch.mean(col) - 2*torch.std(col)

    lines = torch.mean(acquisition, axis=1)

    #cost_value = - torch.var(torch.log(col))

    cost_value = - torch.var((torch.log(col)- torch.log(torch.tensor([40000])))**2)
    cost_value = - torch.var(torch.log(col)) - torch.log(torch.var(col))# - torch.mean((torch.log(col)- torch.log(torch.tensor([14000])))**2)
    #cost_value = - torch.var(torch.log(col)**2) - torch.var((torch.log(col)- torch.log(torch.tensor([20000])))**2)
    cost_value = - torch.var(torch.log(col)**2) - 2*torch.var((torch.log(col)- torch.log(torch.tensor([6000])))**2)
    #cost_value = - torch.sum((2000*10000*((col-2000)+(col-10000)) - (10000-2000))/2)
    #cost_value = -torch.var(torch.log(col)) """
    #cost_value = -torch.var(torch.exp(col/11000))
    row_diffs = torch.abs(widths[0,1:] - widths[0,:-1])
    #cost_value = - torch.var(torch.exp(col/18100)) - torch.sum(-torch.log(1+row_diffs))
    #print(torch.var(torch.exp(col/20000)))
    #print(torch.mean(torch.log(col)))
    #print(torch.sum(-torch.log(1+row_diffs)))
    

    def saturation(scene, target_, margin=0.05):
        cost = 0
        for elem in scene:
            if elem <= target_*(1+margin):
                cost += (target_-elem)
            else:
                cost += (elem-target_)**3
        return cost
    
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
                #cost += - math.exp(80*(1-(saturation-elem)/(saturation-target_)))
                #cost += (elem-target_)**4
            else:
                cost += 1e18*x**2
        return cost
    
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
                #cost += - math.exp(80*(1-(saturation-elem)/(saturation-target_)))
                #cost += (elem-target_)**4
            else:
                cost += 1e18*x**2
        return cost


    #cost_value = - torch.var(col) - torch.sum(-torch.log(1+row_diffs))
    #cost_value = - torch.var(torch.exp(col/18000)) - 10000*torch.sum(torch.log(1/(1+row_diffs)))
    #cost_value = - torch.var(torch.exp(col/30000))# - 2*torch.count_nonzero(row_diffs)
    #cost_value = - torch.var(torch.exp(col/20000)) - 40*torch.count_nonzero(row_diffs)
    #cost_value = torch.mean(col)
    #cost_value = - 2*torch.var((torch.exp(col/20000)- torch.exp(torch.tensor([9000])/20000))**2)
    #cost_value = - saturation(col, 45000, margin=0.1)

    #print("Jumps: ", torch.count_nonzero(row_diffs))
    print("Var: ", torch.var(col))
    #print("Saturation: ", - saturation(col.flatten(), 120000, margin=0.1))
    print("Min: ", torch.min(col))
    print("Mean: ", torch.mean(col))
    print("Max: ", torch.max(col))

    #cost_value = - torch.var(torch.exp(col/100000)) #- 1e-5*saturation(col.flatten(), 200000, margin=0.1)
    #cost_value = - torch.var((torch.log(col.squeeze())- torch.log(torch.tensor([200000]).to('cuda')))**2)
    #cost_value = - torch.var(col) #- saturation(col.flatten(), 120000, margin=0.1)
    cost_value = - bowl(col, target, saturation=2.2e6)
    print("Cost: ", - cost_value)
    return - cost_value  

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