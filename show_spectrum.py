import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

file_binary = "./metrics-results/recons_cube_learned_mask.pt"
file_float = "./metrics-results/recons_cube_learned_mask_float.pt"
file_random = "./metrics-results/recons_cube_random.pt"
gt = "./metrics-results/gt_cube.pt"

binary_cube = torch.load(file_binary, map_location=torch.device('cpu')).cpu()[0,...]
float_cube = torch.load(file_float, map_location=torch.device('cpu')).cpu()[0,...]
random_cube = torch.load(file_random, map_location=torch.device('cpu')).cpu()[0,...]
gt_cube = torch.load(gt, map_location=torch.device('cpu')).cpu()[0,...]

panchro = torch.sum(gt_cube, dim=2)

pixel_i_1 = 104
pixel_j_1 = 105

pixel_i_2 = 43
pixel_j_2 = 80

pixel_i_3 = 45
pixel_j_3 = 2

# Slice at 554nm
slice = binary_cube[:,:,14]

# Cube plot
fig, ax = plt.subplots()
ax.imshow(slice.numpy(), cmap='gray')
rect1 = patches.Rectangle((pixel_j_1, pixel_i_1), 2, 2, linewidth=1, edgecolor='r', facecolor='none')
rect2 = patches.Rectangle((pixel_j_2, pixel_i_2), 2, 2, linewidth=1, edgecolor='r', facecolor='none')
rect3 = patches.Rectangle((pixel_j_3, pixel_i_3), 2, 2, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
plt.show()
fig.savefig('./metrics-results/reconstructed_cube.svg', format='svg')

# Spectra plot
fig, ax = plt.subplots()
ax.plot(np.linspace(450,650,28), gt_cube[pixel_i_1, pixel_j_1].cpu().numpy(), 'k--', label='truth')
ax.plot(np.linspace(450,650,28), binary_cube[pixel_i_1, pixel_j_1].cpu().numpy(), 'r', label='binary')
ax.plot(np.linspace(450,650,28), float_cube[pixel_i_1, pixel_j_1].cpu().numpy(), 'b', label = 'float')
ax.plot(np.linspace(450,650,28), random_cube[pixel_i_1, pixel_j_1].cpu().numpy(), 'g', label = 'random')

ax.plot(np.linspace(450,650,28), gt_cube[pixel_i_2, pixel_j_2].cpu().numpy(), 'k--')
ax.plot(np.linspace(450,650,28), binary_cube[pixel_i_2, pixel_j_2].cpu().numpy(), 'r')
ax.plot(np.linspace(450,650,28), float_cube[pixel_i_2, pixel_j_2].cpu().numpy(), 'b')
ax.plot(np.linspace(450,650,28), random_cube[pixel_i_2, pixel_j_2].cpu().numpy(), 'g')

ax.plot(np.linspace(450,650,28), gt_cube[pixel_i_3, pixel_j_3].cpu().numpy(), 'k--')
ax.plot(np.linspace(450,650,28), binary_cube[pixel_i_3, pixel_j_3].cpu().numpy(), 'r')
ax.plot(np.linspace(450,650,28), float_cube[pixel_i_3, pixel_j_3].cpu().numpy(), 'b')
ax.plot(np.linspace(450,650,28), random_cube[pixel_i_3, pixel_j_3].cpu().numpy(), 'g')
ax.legend()
plt.show()

fig.savefig('./metrics-results/spectra.svg', format='svg')