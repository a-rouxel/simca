import numpy as np
import matplotlib.pyplot as plt
def supergaussian_2D(x, y, center,sigma_x, sigma_y, order=2):
    """
    Calculate the value of a 2D supergaussian given x, y, sigma_x, sigma_y, and order.
    """
    gx = np.exp(-np.abs(x-center[0])**order / (2 * sigma_x**order))
    gy = np.exp(-np.abs(y-center[1])**order / (2 * sigma_y**order))
    return gx * gy

def generate_2D_supergaussian(center, height, aspect_ratio, order,pixel_size,sampling):
    """
    Generate a 2D supergaussian using x_array_out and given sigma_x, sigma_y, and order.
    """
    x = np.arange(-sampling[0]/2, sampling[0]/2) * pixel_size
    y = np.arange(-sampling[1]/2, sampling[1]/2) * pixel_size
    sigma_x = height*aspect_ratio
    sigma_y = height

    X, Y = np.meshgrid(x, y)
    Z = supergaussian_2D(X, Y, center,sigma_x/2, sigma_y/2, order)
    return Z


# height = 475.6 * 10 ** -6
# aspect_ratio = 1.497
# order = 10
#
# center = (0, 0)
#
# pixel_size = 15 * 10 ** -6
# sampling_x, sampling_y = (320, 256)
#
# SG = generate_2D_supergaussian(center, height, aspect_ratio, order, pixel_size, (sampling_x, sampling_y))
#
# plt.imshow(SG,extent=[-sampling_x*pixel_size/2,sampling_x*pixel_size/2,sampling_y*pixel_size/2,-sampling_y*pixel_size/2])
# # plt.xlim(-sampling*pixel_size/2,sampling*pixel_size/2)
# # plt.ylim(sampling*pixel_size/2,-sampling*pixel_size/2)
# plt.show()