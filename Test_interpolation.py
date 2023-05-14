import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
import time

# Create some random unstructured data
np.random.seed(0)  # for reproducibility
npts = 400*500
x = np.random.uniform(-2, 2, npts)
y = np.random.uniform(-2, 2, npts)

# Create multiple z arrays, one for each wavelength
# (for this example, we'll just use the same function for each wavelength)
wavelengths = [x for x in range(400, 700, 5)]
print(len(wavelengths))
z_values = [x * np.exp(-x**2 - y**2) for _ in wavelengths]

# Define the structured grid
xi = np.linspace(-2.0, 2.0, 600)
yi = np.linspace(-2.0, 2.0, 500)
xi, yi = np.meshgrid(xi, yi)

# Define a worker function that will do the interpolation for one wavelength
def worker(z):
    return griddata((x, y), z, (xi, yi), method='cubic')

# Use a multiprocessing Pool to do the interpolations in parallel
t_start = time.time()
print(mp.cpu_count())
with Pool(mp.cpu_count()) as p:
    zi_values = p.map(worker, z_values)
t_end = time.time()

print('Interpolation took {:.2f} seconds'.format(t_end - t_start))

# Plot the results
# for i, zi in enumerate(zi_values):
#     plt.figure(figsize=(6,6))
#     plt.imshow(zi, extent=(-2, 2, -2, 2), origin='lower')
#     plt.colorbar()
#     plt.title('Interpolated to Structured Grid for wavelength {}'.format(wavelengths[i]))
#     plt.show()
