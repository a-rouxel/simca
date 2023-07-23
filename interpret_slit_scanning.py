import h5py
import matplotlib.pyplot as plt
import numpy as np

nb_of_acq = 91
data_path = "./data/results/slit_scanning_high_res/"

wavelengths_file = h5py.File(data_path + f"wavelengths" + ".h5", "r")
wavelength = np.array(wavelengths_file["wavelengths"])

interpolated_scene = h5py.File(data_path + f"interpolated_scene" + ".h5", "r")
interpolated_scene = np.array(interpolated_scene["interpolated_scene"])

list_of_3D_meas_cube = list()

plt.imshow(np.sum(interpolated_scene,axis=2))
plt.show()

for i in range(nb_of_acq):

    print(f"Loading measurement {i}")

    measurement_file = h5py.File(data_path + f"measurement_{i}" + ".h5", "r")
    filtering_cube_file = h5py.File(data_path + f"filtering_cube_{i}" + ".h5", "r")

    # Load the datasets into numpy arrays
    # 2D array
    meas = np.array(measurement_file["measurement"])

    # 3D array
    filt = np.array(filtering_cube_file["filtering_cube"])

    recons_cube = meas[:, :, np.newaxis] * filt

    list_of_3D_meas_cube.append(recons_cube)

final_cube = np.zeros_like(list_of_3D_meas_cube[0])

for cube in list_of_3D_meas_cube:
    final_cube += cube

print(final_cube.shape)


fluo_spectrum = np.load("./fluo_intensity_val.npy")

fluo_spectrum *=   np.max(final_cube) / np.max(fluo_spectrum)
interpolated_scene *= np.max(final_cube) / np.max(interpolated_scene)

wavelengths = np.load("./wavelengths.npy")

plt.scatter(wavelength,final_cube[55, 101, :],label="pixel 1")
plt.scatter(wavelength,final_cube[55, 102, :],label="pixel 2")
plt.scatter(wavelength,final_cube[55, 103, :],label="pixel 3")
plt.scatter(wavelength,final_cube[55, 104, :],label="pixel 4")
plt.plot(wavelengths,fluo_spectrum)
plt.plot(wavelength,interpolated_scene[55, 102, :],label="pixel 1 from interpolated scene")
plt.legend()
plt.show()