import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils.functions_reconstruction import generate_spectral_angular_map_optimized

nb_of_acq = 239
data_path = "./data/results/slit_scanning_high_res_prism/"

wavelengths_file = h5py.File(data_path + f"wavelengths" + ".h5", "r")
wavelength = np.array(wavelengths_file["wavelengths"])

interpolated_scene = h5py.File(data_path + f"interpolated_scene" + ".h5", "r")
interpolated_scene = np.array(interpolated_scene["interpolated_scene"])

list_of_3D_meas_cube = list()

filtering_cube = np.zeros_like(interpolated_scene)


for i in range(nb_of_acq):

    print(f"Loading measurement {i}")

    measurement_file = h5py.File(data_path + f"measurement_{i}" + ".h5", "r")
    filtering_cube_file = h5py.File(data_path + f"filtering_cube_{i}" + ".h5", "r")

    # Load the datasets into numpy arraysqq
    # 2D array
    meas = np.array(measurement_file["measurement"])

    # 3D array
    filt = np.array(filtering_cube_file["filtering_cube"])
    panchro_filtering_cube = np.sum(filt,axis=2)
    panchro_filtering_cube = np.nan_to_num(panchro_filtering_cube)

    # plt.imshow(meas)
    # plt.show()

    # plt.imshow(meas)
    # plt.colorbar()
    #
    # plt.title("meas_before")
    # plt.show()

    panchro_filtering_cube[panchro_filtering_cube<1] = 1

    # plt.imshow(panchro_filtering_cube)
    # plt.show()


    meas = meas / panchro_filtering_cube

    # plt.imshow(meas)
    # plt.colorbar()
    # plt.title("meas_after")
    # plt.show()

    # calcul_1 = np.nan_to_num(filt[0, :, 0:100])
    # print(f"energy of the first part of the filtering cube {i}",np.sum(np.sum(calcul_1)))
    # calcul_2 = np.nan_to_num(filt[0, :, 100:201])
    # print(f"energy of the second part of the filtering cube {i}", np.sum(np.sum(calcul_2)))
    filtering_cube += filt

    recons_cube = meas[:, :, np.newaxis] * filt

    list_of_3D_meas_cube.append(recons_cube)



final_cube = np.zeros_like(list_of_3D_meas_cube[0])

for cube in list_of_3D_meas_cube:
    final_cube += cube

# plt.imshow(filtering_cube[0,:,:])
# plt.show()
# plt.imshow(filtering_cube[0,:,60])
# plt.show()
# plt.imshow(filtering_cube[:,:,120])
# plt.show()
# plt.imshow(filtering_cube[:,:,180])
# plt.show()
# plt.imshow(filtering_cube[:,:,198])
# plt.show()


final_cube = np.nan_to_num(final_cube)


fluo_spectrum = np.load("./fluo_intensity_val.npy")
sun_spectrum = np.load("./sun_spectrum.npy")
sun_wavelengths = np.load("./wavelengths_sun.npy")

fluo_spectrum *=   np.max(final_cube) / np.max(fluo_spectrum)
sun_spectrum *= np.max(final_cube[0,0,:]) / np.max(sun_spectrum)
print(np.sum(interpolated_scene,axis=2))

interpolated_scene *= np.max(final_cube) / np.max(interpolated_scene)



mask = (wavelength >= 450) & (wavelength <= 600)
mask_ = (sun_wavelengths >= 450) & (sun_wavelengths <= 600)

sun_wavelengths = sun_wavelengths[mask_]
wavelength = wavelength[mask]
interpolated_scene = interpolated_scene[:,:,mask]
final_cube = final_cube[:,:,mask]
sun_spectrum = sun_spectrum[mask_]





nb_row, nb_col, nb_wavelength = interpolated_scene.shape

point_1_idx = [24, 52]
point_2_idx = [24, 154]
point_3_idx = [101, 101]
point_4_idx = [55, 102]

fig, ax = plt.subplots(1, 2, figsize=(8, 15))

print(np.sum(interpolated_scene,axis=2).shape)
ax[0].imshow(np.sum(interpolated_scene,axis=2))

print(np.sum(interpolated_scene,axis=2))
ax[0].scatter(point_1_idx[1],point_1_idx[0],label="pixel 1",c="r")
ax[0].scatter(point_2_idx[1],point_2_idx[0],label="pixel 2",c="b")
ax[0].scatter(point_3_idx[1],point_3_idx[0],label="pixel 3",c="g")
ax[0].scatter(point_4_idx[1],point_4_idx[0],label="pixel 4",c="grey")

ax[1].scatter(wavelength,final_cube[point_1_idx[0], point_1_idx[1], :],label="pixel 1",c="r")
ax[1].scatter(wavelength,final_cube[point_2_idx[0], point_2_idx[1], :],label="pixel 2",c="b")
ax[1].scatter(wavelength,final_cube[point_3_idx[0], point_3_idx[1], :],label="pixel 3", c="g")
ax[1].scatter(wavelength,final_cube[point_4_idx[0], point_4_idx[1], :],label="pixel 4",c="grey")
ax[1].plot(wavelength,interpolated_scene[nb_row//2, nb_col//2, :],label="fluo")
ax[1].plot(sun_wavelengths,sun_spectrum,label="sun")
plt.legend()
plt.show()

sam = generate_spectral_angular_map_optimized(final_cube, interpolated_scene)

plt.imshow(sam,vmin=0,vmax=1)
plt.colorbar()
plt.show()