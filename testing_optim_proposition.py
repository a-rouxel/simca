from simca import  load_yaml_config
import matplotlib.pyplot as plt
from simca.CassiSystemOptim import CassiSystemOptim
from simca.cost_functions_optics import *
from opticalglass.glassfactory import get_glass_catalog
import pandas as pd


config_dataset = load_yaml_config("simca/configs/dataset.yml")
config_system = load_yaml_config("simca/configs/cassi_system_optim_optics_full_triplet.yml")
config_patterns = load_yaml_config("simca/configs/pattern.yml")
config_acquisition = load_yaml_config("simca/configs/acquisition.yml")

hoya_pd = get_glass_catalog('Schott')

# Access the 'nd' values where the first level is NaN
idx = pd.IndexSlice

nd_values = hoya_pd.df.loc[:, idx[pd.NA, 'nd']]
vd_values = hoya_pd.df.loc[:, idx["abbe number", 'vd']]

if __name__ == '__main__':

    # Initialize the CASSI system
    cassi_system = CassiSystemOptim(system_config=config_system)
    #
    # config_system["system architecture"]["dispersive element"]["glass1"] = "N-BK7"
    # # config_system["system architecture"]["dispersive element"]["glass2"] = "N-SF11"
    # config_system["system architecture"]["dispersive element"]["glass3"] = "N-BK7"
    # config_system["system architecture"]["dispersive element"]["A1"] = 70
    # # config_system["system architecture"]["dispersive element"]["A2"] = 60
    # config_system["system architecture"]["dispersive element"]["A3"] = 70
    # config_system["system architecture"]["dispersive element"]["alpha_c"] = 55


    cassi_system.update_optical_model(system_config=config_system)

    x_vec_out, y_vec_out = cassi_system.propagate_coded_aperture_grid()

    #
    alpha_c = cassi_system.optical_model.alpha_c
    alpha_c_out = cassi_system.optical_model.alpha_c_transmis
    list_apex_angles = [cassi_system.optical_model.A1, cassi_system.optical_model.A2, cassi_system.optical_model.A3]
    list_theta_in = cassi_system.optical_model.list_theta_in
    list_theta_out = cassi_system.optical_model.list_theta_out

    # evaluate spectral dispersion
    dispersion, central_coordinates_X = evaluate_spectral_dispersion_values(x_vec_out, y_vec_out)
    print("dispersion in microns", dispersion)

    # evaluate direct_view
    deviation = evaluate_direct_view(alpha_c,alpha_c_out,list_apex_angles)
    print("deviation in degrees", deviation*180/np.pi)

    # evaluate beam compression
    beam_compression = evaluate_beam_compression(list_theta_in, list_theta_out)
    print("beam compression", beam_compression)

    #evaluate thickness
    thickness = evaluate_thickness(list_apex_angles)
    print("thickness", thickness)

    # evaluate distortions
    x_vec_no_dist, y_vec_no_dist = get_cassi_system_no_spectial_distorsions(cassi_system.X_coded_aper_coordinates,
                                                                                     cassi_system.Y_coded_aper_coordinates,
                                                                                     central_coordinates_X)

    # print(x_vec_no_dist)

    print(x_vec_no_dist.shape, y_vec_no_dist.shape)
    distortion_metric = evaluate_distortions(x_vec_out, y_vec_out, x_vec_no_dist, y_vec_no_dist)

    print("distortion metric", distortion_metric)

    plt.plot(cassi_system.wavelengths.detach().numpy(), central_coordinates_X.detach().numpy())
    plt.show()
    #
    # y_second_derivative = calculate_second_derivative(cassi_system.wavelengths, central_coordinates_X)
    #
    # plt.plot(y_second_derivative.detach().numpy())
    # plt.show()
    print(x_vec_out.detach().numpy().shape)

    plt.scatter(x_vec_out.detach().numpy()[...,0], y_vec_out.detach().numpy()[...,0],color='blue',label="distor")
    plt.scatter(x_vec_out.detach().numpy()[...,-1], y_vec_out.detach().numpy()[...,-1],color='red',label="distor")
    #
    plt.scatter(x_vec_no_dist.detach().numpy()[...,0], y_vec_no_dist.detach().numpy()[...,0],label="no dist",color='green')
    plt.scatter(x_vec_no_dist.detach().numpy()[...,-1], y_vec_no_dist.detach().numpy()[...,-1],label="no dist",color='green')

    plt.legend()

    plt.show()


    # distance from closest point
    current_value = [cassi_system.optical_model.nd1, cassi_system.optical_model.vd1]
    distance_closest_point = evaluate_distance(current_value,nd_values, vd_values)










    # print(cassi_system.optical_model.list_theta_in, cassi_system.optical_model.list_theta_out)
    #
    # dispersion,central_coordinates_X = evaluate_spectral_dispersion_values(x_vec_out, y_vec_out)
    # r2, y_pred = evaluate_linearity(central_coordinates_X)
    #
    # y_second_derivative = calculate_second_derivative(cassi_system.wavelengths,central_coordinates_X)
    #
    # plt.plot(y_second_derivative)
    # plt.show()
    #
    # #no distorsions X_vec_out, Y_vec_out
    # x_vec_out_no_dist, y_vec_out_no_dist = get_cassi_system_no_spectial_distorsions(cassi_system.X_coded_aper_coordinates,
    #                                                                                 cassi_system.Y_coded_aper_coordinates,
    #                                                                                 central_coordinates_X)
    #
    # print(r2)
    #
    # plt.plot(central_coordinates_X)
    # plt.plot(y_pred)
    # plt.show()
    #
    # plt.scatter(x_vec_out[...,0], y_vec_out[...,0],color='blue')
    # plt.scatter(x_vec_out[:,:,-1], y_vec_out[:,:,-1],color='red')
    # plt.scatter(x_vec_out_no_dist[...,0], y_vec_out_no_dist[...,0])
    # plt.scatter(x_vec_out_no_dist[:,:,-1], y_vec_out_no_dist[:,:,-1])
    #
    # plt.show()