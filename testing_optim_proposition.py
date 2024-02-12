from simca import  load_yaml_config
import matplotlib.pyplot as plt
from simca.CassiSystemOptim import CassiSystemOptim
from simca.cost_functions_optics import evaluate_spectral_dispersion_values, evaluate_linearity,get_cassi_system_no_spectial_distorsions,calculate_second_derivative

config_dataset = load_yaml_config("simca/configs/dataset.yml")
config_system = load_yaml_config("simca/configs/cassi_system_optim_optics.yml")
config_patterns = load_yaml_config("simca/configs/pattern.yml")
config_acquisition = load_yaml_config("simca/configs/acquisition.yml")

if __name__ == '__main__':

    # Initialize the CASSI system
    cassi_system = CassiSystemOptim(system_config=config_system)

    config_system["system architecture"]["dispersive element"]["glass1"] = "N-PK52A"
    # config_system["system architecture"]["dispersive element"]["glass2"] = x["glass2"]
    # config_system["system architecture"]["dispersive element"]["glass3"] = x["glass1"]
    config_system["system architecture"]["dispersive element"]["A1"] = 63.04
    # config_system["system architecture"]["dispersive element"]["A2"] = x["A2"]
    # config_system["system architecture"]["dispersive element"]["A3"] = x["A1"]
    config_system["system architecture"]["dispersive element"]["alpha_c"] = 65.15

    cassi_system.update_optical_model(system_config=config_system)

    x_vec_out, y_vec_out = cassi_system.propagate_coded_aperture_grid()

    print(cassi_system.optical_model.list_theta_in, cassi_system.optical_model.list_theta_out)

    dispersion,central_coordinates_X = evaluate_spectral_dispersion_values(x_vec_out, y_vec_out)
    r2, y_pred = evaluate_linearity(central_coordinates_X)

    y_second_derivative = calculate_second_derivative(cassi_system.wavelengths,central_coordinates_X)

    plt.plot(y_second_derivative)
    plt.show()

    #no distorsions X_vec_out, Y_vec_out
    x_vec_out_no_dist, y_vec_out_no_dist = get_cassi_system_no_spectial_distorsions(cassi_system.X_coded_aper_coordinates,
                                                                                    cassi_system.Y_coded_aper_coordinates,
                                                                                    central_coordinates_X)

    print(r2)

    plt.plot(central_coordinates_X)
    plt.plot(y_pred)
    plt.show()

    plt.scatter(x_vec_out[...,0], y_vec_out[...,0],color='blue')
    plt.scatter(x_vec_out[:,:,-1], y_vec_out[:,:,-1],color='red')
    plt.scatter(x_vec_out_no_dist[...,0], y_vec_out_no_dist[...,0])
    plt.scatter(x_vec_out_no_dist[:,:,-1], y_vec_out_no_dist[:,:,-1])

    plt.show()