import os 
import json
import yaml


from simca import load_yaml_config
from simca.CassiSystem import CassiSystem
from simca.cost_functions_optics import evaluate_distance, get_catalog_glass_infos





def save_config_system(output_config_path, results_dir, config_path, index_estimation_method='cauchy', iteration_nb=None):
    
    
    config_system = load_yaml_config(config_path)
    cassi_system = CassiSystem(system_config=config_system, index_estimation_method=index_estimation_method, device=device)
    device = cassi_system.device
    catalog = config_system["system architecture"]["dispersive element"]["catalog"]


    with open(os.path.join(results_dir, 'optimization_details.json'), 'r') as f:
        optimization_details = json.load(f)

    if iteration_nb is None:
        iteration_nb = max(detail['iterations'] for detail in optimization_details)


    iteration_details = next(detail for detail in optimization_details if detail['iterations'] == iteration_nb)
    latest_optical_params = iteration_details['end_parameters']

    # test a given configuration
    list_of_glasses, nd_values, vd_values = get_catalog_glass_infos(catalog=catalog, device=device)

    if cassi_system.optical_model.index_estimation_method == "cauchy":
        current_glass_values_1 = [latest_optical_params["nd1"], latest_optical_params["vd1"]]
        distance_closest_point1, min_idx_1 = evaluate_distance(current_glass_values_1, nd_values, vd_values)
        glass_1 = list_of_glasses[min_idx_1]

        try:
            current_glass_values_2 = [latest_optical_params["nd2"], latest_optical_params["vd2"]]
            distance_closest_point2, min_idx_2 = evaluate_distance(current_glass_values_2, nd_values, vd_values)
            glass_2 = list_of_glasses[min_idx_2]
        except:
            pass
        try:
            current_glass_values_3 = [latest_optical_params["nd3"], latest_optical_params["vd3"]]
            distance_closest_point3, min_idx_3 = evaluate_distance(current_glass_values_3, nd_values, vd_values)
            glass_3 = list_of_glasses[min_idx_3]
        except:
            pass
        
    else:
        glass_1 = cassi_system.optical_model.glass1
        try:
            glass_2 = cassi_system.optical_model.glass2
        except:
            pass
        try:
            glass_3 = cassi_system.optical_model.glass3
        except:
            pass

    cassi_system.system_config["system architecture"]["dispersive element"]["wavelength center"] = latest_optical_params["lba_c"]
    cassi_system.system_config["system architecture"]["dispersive element"]["alpha_c"] = latest_optical_params["alpha_c"]
    cassi_system.system_config["system architecture"]["dispersive element"]["A1"] = latest_optical_params["A1"]
    cassi_system.system_config["system architecture"]["dispersive element"]['glass1'] = glass_1


    if cassi_system.system_config['system architecture']['dispersive element']['type'] in ["doubleprism", "amici", "tripleprism"]:
        cassi_system.system_config["system architecture"]["dispersive element"]["A2"] = latest_optical_params["A2"]
        cassi_system.system_config["system architecture"]["dispersive element"]['glass2'] = glass_2

        
    if cassi_system.system_config['system architecture']['dispersive element']['type'] in ["amici", "tripleprism"]:
        cassi_system.system_config["system architecture"]["dispersive element"]["A3"] = latest_optical_params["A3"]
        cassi_system.system_config["system architecture"]["dispersive element"]['glass3'] = glass_3

    if cassi_system.system_config['system architecture']['dispersive element']['type'] in ["amici"]:
        cassi_system.system_config["system architecture"]["dispersive element"]["A3"] = latest_optical_params["A1"]
        cassi_system.system_config["system architecture"]["dispersive element"]['glass3'] = glass_1

    

    cassi_system.update_optical_model(cassi_system.system_config)

    with open(output_config_path, 'w') as yaml_file:
        yaml.dump(config_system, yaml_file, default_flow_style=False)

