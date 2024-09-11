"""
SD-CASSI System Design Optimization Script

This script optimizes the design of a Snapshot Dispersive Coded Aperture Spectral Imaging (SD-CASSI) system. 
It performs a two-step optimization process and generates performance metrics for the final optimized system.

Usage:
    python design_cassi.py --prism_type [amici|single] --output_dir <output_directory>

Author: Antoine Rouxel
Date: 06/09/2024
"""





import argparse
from pprint import pprint
import os


from simca.cost_functions_optics import (
    optimize_cassi_system,
    test_cassi_system,
    plot_optimization_process
)
from simca.functions_optim import save_config_system

def optimize_step(params_to_optimize, cost_weights, init_config_path, target_dispersion, iterations, patience, device, index_estimation_method, step_name, output_dir):
    """
    Perform a single optimization step for the CASSI system.

    Args:
        params_to_optimize (list): List of parameters to optimize.
        cost_weights (dict): Weights for different cost components.
        init_config_path (str): Path to the initial configuration file.
        target_dispersion (float): Target dispersion value in micrometers.
        iterations (int): Maximum number of optimization iterations.
        patience (int): Number of iterations to wait for improvement before early stopping.
        device (str): Computation device ('cuda' or 'cpu').
        index_estimation_method (str): Method for estimating refractive index ('cauchy' or 'sellmeier').
        step_name (str): Name of the optimization step.
        output_dir (str): Directory to save optimization results.

    Returns:
        tuple: Final configuration and latest optical parameters.
    """
    final_config, latest_optical_params = optimize_cassi_system(
        params_to_optimize,
        target_dispersion,
        cost_weights,
        init_config_path,
        iterations,
        patience,
        output_dir,
        step_name,
        index_estimation_method=index_estimation_method,
        device=device
    )
    
    # Plot the optimization process after each step
    plot_optimization_process(output_dir, step_name)
    
    return final_config, latest_optical_params


def main(prism_type, output_dir):
    """
    Main function to run the CASSI system optimization process.

    Args:
        prism_type (str): Type of prism system ('amici' or 'single').
        output_dir (str): Directory to save output files.
    """
    
    target_dispersion = 1000  # in [µm] ---> modify to spectral spreading
    iterations = 2000
    patience = 500
    device = "cuda"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the final config path
    final_config_path = os.path.join(output_dir, "config_system.yml")

    if prism_type == "amici":
        params_step1 = ['lba_c', 'alpha_c', 'A1', 'A2', 'nd1', 'nd2', 'vd1', 'vd2']
        init_config_path_step1 = "./init_configs/system_config_amici.yml"
        params_step2 = ['lba_c', 'alpha_c', 'A1', 'A2']
        cost_weights = {
            'cost_dispersion': 1.0,
            'cost_distance_glasses': 1.0,
            'cost_deviation': 1,
            'cost_distorsion': 1.0,
            'cost_thickness': 1.0,
            'cost_beam_compression': 1.0,
            'cost_non_linearity': 1.0,
            'cost_distance_total_intern_reflection': 1.0
        }
    elif prism_type == "single":  # single prism
        params_step1 = ['lba_c', 'alpha_c', 'A1', 'nd1', 'vd1']
        init_config_path_step1 = "./init_configs/system_config_single.yml"
        params_step2 = ['lba_c', 'alpha_c', 'A1']
        cost_weights = {
            'cost_dispersion': 20.0,
            'cost_distance_glasses': 1.0,
            'cost_deviation': 1e-5,
            'cost_distorsion': 1.0,
            'cost_thickness': 1.0,
            'cost_beam_compression': 1.0,
            'cost_non_linearity': 1.0,
            'cost_distance_total_intern_reflection': 1.0
        }
    else:
        raise ValueError(f"Invalid prism type: {prism_type}")

    # STEP 1: Optimize all parameters 
    _, params_step1 = optimize_step(params_step1, cost_weights, init_config_path_step1, target_dispersion, iterations, patience, device, "cauchy", "step1", output_dir)
    
    temp_config_path = os.path.join(output_dir, "temp.yml")
    save_config_system(temp_config_path, os.path.join(output_dir, "optimization_results", "step1"), init_config_path_step1)

    # STEP 2: Optimize parameters excluding prism materials
    if prism_type == "amici":
        cost_weights['cost_dispersion'] = 2.0  # Increase weight for dispersion in step 2 for Amici
    _, params_step2 = optimize_step(params_step2, cost_weights, temp_config_path, target_dispersion, iterations, patience, device, "sellmeier", "step2", output_dir)
    
    save_config_system(final_config_path, os.path.join(output_dir, "optimization_results", "step2"), temp_config_path, index_estimation_method="sellmeier")

   # Test the final optimized system
    config_system, performances = test_cassi_system(final_config_path, index_estimation_method="sellmeier")

    print("\n---- Optical System Configuration----")
    pprint(config_system)

    print("\n---- Optical Performances ----")
    pprint(performances)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Design SD-CASSI system with Amici or single prism")
    parser.add_argument("--prism_type", choices=["amici", "single"], required=True, help="Type of prism system to design")
    parser.add_argument("--output_dir", required=True, help="Directory for the final configuration file")
    args = parser.parse_args()

    main(args.prism_type, args.output_dir)