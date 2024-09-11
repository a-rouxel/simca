import argparse
from pprint import pprint
import os

from simca.cost_functions_optics import test_cassi_system

def analyze_system(input_dir):

    final_config_path = os.path.join(input_dir, "config_system.yml")
    config_system, performances = test_cassi_system(final_config_path, save_fig_dir=input_dir,save_fig=True)

    print(f"\n---- Optical System Configuration for {final_config_path} ----")
    pprint(config_system)

    print(f"\n---- Optical Performances for {final_config_path} ----")
    pprint(performances)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze existing SD-CASSI system")
    parser.add_argument("--input_dir", required=True, help="Path to the configuration file to analyze")
    args = parser.parse_args()

    analyze_system(args.input_dir)

