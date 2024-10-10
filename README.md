# SIMCA: Coded Aperture Snapshot Spectral Imaging (CASSI) Simulator

SIMCA is a package designed to perform optical simulations for Coded Aperture Snapshot Spectral Imaging (CASSI). 


## Installation

To install SIMCA, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/a-rouxel/simca.git
cd simca
```

2. Create and activate a dedicated Python environment using Miniconda:

```bash
conda create -n simca-env python=3.10
conda activate simca-env
```

3. Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

## Usage

SIMCA provides two main scripts for designing and analyzing CASSI systems:

### Designing a CASSI System

Use the `design_cassi.py` script to optimize the design of a Snapshot Dispersive Coded Aperture Spectral Imaging (SD-CASSI) system:

```bash
python design_cassi.py --prism_type [amici|single] --output_dir <output_directory>
```

- `--prism_type`: Choose between "amici" or "single" prism system.
- `--output_dir`: Specify the directory to save the optimization results.

This script performs a two-step optimization process and generates performance metrics for the final optimized system.

### Analyzing a CASSI System

Use the `analyze_cassi.py` script to analyze an existing SD-CASSI system caracteristics from a configuration file:

```bash
python analyze_cassi.py --input_dir <input_directory>
```

- `--input_dir`: Specify the directory containing the `config_system.yml` file to analyze.

This script will display the optical system configuration and performance metrics, and save visualization figures in the input directory.


## License

SIMCA is licensed under the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contact

For questions or feedback, please contact us at arouxel@laas.fr
