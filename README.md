[![pages-build-deployment](https://github.com/a-rouxel/simca/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/a-rouxel/simca/actions/workflows/pages/pages-build-deployment)


# SIMCA: Coded Aperture Snapshot Spectral Imaging (CASSI) Simulator

SIMCA is a Python/Qt application designed to perform optical simulations for Coded Aperture Snapshot Spectral Imaging (CASSI). 

Go check the documentation page [here](https://a-rouxel.github.io/simca/)

## Installation

To install SIMCA, follow the steps below:

1. Clone the repository from Github:

```bash
git clone https://github.com/a-rouxel/simca.git
cd simca
```

2. Create a dedicated Python environment using Miniconda. If you don't have Miniconda installed, you can find the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

```bash
# Create a new Python environment
conda create -n simca-env python=3.9

# Activate the environment
conda activate simca-env
```

3. Install the necessary Python packages that SIMCA relies on. These are listed in the `requirements.txt` file in the repository.

```bash
# Install necessary Python packages with pip
pip install -r requirements.txt
```

## Download datasets

4. Download the standard datasets from this [link](https://cloud.laas.fr/index.php/s/LQjWVsZgeq27Wz6/download), then unzip and paste the `datasets` folder in the root directory of SIMCA.

## Quick Start with GUI (option 1)

5. Start the application:

```bash
# run the app
python main.py
```

## Quick Start from script (option 2)

5. Run the example script :

```bash
# run the script
python simple_script.py
```

## Main Features

SIMCA includes four main features:

- **Scene Analysis**: This module is used to analyze input multi- or hyper-spectral input scenes. It includes data slices, spectrum analysis, and ground truth labeling.

- **Optical Design**: This module is used to compare the performances of various optical systems.

- **Coded Aperture Patterns Generation**: This module is used to generate spectral and/or spatial filtering, based on the optical design.

- **Acquisition Coded Images**: This module is used to encode and acquire images.

For more detailed information about each feature and further instructions, please visit our [documentation website](https://a-rouxel.github.io/simca/).

## Testing the package

If you wish to run tests on the simca package functionnalities: 

**Attention** : Please note that you need a dataset to test acquisition-related tests

1. Run the tests.py script:

```bash
python tests.py
```

## Building Documentation

If you wish to build the Sphinx documentation locally:

1. Navigate to the documentation source directory (e.g., `doc/`):

```bash
cd doc
```

2. Ensure you have Sphinx and other necessary tools installed:

```bash
pip install sphinx sphinx_rtd_theme
```

3. Build the documentation:

```bash
make html
```

4. Once built, you can view the documentation by opening the generated HTML files. For example:

```bash
xdg-open _build/html/index.html
```

## License

SIMCA is licensed under the `GNU General Public License <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.


## Contact

For any questions or feedback, please contact us at arouxel@laas.fr
