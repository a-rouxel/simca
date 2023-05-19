# SIMCA: Coded Aperture Snapshot Spectral Imaging (CASSI) Simulator

SIMCA is a Python/Qt application designed to perform optical simulations for Coded Aperture Snapshot Spectral Imaging (CASSI). 

Go check the documentation page [here](https://arouxel.gitlab.io/simca-documentation/)

## Installation

To install SIMCA, follow the steps below:

1. Clone the repository from GitLab:

```bash
git clone git@gitlab.laas.fr:arouxel/simca.git
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

## Quick Start

4. Download the standard scenes from this [link](https://cloud.laas.fr/index.php/s/zfh5RFmsjYfk108) and paste the `datasets` folder in the root directory of SIMCA.

5. Start the application:

```bash
# run the app
python main.py
```

## Main Features

SIMCA includes four main features:

- **Scene Analysis**: This module is used to analyze input multi- or hyper-spectral input scenes. It includes data slices, spectrum analysis, and ground truth labeling.

- **Optical Design**: This module is used to compare the performances of various optical systems.

- **Masks Generation**: This module is used to generate spectral and/or spatial filtering, based on the optical design.

- **Acquisition Coded Images**: This module is used to encode and acquire images.

![SIMCA Layout](/img/layout_general.svg)

For more detailed information about each feature and further instructions, please visit our [documentation website](https://arouxel.gitlab.io/simca-documentation/).

## License

SIMCA is licensed under the [MIT License](https://www.mit.edu/~amini/LICENSE.md).

## Contact

For any questions or feedback, please contact us at arouxel@laas.fr
