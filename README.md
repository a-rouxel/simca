# SIMCA -- Dimensioning and Acquisition

This Python/Qt application allows users to run dimensioning analysis and acquisition procedures on CASSI systems.

## Features

The application provides the following features:

1. Allows users to specify various system parameters.
2. Displays results of the dimensioning analysis in different forms, such as camera pixelization, mapping in the DMD plane, spectral dispersion, and distortion map.
3. Facilitates the creation and configuration of a filtering cube.
4. Allows for the execution of an acquisition procedure.

## Install

```bash
# Clone the repository 
git clone git@gitlab.laas.fr:arouxel/simca-revival.git
cd simca-revival
```

Once the repository has been cloned, create a dedicated python environment with python=3.9. In these instructions, the python environment is created using Miniconda. It can be installed by following the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

```bash
# create a new python environment
conda create -n simca-revival-env python=3.9
# activate the environment
conda activate simca-revival-env
```

Now we have to install the packages we use:

```bash
# install packages contained in the "requirements.txt" with pip:
pip install -r requirements.txt
```

## Usage

The application provides a graphical interface for running dimensioning analysis and acquisition procedures.

- The application window is divided into multiple sections. On the left is a dock for inputting parameters. On the right are dockable tabs for dimensioning, filtering cube configuration, and acquisition.

- You can input the system configuration parameters in the left dock of the window.

- The dimensioning, filtering cube, and acquisition procedures can be initiated and configured in their respective tabs.

