[![pages-build-deployment](https://github.com/a-rouxel/simca/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/a-rouxel/simca/actions/workflows/pages/pages-build-deployment)

# SIMCA: Coded Aperture Snapshot Spectral Imaging (CASSI) Simulator

SIMCA is a Python/Qt application designed to perform optical simulations for Coded Aperture Snapshot Spectral Imaging (CASSI).

Go check the documentation page [here](https://a-rouxel.github.io/simca/)

## Installation

To perform again our experiments on SIMCA for the Optical Sensing Congress 2024, follow the steps below:

1. Clone the repository from Github:

```bash
git clone -b optica-sensing-congress https://github.com/a-rouxel/simca.git
cd simca
```

2. Create a dedicated Python environment using Miniconda. If you don't have Miniconda installed, you can find the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

```bash
# Create a new Python environment with the required packages
conda env create -f environment.yml
```

3. Activate the environment

```bash
conda activate simca
```

## Download datasets

4. Download the standard datasets from this [link](https://partage.laas.fr/owncloud/index.php/s/geUrFeV1tI32pCr), then unzip and paste the `datasets_reconstruction` folder in the root directory of SIMCA.
```bash
|--simca
    |--MST
    |--simca
    |--utils
    |--datasets_reconstruction
        |--mst_datasets
            |--cave_1024_28_test
                |--scene2.mat
                ：  
                |--scene191.mat
            |--cave_1024_28_train
                |--scene1.mat
                ：  
                |--scene205.mat
```

## Download the checkpoints

5. If you want to use our saved checkpoints to run the architecture on the test dataset, download the checkpoints from this [link](https://partage.laas.fr/owncloud/index.php/s/IMtIe5IKeyvf2E0), then unzip and paste the `saved_checkpoints` folder in the root directory of SIMCA.
```bash
|--simca
    |--MST
    |--simca
    |--utils
    |--datasets_reconstruction
        |--mst_datasets
            |--cave_1024_28_test
                |--scene2.mat
                ：  
                |--scene191.mat
            |--cave_1024_28_train
                |--scene1.mat
                ：  
                |--scene205.mat
    |--saved_checkpoints
```

## Train the framework from scratch

If you wish to train the framework again :
1. Run the training_simca_reconstruction.py script, this corresponds to the reconstruction with random masks:
```bash
python training_simca_reconstruction.py
```
The checkpoints will be saved in the ```checkpoints``` folder.

2. Run the training_simca_reconstruction_full scripts, this corresponds to the reconstruction either with fine-tuned float or binary masks:
```bash
python training_simca_reconstruction_full_binary.py
python training_simca_reconstruction_full_float.py
```
The checkpoints will be saved in the ```checkpoints_full_binary``` and ```checkpoints_full_float``` folders .

## Testing the framework

If you wish to test the framework:

0. If you want to use other checkpoints than the ones provided, change the value of the following variables with the path of your checkpoints:
```bash
test_simca_reconstruction.py > reconstruction_checkpoint
test_simca_reconstruction_full_binary.py > reconstruction_checkpoint, full_model_checkpoint
test_simca_reconstruction_full_float.py > reconstruction_checkpoint, full_model_checkpoint
```
```reconstruction_checkpoint``` is the path to the checkpoint generated in the ```checkpoints``` folder.
```full_model_checkpoint``` is the path to the checkpoint generated in the ```checkpoints_full_binary``` and  ```checkpoints_full_float``` folders respectively.

1. Run the test scripts:

```bash
python test_simca_reconstruction.py
python test_simca_reconstruction_full_binary.py
python test_simca_reconstruction_full_float.py
```
The results will be saved in the ```results``` folder. Afterwards, you can also run the ```summarize_results.py``` script to average results over the runs and per scene. 

2. (Optional) Run the visualization script:
With this script you will be able to compare the reconstruction spectra of a few points in a scene.
```bash
python show_spectrum.py
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

SIMCA is licensed under the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contact

For any questions or feedback, please contact us at lpaillet@laas.fr
