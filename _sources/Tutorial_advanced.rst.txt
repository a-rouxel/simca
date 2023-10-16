Tutorial - Advanced (only script)
=================================




Single acquisition
------------------


This tutorial walks you through the process of running a simple acquisition using the `CassiSystem` class from the :code:`simca` package.

Setup
.....

First, make sure to import the necessary modules:

.. code-block:: python

   from simca import CassiSystem, load_yaml_config

Next, load the configuration files:

.. code-block:: python

   config_dataset = load_yaml_config("simca/configs/dataset.yml")
   config_system = load_yaml_config("simca/configs/cassi_system.yml")
   config_patterns = load_yaml_config("simca/configs/pattern.yml")
   config_acquisition = load_yaml_config("simca/configs/acquisition.yml")

Then, set the name of the dataset of interest:

.. code-block:: python

   dataset_name = "indian_pines"

Initialize the CassiSystem
..............................

Initialize the `CassiSystem`:

.. code-block:: python

   cassi_system = CassiSystem(system_config=config_system)

Load the Hyperspectral dataset
..............................

Load the hyperspectral dataset:

.. code-block:: python

   cassi_system.load_dataset(dataset_name, config_dataset["datasets directory"])

Generate the Coded Aperture Pattern
........................................

Generate the coded aperture pattern:

.. code-block:: python

   cassi_system.generate_2D_pattern(config_patterns)

Propagate the Coded Aperture Grid
...................................

Propagate the coded aperture grid to the detector plane:

.. code-block:: python

   cassi_system.propagate_coded_aperture_grid()

Generate the Filtering Cube
..............................

Generate the filtering cube:

.. code-block:: python

   cassi_system.generate_filtering_cube()

(Optional) Generate the PSF
..............................

Generate the PSF of the optical system:

.. code-block:: python

   cassi_system.optical_model.generate_psf(type="Gaussian",radius=100)

Simulate the Acquisition
.........................

Simulate the acquisition (with PSF in this case):

.. code-block:: python

   cassi_system.image_acquisition(use_psf=True, chunck_size=50)

Save the Acquisition
.........................

Finally, save the acquisition:

.. code-block:: python

   cassi_system.save_acquisition(config_patterns, config_acquisition)

And that's it! You've successfully run an acquisition using the `CassiSystem` class from the :code:`simca` package.


Multiple acquisitions
----------------------

This tutorial walks you through the process of running multiple acquisitions using the `CassiSystem` class from the :code:`simca` package.

Setup
.........................

First, make sure to import the necessary modules and configurations:


.. code-block:: python

   import matplotlib.pyplot as plt
   from simca import CassiSystem
   from simca.functions_general_purpose import *
   import os

   config_dataset = load_yaml_config("simca/configs/dataset.yml")
   config_system = load_yaml_config("simca/configs/cassi_system.yml")
   config_patterns = load_yaml_config("simca/configs/pattern.yml")
   config_acquisition = load_yaml_config("simca/configs/acquisition.yml")

   dataset_name = "indian_pines"
   results_directory = "./data/results/lego_test_1"
   nb_of_acq = 10


Initialize the CassiSystem
..................................................

Initialize the `CassiSystem`:


.. code-block:: python

   cassi_system = CassiSystem(system_config=config_system)


Load the Hyperspectral dataset
..................................................

Load the hyperspectral dataset:


.. code-block:: python

   cassi_system.load_dataset(dataset_name, config_dataset["datasets directory"])


Generate Multiple Patterns for Acquisition
...........................................................................

Generate multiple coded aperture patterns:


.. code-block:: python

   cassi_system.generate_multiple_patterns(config_patterns, nb_of_acq)


Propagate the Coded Aperture Grid
..................................................

Propagate the coded aperture grid to the detector plane:


.. code-block:: python

   cassi_system.propagate_coded_aperture_grid()


Generate Multiple Filtering Cubes
..................................................

Generate the multiple filtering cubes:


.. code-block:: python

   cassi_system.generate_multiple_filtering_cubes(nb_of_acq)


Simulate Multiple Acquisitions
..................................................

Simulate multiple acquisitions:


.. code-block:: python

   cassi_system.multiple_image_acquisitions(use_psf=False, nb_of_filtering_cubes=nb_of_acq, chunck_size=50)


Save the Acquisition
.........................

Set up the results directory and save the acquisition:


.. code-block:: python

   cassi_system.result_directory = results_directory
   os.makedirs(results_directory, exist_ok=True)
   
   save_config_file("config_system", cassi_system.system_config, cassi_system.result_directory)
   save_config_file("config_pattern", config_patterns, cassi_system.result_directory)
   save_config_file("config_acquisition", config_acquisition, cassi_system.result_directory)
   save_data_in_hdf5("interpolated_scene", cassi_system.interpolated_scene, cassi_system.result_directory)
   save_data_in_hdf5("panchro", cassi_system.panchro, cassi_system.result_directory)
   save_data_in_hdf5("wavelengths", cassi_system.optical_model.system_wavelengths, cassi_system.result_directory)
   save_data_in_hdf5("list_of_compressed_measurements", cassi_system.list_of_measurements, cassi_system.result_directory)
   save_data_in_hdf5("list_of_filtering_cubes", cassi_system.list_of_filtering_cubes, cassi_system.result_directory)
   save_data_in_hdf5("list_of_patterns", cassi_system.list_of_patterns, cassi_system.result_directory)

Congratulations! You've successfully performed and saved multiple acquisitions using the `CassiSystem` class from the :code:`simca` package.

