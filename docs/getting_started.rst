.. _getting_started:

Getting started
===============



Installation
------------

To install :code:`simca`, follow the steps below:

1. Clone the repository from GitLab:

.. code-block:: bash

   git clone git@gitlab.laas.fr:arouxel/simca.git
   cd simca

2. Create a dedicated Python environment using Miniconda. If you don't have Miniconda installed, you can find the instructions `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.

.. code-block:: bash

   # Create a new Python environment
   conda create -n simca-env python=3.9

   # Activate the environment
   conda activate simca-env

3. Install the necessary Python packages that simca relies on. These are listed in the `requirements.txt` file in the repository.

.. code-block:: bash

   # Install necessary Python packages with pip
   pip install -r requirements.txt
   

Usage
-----   

Download datasets
^^^^^^^^^^^^^^^^^^

4. Download the standard datasets from this `link <https://cloud.laas.fr/index.php/s/zfh5RFmsjYfk108/download>`_, then unzip and paste the `datasets` folder in the root directory of SIMCA.

Quick Start with GUI (option 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

5. Start the application:

.. code-block:: bash

   # run the app
   python main.py

Quick Start with API (option 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

5. Run the example script :

.. code-block:: bash

   # run the script
   python simple_script.py

