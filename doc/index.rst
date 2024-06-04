.. simca documentation master file, created by
   sphinx-quickstart on Fri Aug  4 17:05:16 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SIMCA : optical simulations for coded spectral imaging
=======================================================


.. image:: ./resources/simca_logo.png

SIMCA is a python-based tool designed to perform optical simulations of Coded Aperture Snapshot Spectral Imaging (CASSI) systems.
We provide a python package available on Pypi.

It is built upon ray-tracing equations and interpolation methods to estimate the image formation process and generate realistic measurements of various cassi instruments.

Available **system architectures** are:

- Single-Disperser CASSI (:cite:`Wagadarikar2008`) 
- Double-Disperser CASSI (:cite:`Gehm2007`)

Available **propagation models** are:

- Ray-tracing from :cite:`Rouxel2024`
- Higher-Order from :cite:`Arguello2013`

Available **optical components** and related characteristics are:

- Lens (params: focal length)
- Simple Prism (params: apex angle, glass type, orientation misalignments)
- Doublet Prism (params: apex angle, glass type, orientation misalignments)
- Amici Prism (params: apex angle, glass type, orientation misalignments)
- Triple Prism (params: apex angle, glass type, orientation misalignments) 
- Grating (params: groove density, orientation misalignments)


Main Features
=============

SIMCA includes four main features:

- **Scene Analysis** (only with GUI): for analyzing multi- or hyper-spectral datasets. It includes vizualization of data slices, spectrum analysis, and dataset labeling.

- **Optical Design**: for evaluating and comparing the performances of various optical systems.

- **Coded Aperture patterns Generation**: for generating various patterns and corresponding sensing matrix.

- **Acquisition Coded Images**: for simulating the acquisition process

For more detailed information about each feature and further instructions, please visit our `Tutorial - Basics (with GUI) <Tutorial_with_GUI.html>`_ and `Tutorial - Advanced (only script) <Tutorial_advanced.html>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   Tutorial_with_GUI
   Tutorial_advanced
   simca




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


License
=======

SIMCA is licensed under the `GNU General Public License <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.


Contact
=======

For any questions or feedback, please contact us at arouxel@laas.fr

References
==========
.. bibliography:: ./resources/biblio.bib
