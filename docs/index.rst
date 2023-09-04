.. simca documentation master file, created by
   sphinx-quickstart on Fri Aug  4 17:05:16 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SIMCA : optical simulations for coded spectral imaging
=======================================================


.. image:: ./resources/SIMCA_logo-2-cropped.png

SIMCA is a python-based tool designed to perform optical simulations of Coded Aperture Snapshot Spectral Imaging (CASSI) systems.
We provide an application programming interface (API) and a graphical-user interface (GUI) developped in PyQt5.

It is built upon ray-tracing equations and interpolation methods to estimate the image formation process and generate realistic measurements of various cassi instruments.

Available **system architectures** are:

- Single-Disperser CASSI (:cite:`Wagadarikar2008`) 
- Double-Disperser CASSI (:cite:`Gehm2007`)

Available **propagation models** are:

- Higher-Order from :cite:`Arguello2013`
- Ray-tracing (first implementation in :cite:`Hemsley2020a`, another paper will be submitted soon)

Available **optical components** and related characteristics are:

- Lens (params: focal length)
- Prism (params: apex angle, glass type, orientation misalignments)
- Grating (params: groove density, orientation misalignments)

More system architectures and optical components will be added in the future.


Main Features
=============

:code:`simca` includes four main features:

- **Scene Analysis** (only with GUI): for analyzing multi- or hyper-spectral datasets. It includes vizualization of data slices, spectrum analysis, and dataset labeling.

- **Optical Design**: for evaluating and comparing the performances of various optical systems.

- **Coded Aperture patterns Generation**: for generating various patterns and corresponding filtering cubes.

- **Acquisition Coded Images**: for simulating the acquisition process

For more detailed information about each feature and further instructions, please visit our `Tutorial - Basics (with GUI) <Tutorial_with_GUI.html>`_ and `Tutorial - Advanced (script) <Tutorial_advanced.html>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   Tutorial_with_GUI
   Tutorial_advanced
   cassi_systems




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


License
=======

SIMCA is licensed under the `MIT License <https://www.mit.edu/~amini/LICENSE.md>`_.

Contact
=======

For any questions or feedback, please contact us at arouxel@laas.fr

References
==========
.. bibliography:: ./resources/biblio.bib
