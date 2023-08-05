Tutorial with GUI
=================


Discover Main Features
-----------------------

The are 4 main features included in the application. These modules are not completely independent, using them sequentially is recommended for first usages.

- **Scene Analysis** (only with GUI): for analyzing multi- or hyper-spectral datasets. It includes vizualization of data slices, spectrum analysis, and dataset labeling.

- **Optical Design**: for evaluating and comparing the performances of various optical systems.

- **Coded Aperture patterns Generation**: for generating various patterns and corresponding filtering cubes.

- **Acquisition Coded Images**: for simulating the acquisition process

.. image:: /resources/layout_general.svg
   :alt: Docusaurus logo

Feature A : Scene Analysis
--------------------------

The Scene analysis tab is used to **load & display scene characteristics**.

.. image:: /resources/layout_scene_tab.svg
   :alt: layout scene tab

Settings
..........

Located on the left side of the application window.

Includes:

- `scenes directory` : path to the scenes directory. All scene data should be stored here.
    - click on the button if you change the scenes directory path

- `available scenes` : ComboBox displaying the scenes available in the scenes directory

- `loaded scene dimensions` : These values are displayed once the scene is loaded
    - `dimension along X` : dimension of the scene in the X direction (main spectral dispersion direction)
    - `dimension along Y` : dimension of the scene in the Y direction (perpendicular to spectral dispersion direction)
    - `number of spectral bands` : number of spectral bands in the loaded scene
    - `minimum wavelength` : minimum wavelength, usually corresponds to the spectral band n°0
    - `maximum wavelength` : maximum wavelength, usually corresponds to the last spectral band

Load scene button
.................

By clicking on this button, the scene selected in the `available scenes` ComboBox is loaded by the application.

Display windows
...............

Located on the right side of the application window.

**Once a scene is loaded**, one can inspect the spatial and spectral content of the scene.

Spectral images
^^^^^^^^^^^^^^^^

.. image:: /resources/layout_scene_2.svg
   :alt: SCene layout 2

Compare Spectrums
^^^^^^^^^^^^^^^^

.. image:: /resources/layout_scene_3.svg
   :alt: SCene layout 2

Labelisation map
^^^^^^^^^^^^^^^^

.. image:: /resources/layout_scene_4.svg
   :alt: SCene layout 2

Labelisation Histogram
^^^^^^^^^^^^^^^^

.. image:: /resources/layout_scene_5.svg
   :alt: SCene layout 2
