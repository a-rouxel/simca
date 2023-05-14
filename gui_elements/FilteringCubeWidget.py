import yaml
from PyQt5.QtWidgets import (QTabWidget, QSpinBox,QHBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QWidget, QFormLayout, QScrollArea, QGroupBox,QRadioButton, QButtonGroup,QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot,QCoreApplication

from CassiSystem import CassiSystem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QFormLayout, QGroupBox, QScrollArea
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QLabel, QWidget

from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt

class MaskGridDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.figure_cam = plt.figure()
        self.canvas_cam = FigureCanvas(self.figure_cam)
        self.toolbar_cam = NavigationToolbar(self.canvas_cam, self)

        self.layout.addWidget(self.toolbar_cam)
        self.layout.addWidget(self.canvas_cam)

    def display_mask_grid(self, mask):
        self.figure_cam.clear()

        ax = self.figure_cam.add_subplot(111)
        scatter = ax.imshow(mask,cmap="viridis",interpolation='none',vmin=0,vmax=1)

        # Set labels with LaTeX font.
        ax.set_xlabel(f'X mask grid [um]', fontsize=12)
        ax.set_ylabel(f'Y mask grid[um]', fontsize=12)
        ax.set_title(f'Mask Grid', fontsize=12)

        self.canvas_cam.draw()






class PropagatedMaskGridDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Create a label
        self.label = QLabel("Slice Number: ")
        self.layout.addWidget(self.label)

        # Create a slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.update_image)
        self.layout.addWidget(self.slider)

        # Create ImageView item with a PlotItem as its view box
        self.imageView = pg.ImageView(view=pg.PlotItem())
        self.layout.addWidget(self.imageView)


    def display_propagated_mask_grid(self, filtering_cube,list_wavelengths):
        # Replace NaN values with 0
        self.list_wavelengths = list_wavelengths

        filtering_cube = np.nan_to_num(filtering_cube)



        # remove previous fov

        if hasattr(self, 'fov'):
            self.imageView.getView().removeItem(self.fov)

        # Create a rectangle representing the camera FOV
        self.fov = pg.RectROI([0, 0], [filtering_cube.shape[1], filtering_cube.shape[0]], pen=pg.mkPen(color='r', width=2))
        self.imageView.getView().addItem(self.fov)


        # Normalize filtering_cube values to range [0, 255] for ImageView
        self.data = (filtering_cube - np.min(filtering_cube)) / (np.max(filtering_cube) - np.min(filtering_cube)) * 255
        self.data = self.data.astype(np.uint8)

        # Set slider maximum value
        self.slider.setMaximum(self.data.shape[2] - 1)

        # Display the first slice
        self.update_image(0)

    def update_image(self, slice_index):
        # Update the label
        self.label.setText("wavelength: " + str(int(self.list_wavelengths[slice_index][0,0])) +" nm")

        # Display the slice
        self.imageView.setImage(np.rot90(self.data[:,:,slice_index]), levels=(0, 255))

class Worker(QThread):
    finished_define_mask_grid = pyqtSignal(np.ndarray)
    finished_propagate_mask_grid = pyqtSignal(np.ndarray,list)

    def __init__(self, system_config,simulation_config):
        super().__init__()
        self.system_config = system_config
        self.simulation_config = simulation_config

    def run(self):
        # Put your analysis here
        cassi_system = CassiSystem(system_config=self.system_config ,simulation_config=self.simulation_config)

        X_dmd_grid, Y_dmd_grid = cassi_system.create_dmd_mask()
        mask = cassi_system.generate_2D_mask(self.simulation_config["mask caracteristics"]["type"])


        self.finished_define_mask_grid.emit(mask)  # Emit a tuple of arrays

        list_X_propagated_masks, list_Y_propagated_masks, list_wavelengths = cassi_system.propagate_mask_grid(X_dmd_grid,
                                                                                                            Y_dmd_grid,
                                                                                                              [self.simulation_config["spectral range"]["wavelength min"],
                                                                                                               self.simulation_config["spectral range"]["wavelength max"]],
                                                                                                               self.simulation_config["number of spectral samples"])
        filtering_cube = cassi_system.generate_filtering_cube(cassi_system.X_detector_grid,
                                               cassi_system.Y_detector_grid,
                                               list_X_propagated_masks,
                                               list_Y_propagated_masks,
                                               mask)


        self.finished_propagate_mask_grid.emit(filtering_cube,list_wavelengths)  # Emit a tuple of arrays


class AcquisitionWidgetEditor(QWidget):
    def __init__(self,initial_config_file=None):
        super().__init__()

        self.initial_config_file = initial_config_file


        # Create a QScrollArea
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        # Create a widget for the scroll area
        scroll_widget = QWidget()

        # Add your dimensioning parameters here
        self.results_directory = QLineEdit()
        self.mask_caracteristics_type = QLineEdit()

        self.spectral_samples = QSpinBox()
        self.spectral_samples.setMinimum(1)
        self.spectral_samples.setMaximum(500)

        self.wavelength_min = QSpinBox()
        self.wavelength_min.setMinimum(200)
        self.wavelength_min.setMaximum(5000)

        self.wavelength_max = QSpinBox()
        self.wavelength_max.setMinimum(200)
        self.wavelength_max.setMaximum(5000)

        wavelength_layout = QFormLayout()
        wavelength_layout.addRow("Wavelength min", self.wavelength_min)
        wavelength_layout.addRow("Wavelength max", self.wavelength_max)


        general_layout = QFormLayout()
        general_layout.addRow("results directory", self.results_directory)
        general_layout.addRow("mask caracteristics", self.mask_caracteristics_type)
        general_layout.addRow("spectral samples", self.spectral_samples)



        wavelength_group = QGroupBox("Spectral Range")
        wavelength_group.setLayout(wavelength_layout)

        general_group = QGroupBox("General settings")
        general_group.setLayout(general_layout)

        # Load config button
        self.load_config_button = QPushButton("Load Config")
        self.load_config_button.clicked.connect(self.on_load_config_clicked)

        # Create main layout and add widgets
        main_layout = QVBoxLayout()
        main_layout.addWidget(general_group)
        main_layout.addWidget(wavelength_group)
        main_layout.addWidget(self.load_config_button)

        # Set the layout of the widget within the scroll area
        scroll_widget.setLayout(main_layout)

        # Set the widget for the scroll area
        scroll.setWidget(scroll_widget)

        # Create a layout for the current widget and add the scroll area
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

        # Load the initial configuration file if one was provided
        if self.initial_config_file is not None:
            self.load_config(initial_config_file)


    def on_load_config_clicked(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open YAML", "", "YAML Files (*.yml)")
        if file_name:
            self.load_config(file_name)

    def load_config(self, file_name):
        with open(file_name, 'r') as file:
            self.config = yaml.safe_load(file)
        # Call a method to update the GUI with the loaded config
        self.update_config()

    def update_config(self):
        # This method should update your QLineEdit and QSpinBox widgets with the loaded config.

        self.results_directory.setText(self.config['infos']['results directory'])
        self.mask_caracteristics_type.setText(self.config['mask caracteristics']['type'])

        self.wavelength_min.setValue(self.config['spectral range']['wavelength min'])

        self.wavelength_max.setValue(self.config['spectral range']['wavelength max'])
        #
        # self.sampling_across_X.setValue(self.config['input grid sampling']['sampling across X'])
        # self.sampling_across_Y.setValue(self.config['input grid sampling']['sampling across Y'])
        # self.delta_X.setText(str(self.config['input grid sampling']['delta X']))
        # self.delta_Y.setText(str(self.config['input grid sampling']['delta Y']))
        #
        self.spectral_samples.setValue(self.config['number of spectral samples'])

    def get_config(self):
        return {
            "infos": {
                "results directory": self.results_directory.text()
            },
            "mask caracteristics": {
                "type": self.mask_caracteristics_type.text()
            },
            "number of spectral samples": self.spectral_samples.value(),
            "spectral range": {
                "wavelength min": self.wavelength_min.value(),
                "wavelength max": self.wavelength_max.value()
            },
        }

class FilteringCubeWidget(QWidget):
    def __init__(self, editor_system_config):
        super().__init__()

        self.editor_system_config = editor_system_config


        self.layout = QHBoxLayout()

        # Create the dimensioning configuration editor
        self.acquisition_config_editor = AcquisitionWidgetEditor(initial_config_file="config/acquisition.yml")

        # Create the result display widget (tab widget in this case)
        self.result_display_widget = QTabWidget()

        # Create the result displays and store them as attributes
        self.camera_result_display = MaskGridDisplay()
        self.propagated_mask_display = PropagatedMaskGridDisplay()


        # Add the result displays to the tab widget
        self.result_display_widget.addTab(self.camera_result_display, "Mask Grid")
        self.result_display_widget.addTab(self.propagated_mask_display, "Propagated Mask Grid")


        # Create the run button
        self.run_button = QPushButton('generate filtering cube')
        self.run_button.setStyleSheet('QPushButton {background-color: red; color: white;}')        # Connect the button to the run_dimensioning method
        self.run_button.clicked.connect(self.run_dimensioning)

        # Create a group box for the run button
        self.run_button_group_box = QGroupBox()
        run_button_group_layout = QVBoxLayout()

        run_button_group_layout.addWidget(self.run_button)
        run_button_group_layout.addWidget(self.result_display_widget)

        self.run_button_group_box.setLayout(run_button_group_layout)



        # Add the dimensioning configuration editor, the result display widget, and the run button to the layout
        self.layout.addWidget(self.acquisition_config_editor)
        self.layout.addWidget(self.run_button_group_box)

        self.layout.setStretchFactor(self.run_button_group_box, 1)
        self.layout.setStretchFactor(self.result_display_widget, 2)

        # Set the layout on the widget
        self.setLayout(self.layout)




    def run_dimensioning(self):
        # Get the configs from the editors


        system_config = self.editor_system_config.get_config()
        acquisition_config_editor = self.acquisition_config_editor.get_config()

        self.worker = Worker(system_config, acquisition_config_editor)
        self.worker.finished_define_mask_grid.connect(self.display_mask_grid)
        self.worker.finished_propagate_mask_grid.connect(self.display_propagated_masks)
        self.worker.start()

        QCoreApplication.processEvents()


    @pyqtSlot(np.ndarray)
    def display_mask_grid(self, mask):
        self.camera_result_display.display_mask_grid(mask)

    @pyqtSlot(np.ndarray,list)
    def display_propagated_masks(self, filtering_cube,list_of_wavelengths):
        self.propagated_mask_display.display_propagated_mask_grid(filtering_cube,list_of_wavelengths)
