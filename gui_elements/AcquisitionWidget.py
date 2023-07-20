from PyQt5.QtWidgets import (QTabWidget, QHBoxLayout, QPushButton, QComboBox, QLineEdit, QCheckBox)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QFormLayout, QGroupBox, QScrollArea
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QLabel, QWidget
from utils.helpers import *
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np

class AcquisitionPanchromaticWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Create ImageView item with a PlotItem as its view box
        self.imageView = pg.ImageView(view=pg.PlotItem())
        self.layout.addWidget(self.imageView)

    def display_panchrom_acquisition(self, measurement_3D):
        """

        """
        # Compute the sum along the third axis
        image = np.sum(measurement_3D, axis=2)

        # Display the image
        self.imageView.setImage(np.rot90(image))


class AcquisitionSlideBySlideWidget(QWidget):
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
    def display_measurement_slice_by_slice(self, measurement_3D):

        # Normalize filtering_cube values to range [0, 255] for ImageView
        self.data = measurement_3D


        # Set slider maximum value
        self.slider.setMaximum(self.data.shape[2] - 1)

        # Display the first slice
        self.update_image(0)


    def update_image(self, slice_index):
        # Update the label
        self.label.setText("slice_index: " + str(int(slice_index)) )
        # Display the slice
        self.imageView.setImage(np.rot90(self.data[:, :, slice_index]))

class AcquisitionDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Create ImageView item with a PlotItem as its view box
        self.imageView = pg.ImageView(view=pg.PlotItem())
        self.layout.addWidget(self.imageView)

    def display_acquisition(self, measurement_3D):
        """

        """
        # Compute the sum along the third axis
        image = np.sum(measurement_3D, axis=2)

        # Display the image
        self.imageView.setImage(np.rot90(image))



class AcquisitionEditorWidget(QWidget):

    def __init__(self,initial_config=None):
        super().__init__()

        self.initial_config_file = initial_config


        # Create a QScrollArea
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        # Create a widget for the scroll area
        scroll_widget = QWidget()

        self.directories_combo = QComboBox()
        self.acquisition_types = ['single acq.']
        self.directories_combo.addItems(self.acquisition_types)


        self.results_directory = QLineEdit()
        self.acquisition_name = QLineEdit()
        self.use_psf = QCheckBox()
        self.psf_combo = QComboBox()
        self.psf_radius = QLineEdit()
        self.psf_radius.setText("10")
        self.psf_types = ['Gaussian']
        self.psf_combo.addItems(self.psf_types)


        PSF_form = QFormLayout()
        PSF_form.addRow("apply PSF",self.use_psf)
        PSF_form.addRow("type",self.psf_combo)
        PSF_form.addRow("radius [in um]", self.psf_radius)
        PSF_form_group = QGroupBox("PSF Settings")
        PSF_form_group.setLayout(PSF_form)



        acquisition_layout = QFormLayout()
        acquisition_layout.addRow("acquisition name", self.acquisition_name)
        acquisition_layout.addRow("acquisition type", self.directories_combo)
        acquisition_layout.addRow("results directory", self.results_directory)
        acquisition_layout.addRow(PSF_form_group)

        acquisition_group = QGroupBox("Settings")
        acquisition_group.setLayout(acquisition_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(acquisition_group)

        # Set the layout on the widget within the scroll area
        scroll_widget.setLayout(main_layout)

        # Set the widget for the scroll area
        scroll.setWidget(scroll_widget)

        # Create a layout for the current widget and add the scroll area
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

        # Load the initial configuration file if one was provided
        if self.initial_config_file is not None:
            self.load_config(initial_config)

    def load_config(self, file_name):
        with open(file_name, 'r') as file:
            self.config = yaml.safe_load(file)
        # Call a method to update the GUI with the loaded config
        self.update_config()
    def update_config(self):
        # This method should update your QLineEdit and QSpinBox widgets with the loaded config.
        # self.acquisitions_directory.setText(self.config['acquisition directory'])
        self.directories_combo.setCurrentText(self.config['acquisition type'])
        self.results_directory.setText(self.config['results directory'])
        self.acquisition_name.setText(self.config['acquisition name'])
        self.use_psf.setChecked(self.config['psf']['use_psf'])
        self.psf_combo.setCurrentText(self.config['psf']['type'])
        self.psf_radius.setText(str(self.config['psf']['radius']))

    def get_config(self):
        return {
            "acquisition name": self.acquisition_name.text(),
            "acquisition type": self.directories_combo.currentText(),
            "results directory": self.results_directory.text(),
            "psf": {
                "use_psf": self.use_psf.isChecked(),
                "type": self.psf_combo.currentText(),
                "radius": float(self.psf_radius.text())
            }

        }

class Worker(QThread):
    finished_acquire_measure = pyqtSignal(np.ndarray)
    finished_interpolated_scene = pyqtSignal(np.ndarray)


    def __init__(self,cassi_system,system_editor,filtering_widget, dataset_widget,acquisition_config_editor):
        super().__init__()
        self.cassi_system = cassi_system
        self.system_editor = system_editor
        self.filtering_widget = filtering_widget
        self.dataset_widget = dataset_widget
        self.acquisition_config = acquisition_config_editor.get_config()


    def run(self):

        print("Acquisition started")

        self.system_config = self.system_editor.get_config()
        self.cassi_system.update_config(self.system_config)

        if self.acquisition_config["psf"]["use_psf"] == True:
            self.cassi_system.generate_psf(self.acquisition_config["psf"]["type"],self.acquisition_config["psf"]["radius"])
        self.cassi_system.image_acquisition(use_psf=self.acquisition_config["psf"]["use_psf"],chunck_size=50)
        self.finished_interpolated_scene.emit(self.cassi_system.interpolated_scene)
        self.finished_acquire_measure.emit(self.cassi_system.last_filtered_interpolated_scene)  # Emit a tuple of arrays

        print("Acquisition finished")

class AcquisitionWidget(QWidget):
    def __init__(self,cassi_system,system_editor,dataset_widget, filtering_widget,acquisition_config_path="config/acquisition.yml"):
        super().__init__()

        self.cassi_system = cassi_system
        self.system_editor = system_editor
        self.dataset_widget = dataset_widget
        self.filtering_widget = filtering_widget
        self.last_filtered_interpolated_scene = None
        self.interpolated_scene = None

        self.layout = QHBoxLayout()

        if acquisition_config_path is not None:
            self.acquisition_config_editor = AcquisitionEditorWidget(acquisition_config_path)

        self.layout.addWidget(self.acquisition_config_editor)

        self.result_display_widget = QTabWidget()

        self.acquisition_display = AcquisitionDisplay()
        self.acquisition_by_slice_display = AcquisitionSlideBySlideWidget()
        self.acquisition_panchro_display = AcquisitionPanchromaticWidget()

        self.result_display_widget.addTab(self.acquisition_display, "Measured Image")
        self.result_display_widget.addTab(self.acquisition_by_slice_display, "Measured Image for each spectral sample")
        self.result_display_widget.addTab(self.acquisition_panchro_display, "Panchromatic Image")

        self.run_button = QPushButton('Run Acquisition')
        # self.run_button.setStyleSheet('QPushButton {background-color: black; color: white;}')        # Connect the button to the run_dimensioning method
        self.run_button.clicked.connect(self.run_acquisition)

        self.save_acquisition_button = QPushButton("Save Acquisition")
        self.save_acquisition_button.clicked.connect(self.on_acquisition_saved)


        # Create a group box for the run button
        self.run_button_group_box = QGroupBox()
        run_button_group_layout = QVBoxLayout()

        run_button_group_layout.addWidget(self.run_button)
        run_button_group_layout.addWidget(self.save_acquisition_button)
        run_button_group_layout.addWidget(self.result_display_widget)

        self.run_button_group_box.setLayout(run_button_group_layout)
        self.layout.addWidget(self.run_button_group_box)


        self.layout.setStretchFactor(self.run_button_group_box, 1)
        self.layout.setStretchFactor(self.result_display_widget, 3)
        self.setLayout(self.layout)

    def run_acquisition(self):
        # Get the configs from the editors

        self.worker = Worker(self.cassi_system,self.system_editor,self.filtering_widget,self.dataset_widget,self.acquisition_config_editor)
        self.worker.finished_acquire_measure.connect(self.display_acquisition)
        self.worker.finished_acquire_measure.connect(self.display_measurement_by_slide)
        self.worker.finished_interpolated_scene.connect(self.display_panchrom_display)
        self.worker.start()

    def on_acquisition_saved(self):
        self.config_acquisition = self.acquisition_config_editor.get_config()
        self.config_filtering = self.filtering_widget.get_config()
        self.cassi_system.save_acquisition(self.config_filtering,self.config_acquisition)

    @pyqtSlot(np.ndarray)
    def display_acquisition(self, measurement_3D):
        self.acquisition_display.display_acquisition(measurement_3D)
    def display_measurement_by_slide(self, measurement_3D):
        self.last_filtered_interpolated_scene = measurement_3D
        self.acquisition_by_slice_display.display_measurement_slice_by_slice(measurement_3D)
    def display_panchrom_display(self, scene):
        self.interpolated_scene = scene
        self.acquisition_panchro_display.display_panchrom_acquisition(scene)


