import yaml
from PyQt5.QtWidgets import (QTabWidget,QHBoxLayout, QPushButton,QComboBox,QLineEdit)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QFormLayout, QGroupBox, QScrollArea
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QLabel, QWidget
from utils.helpers import *
import h5py
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import time
from utils.functions_acquisition import get_measurement_in_3D, match_scene_to_instrument
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
        Display the ground truth with each color corresponding to a label_value.

        :param ground_truth: np.array, ground truth labels
        :param label_values: list, label_values[i] = name of the class i
        :param palette: color palette to use, must be a list of colors where the index corresponds to the label
        :param ignored_labels: list of ignored labels (pixel with no label)
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
        Display the ground truth with each color corresponding to a label_value.

        :param ground_truth: np.array, ground truth labels
        :param label_values: list, label_values[i] = name of the class i
        :param palette: color palette to use, must be a list of colors where the index corresponds to the label
        :param ignored_labels: list of ignored labels (pixel with no label)
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

        # Add the dimensioning configuration editor, the result display widget, and the run button to the layout
        self.results_directory = QLineEdit()
        self.acquisition_name = QLineEdit()


        acquisition_layout = QFormLayout()
        acquisition_layout.addRow("acquisition name", self.acquisition_name)
        acquisition_layout.addRow("acquisition type", self.directories_combo)
        acquisition_layout.addRow("results directory", self.results_directory)

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

    def get_config(self):
        return {
            "acquisition name": self.acquisition_name.text(),
            "acquisition type": self.directories_combo.currentText(),
            "results directory": self.results_directory.text(),

        }

class Worker(QThread):
    finished_acquire_measure = pyqtSignal(np.ndarray)
    finished_interpolated_scene = pyqtSignal(np.ndarray)


    def __init__(self,filtering_widget, scene_widget,acquisition_config_editor):
        super().__init__()

        self.filtering_widget = filtering_widget
        self.scene_widget = scene_widget

    def run(self):

        print("Acquisition started")

        filtering_cube = self.filtering_widget.filtering_cube
        filtering_cube_wavelengths = self.filtering_widget.list_wavelengths


        scene = self.scene_widget.scene_config_editor.interpolate_scene(filtering_cube_wavelengths,chunk_size=50)
        scene = match_scene_to_instrument(scene, filtering_cube)

        self.finished_interpolated_scene.emit(scene)

        # Define chunk size
        chunk_size = 50  # Adjust this value based on your system's memory

        t_0 = time.time()
        measurement_in_3D = get_measurement_in_3D(scene, filtering_cube, chunk_size)
        print("Acquisition time: ", time.time() - t_0)

        self.finished_acquire_measure.emit(measurement_in_3D)  # Emit a tuple of arrays

        self.last_measurement_3D = measurement_in_3D
        self.interpolated_scene = scene


        print("Acquisition finished")

class AcquisitionWidget(QWidget):
    def __init__(self,system_editor,scene_widget, filtering_widget,acquisition_config_path="config/acquisition.yml"):
        super().__init__()

        self.system_editor = system_editor
        self.scene_widget = scene_widget
        self.filtering_widget = filtering_widget
        self.last_measurement_3D = None
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
        self.run_button.setStyleSheet('QPushButton {background-color: black; color: white;}')        # Connect the button to the run_dimensioning method
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

        self.worker = Worker(self.filtering_widget,self.scene_widget,self.acquisition_config_editor)
        self.worker.finished_acquire_measure.connect(self.display_acquisition)
        self.worker.finished_acquire_measure.connect(self.display_measurement_by_slide)
        self.worker.finished_interpolated_scene.connect(self.display_panchrom_display)
        self.worker.start()

    def on_acquisition_saved(self):

        self.result_directory = initialize_directory(self.acquisition_config_editor.get_config())

        self.system_editor_config = self.system_editor.get_config()
        with open(self.result_directory + "/config_system.yml", 'w') as file:
            yaml.safe_dump(self.system_editor_config, file)

        with open(self.result_directory + "/config_acquisition.yml", 'w') as file:
            yaml.safe_dump(self.acquisition_config_editor.get_config(), file)

        if self.last_measurement_3D is not None:
            last_measurement = self.last_measurement_3D

            # Calculate the other two arrays
            sum_last_measurement = np.sum(last_measurement, axis=2)
            sum_scene_interpolated = np.sum(self.interpolated_scene, axis=2)

            # Save the arrays in an H5 file
            with h5py.File(self.result_directory + '/filtered_image.h5', 'w') as f:
                f.create_dataset('filtered_image', data=last_measurement)
            with h5py.File(self.result_directory + '/image.h5', 'w') as f:
                f.create_dataset('image', data=sum_last_measurement)
            with h5py.File(self.result_directory + '/panchro.h5', 'w') as f:
                f.create_dataset('panchro', data=sum_scene_interpolated)
            with h5py.File(self.result_directory + '/filtering_cube.h5', 'w') as f:
                f.create_dataset('filtering_cube', data=self.filtering_widget.filtering_cube)
            with h5py.File(self.result_directory + '/wavelengths.h5', 'w') as f:
                f.create_dataset('wavelengths', data=self.filtering_widget.list_wavelengths)

            print("Measurement saved")
        else:
            print("No measurement to save")

    @pyqtSlot(np.ndarray)
    def display_acquisition(self, measurement_3D):

        self.acquisition_display.display_acquisition(measurement_3D)

    def display_measurement_by_slide(self, measurement_3D):
        self.last_measurement_3D = measurement_3D
        self.acquisition_by_slice_display.display_measurement_slice_by_slice(measurement_3D)
    def display_panchrom_display(self, scene):
        self.interpolated_scene = scene
        self.acquisition_panchro_display.display_panchrom_acquisition(scene)


