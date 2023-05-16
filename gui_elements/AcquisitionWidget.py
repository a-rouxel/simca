
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
        print(self.data)
        # Display the slice
        self.imageView.setImage(np.rot90(self.data[:, :, slice_index]))

class AcquisitionDisplay(QWidget):
    def __init__(self):
        super().__init__()

        # Create a QVBoxLayout for the widget
        self.layout = QVBoxLayout(self)

        # Create a figure and a canvas
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)



    def display_acquisition(self, measurement_3D):
        """
        Display the ground truth with each color corresponding to a label_value.

        :param ground_truth: np.array, ground truth labels
        :param label_values: list, label_values[i] = name of the class i
        :param palette: color palette to use, must be a list of colors where the index corresponds to the label
        :param ignored_labels: list of ignored labels (pixel with no label)
        """
        # Clear the figure
        self.figure.clear()

        # Create an axes
        ax = self.figure.add_subplot(111)
        print(measurement_3D)

        imshow = ax.imshow(np.sum(measurement_3D,axis=2), cmap='gray')

        cbar = self.figure.colorbar(imshow, ax=ax)  # Use the new figure for the colorbar

        # Redraw the canvas
        self.canvas.draw()



class AcquisitionEditorWidget(QWidget):

    def __init__(self,initial_config=None):
        super().__init__()

        self.initial_config_file = initial_config

        # Create a QScrollArea
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        # Create a widget for the scroll area
        scroll_widget = QWidget()

        self.acquisitions_directory = QLineEdit()

        self.directories_combo = QComboBox()
        self.acquisition_types = ['DD-CASSI', 'SD-CASSI']
        self.directories_combo.addItems(self.acquisition_types)

        # Add the dimensioning configuration editor, the result display widget, and the run button to the layout

        acquisition_layout = QFormLayout()
        acquisition_layout.addRow("acquisition directory", self.acquisitions_directory)
        acquisition_layout.addRow("acquisition type", self.directories_combo)

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
        self.acquisitions_directory.setText(self.config['acquisition directory'])
        # self.directories_combo.addItem(self.config['acquisition type'])

    def get_config(self):
        return {
            "acquisition directory": self.acquisitions_directory.text(),
            "acquisition type": self.directories_combo.text(),
        }

class Worker(QThread):
    finished_acquire_measure = pyqtSignal(np.ndarray)


    def __init__(self,filtering_widget, scene_widget,acquisition_config_editor):
        super().__init__()

        self.filtering_widget = filtering_widget
        self.scene_widget = scene_widget

    def run(self):

        filtering_cube = self.filtering_widget.filtering_cube
        scene = self.scene_widget.scene




        if filtering_cube.shape[0] != scene.shape[0] or filtering_cube.shape[1] != scene.shape[1] :
                scene = scene[0:filtering_cube.shape[0], 0:filtering_cube.shape[1], :]
                print("Filtering cube and scene must have the same lines and columns")

                if  filtering_cube.shape[2] != scene.shape[2]:
                    scene = scene[:, :, 0:filtering_cube.shape[2]]
                    print("Filtering cube and scene must have the same number of wavelengths")

        measurement_in_3D = filtering_cube * scene
        measurement_in_3D = np.nan_to_num(measurement_in_3D)

        self.finished_acquire_measure.emit(measurement_in_3D)  # Emit a tuple of arrays

class AcquisitionWidget(QWidget):
    def __init__(self,scene_widget, filtering_widget,acquisition_config_path="config/acquisition.yml"):
        super().__init__()

        self.scene_widget = scene_widget
        self.filtering_widget = filtering_widget


        self.layout = QHBoxLayout()

        if acquisition_config_path is not None:
            self.acquisition_config_editor = AcquisitionEditorWidget(acquisition_config_path)

        self.layout.addWidget(self.acquisition_config_editor)

        self.result_display_widget = QTabWidget()

        self.acquisition_display = AcquisitionDisplay()
        self.acquisition_by_slice_display = AcquisitionSlideBySlideWidget()

        self.result_display_widget.addTab(self.acquisition_display, "Measured Image")
        self.result_display_widget.addTab(self.acquisition_by_slice_display, "Measured Image for each spectral sample")

        self.run_button = QPushButton('Run Acquisition')
        self.run_button.setStyleSheet('QPushButton {background-color: black; color: white;}')        # Connect the button to the run_dimensioning method
        self.run_button.clicked.connect(self.run_acquisition)

        # Create a group box for the run button
        self.run_button_group_box = QGroupBox()
        run_button_group_layout = QVBoxLayout()

        run_button_group_layout.addWidget(self.run_button)
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
        self.worker.start()


    @pyqtSlot(np.ndarray)
    def display_acquisition(self, measurement_3D):
        self.acquisition_display.display_acquisition(measurement_3D)

    def display_measurement_by_slide(self, measurement_3D):
        self.acquisition_by_slice_display.display_measurement_slice_by_slice(measurement_3D)


