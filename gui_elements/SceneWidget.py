import yaml
from PyQt5.QtWidgets import (QTabWidget, QSpinBox,QHBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QWidget, QFormLayout, QScrollArea, QGroupBox,QRadioButton, QButtonGroup,QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot,QCoreApplication
from PyQt5.QtWidgets import  QVBoxLayout, QSlider
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from utils.scenes_helper import *

class SceneContentDisplay(QWidget):

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

    def diplay_scene_content(self, scene, list_wavelengths):

        self.list_wavelengths = list_wavelengths

        # Normalize filtering_cube values to range [0, 255] for ImageView
        self.data = (scene - np.min(scene)) / (np.max(scene) - np.min(scene)) * 255
        self.data = self.data.astype(np.uint8)

        # Set slider maximum value
        self.slider.setMaximum(self.data.shape[2] - 1)

        # Display the first slice
        self.update_image(0)

    def update_image(self, slice_index):
        # Update the label
        self.label.setText("wavelength: " + str(int(self.list_wavelengths[slice_index])) + " nm")

        # Display the slice
        self.imageView.setImage(np.rot90(self.data[:, :, slice_index]), levels=(0, 255))




class SceneConfigEditor(QWidget):
    def __init__(self,initial_config_file=None):
        super().__init__()
        self.initial_config_file = initial_config_file

        # Create a QScrollArea
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        # Create a widget for the scroll area
        scroll_widget = QWidget()

        self.scenes_directory = QLineEdit()
        self.directories_combo = QComboBox()

        self.scene_directory_button = QPushButton("Load Scene")
        self.scene_directory_button.clicked.connect(self.load_scene)

        # Add the dimensioning configuration editor, the result display widget, and the run button to the layout
        scene_layout = QFormLayout()
        scene_layout.addRow("scene directory", self.scenes_directory)
        scene_layout.addRow("avalaible scenes", self.directories_combo)

        scene_group = QGroupBox("Scene settings")
        scene_group.setLayout(scene_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scene_directory_button)
        main_layout.addWidget(scene_group)

        # Set the layout on the widget within the scroll area
        scroll_widget.setLayout(main_layout)

        # Set the widget for the scroll area
        scroll.setWidget(scroll_widget)

        # Create a layout for the current widget and add the scroll area
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

        # Load the initial configuration file if one was provided
        if self.initial_config_file is not None:
            self.load_config(initial_config_file)

        self.load_scenes()


    def load_scene(self):


            img, gt, list_wavelengths, label_values, ignored_labels, rgb_bands, palette, delta_lambda = get_dataset(self.directories_combo.currentText(), self.scenes_directory.text())
            self.scene = img
            self.scene_gt = gt
            self.list_wavelengths = list_wavelengths
            self.scene_label_values = label_values
            self.scene_ignored_labels = ignored_labels
            self.scene_rgb_bands = rgb_bands
            self.scene_palette = palette
            self.scene_delta_lambda = delta_lambda
            #
            # print("Error: scene not found")


    def load_scenes(self):
        scene_dir = self.scenes_directory.text()
        if os.path.isdir(scene_dir):
            sub_dirs = [name for name in os.listdir(scene_dir)
                        if os.path.isdir(os.path.join(scene_dir, name))]
            self.directories_combo.clear()
            self.directories_combo.addItems(sub_dirs)

    def load_config(self, file_name):
        with open(file_name, 'r') as file:
            self.config = yaml.safe_load(file)
        # Call a method to update the GUI with the loaded config
        self.update_config()
    def update_config(self):
        # This method should update your QLineEdit and QSpinBox widgets with the loaded config.
        self.scenes_directory.setText(self.config['scenes directory'])

    def get_config(self):
        return {
            "scenes directory": self.scene_directory.value()
        }

class Worker(QThread):
    finished_load_scene = pyqtSignal(np.ndarray,list)

    def __init__(self,scene_config_editor):
        super().__init__()

        self.scene_config_editor = scene_config_editor

    def run(self):

        self.scene_config_editor.load_scene()

        self.finished_load_scene.emit(self.scene_config_editor.scene,self.scene_config_editor.list_wavelengths)  # Emit a tuple of arrays

class SceneWidget(QWidget):
    def __init__(self,scene_config_path="config/scene.yml"):
        super().__init__()

        self.layout = QHBoxLayout()

        if scene_config_path is not None:
            self.scene_config_editor = SceneConfigEditor(scene_config_path)

        self.layout.addWidget(self.scene_config_editor)

        self.result_display_widget = QTabWidget()

        self.scene_content_display = SceneContentDisplay()

        self.result_display_widget.addTab(self.scene_content_display, "Scene Content")


        self.run_button = QPushButton('Load Scene')
        self.run_button.setStyleSheet('QPushButton {background-color: red; color: white;}')        # Connect the button to the run_dimensioning method
        self.run_button.clicked.connect(self.run_load_scene)

        # Create a group box for the run button
        self.run_button_group_box = QGroupBox()
        run_button_group_layout = QVBoxLayout()

        run_button_group_layout.addWidget(self.run_button)
        run_button_group_layout.addWidget(self.result_display_widget)

        self.run_button_group_box.setLayout(run_button_group_layout)

        self.layout.addWidget(self.run_button_group_box)
        self.setLayout(self.layout)
    #
    def run_load_scene(self):
        # Get the configs from the editors

        self.worker = Worker(self.scene_config_editor)
        self.worker.finished_load_scene.connect(self.display_scene_content)
        self.worker.start()

    @pyqtSlot(np.ndarray,list)
    def display_scene_content(self, scene,list_wavelengths):
        self.scene_content_display.diplay_scene_content(scene,list_wavelengths)
