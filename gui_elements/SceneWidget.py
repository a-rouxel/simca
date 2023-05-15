import yaml
from PyQt5.QtWidgets import (QTabWidget, QSpinBox,QHBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QWidget, QFormLayout, QScrollArea, QGroupBox,QRadioButton, QButtonGroup,QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot,QCoreApplication
from PyQt5.QtWidgets import  QVBoxLayout
import os
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from utils.scenes_helper import *

class SceneContentDisplay(QWidget):

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.figure_scene = plt.figure()
        self.canvas_scene = FigureCanvas(self.figure_scene)
        self.toolbar_scene = NavigationToolbar(self.canvas_scene, self)

        self.layout.addWidget(self.toolbar_scene)
        self.layout.addWidget(self.canvas_scene)

    def diplay_scene_content(self, scene):

        self.figure_scene.clear()

        ax = self.figure_scene.add_subplot(111)
        # scatter = ax.scatter(X_cam, Y_cam)

        # Set labels with LaTeX font.
        ax.set_xlabel(f'X input grid [um]', fontsize=12)
        ax.set_ylabel(f'Y input grid [um]', fontsize=12)
        ax.set_title(f'Input Grid', fontsize=12)

        self.canvas_scene.draw()



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

        try:
            img, gt, label_values, ignored_labels, rgb_bands, palette, delta_lambda = get_dataset(self.scene_config_editor.directories_combo.currentText(), self.scene_config_editor.scenes_directory.text())
            self.scene = img
            self.scene_gt = gt
            self.scene_label_values = label_values
            self.scene_ignored_labels = ignored_labels
            self.scene_rgb_bands = rgb_bands
            self.scene_palette = palette
            self.scene_delta_lambda = delta_lambda
        except:
            print("Error: scene not found")


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

# class Worker(QThread):
#     finished_load_scene = pyqtSignal(np.ndarray)
#
#     def __init__(self,scene_config):
#         super().__init__()
#
#         self.scene_config = scene_config
#
#     def run(self):
#
#         scene = load_scene(self.scene_config)
#
#         self.finished_load_scene.emit(scene)  # Emit a tuple of arrays

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
        self.layout.addWidget(self.result_display_widget)

        self.setLayout(self.layout)
    #
    # def run_load_scene(self):
    #     # Get the configs from the editors
    #
    #     self.worker = Worker()
    #     self.worker.finished_load_scene.connect(self.diplay_scene_content)
    #     self.worker.start()

    @pyqtSlot(np.ndarray)
    def display_scene_content(self, scene):
        self.scene_content_display.diplay_scene_content(scene)
