import yaml
from PyQt5.QtWidgets import (QTabWidget, QSpinBox,QHBoxLayout, QPushButton,
                             QFileDialog, QLineEdit, QWidget,
                             QFormLayout, QVBoxLayout, QGroupBox, QScrollArea)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

from CassiSystem import CassiSystem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.helpers import *

class InputGridDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.figure_cam = plt.figure()
        self.canvas_cam = FigureCanvas(self.figure_cam)
        self.toolbar_cam = NavigationToolbar(self.canvas_cam, self)

        self.layout.addWidget(self.toolbar_cam)
        self.layout.addWidget(self.canvas_cam)

    def display_mask_grid(self, X_cam, Y_cam):

        self.figure_cam.clear()

        X_cam = undersample_grid(X_cam)
        Y_cam = undersample_grid(Y_cam)

        ax = self.figure_cam.add_subplot(111)
        ax.scatter(X_cam, Y_cam)

        # Set labels with LaTeX font.
        ax.set_xlabel(f'X input grid [um]', fontsize=12)
        ax.set_ylabel(f'Y input grid [um]', fontsize=12)
        ax.set_title(f'Mask Grid', fontsize=12)

        self.canvas_cam.draw()

class InputGridPropagationDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.figure_dmd = plt.figure()
        self.canvas_dmd = FigureCanvas(self.figure_dmd)
        self.toolbar_dmd = NavigationToolbar(self.canvas_dmd, self)

        self.layout .addWidget(self.toolbar_dmd)
        self.layout .addWidget(self.canvas_dmd)



    def display_mask_propagation(self, list_X_detector, list_Y_detector, list_wavelengths):

        self.figure_dmd.clear()

        ax = self.figure_dmd.add_subplot(111)

        # Define a color palette with Seaborn
        colors = sns.color_palette("husl", 3)

        for idx in range(len(list_X_detector)):

            if idx == 0:
                color = colors[2]  # light blue
            elif idx == len(list_X_detector) // 2:
                color = colors[1]  # light green
            else:
                color = colors[0]  # light red

            if idx == 0 or idx == len(list_X_detector) // 2 or idx == len(list_X_detector) - 1:

                wavelength = list_wavelengths[idx]
                X_detector = undersample_grid(list_X_detector[idx])
                Y_detector = undersample_grid(list_Y_detector[idx])

                X_detector = X_detector.reshape(-1, 1)
                Y_detector = Y_detector.reshape(-1, 1)


                ax.scatter(X_detector, Y_detector, alpha= 0.5,color=color, label=f'{int(wavelength[0, 0])} nm')

        ax.set_xlabel(f'X image plane [um]', fontsize=12)
        ax.set_ylabel(f'Y image plane [um]', fontsize=12)
        ax.set_title(f'Propagated Grids', fontsize=12)
        ax.legend()

        self.canvas_dmd.draw()

class DistorsionResultDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.figure_distorsion= plt.figure()
        self.canvas_distorsion = FigureCanvas(self.figure_distorsion)
        self.toolbar_distorsion = NavigationToolbar(self.canvas_distorsion, self)

        self.layout.addWidget(self.toolbar_distorsion)
        self.layout.addWidget(self.canvas_distorsion)

    def display_results_distorsion(self, X_input_grid, Y_input_grid, list_X_detector, list_Y_detector,
                                   list_wavelengths):

        self.figure_distorsion.clear()

        fig, axs = plt.subplots(3, 1)  # Create a new figure with 3 subplots
        self.figure_distorsion = fig  # Replace the old figure with the new one

        selected_indices = [0, len(list_X_detector) // 2, len(list_X_detector) - 1]

        for i, idx in enumerate(selected_indices):
            # ax = axs[i,0]
            X_input_grid_subsampled = undersample_grid(X_input_grid)
            Y_input_grid_subsampled = undersample_grid(Y_input_grid)

            X_detector = undersample_grid(list_X_detector[idx])
            Y_detector = undersample_grid(list_Y_detector[idx])
            wavelength = list_wavelengths[idx]

            X_ref = -1 * X_input_grid_subsampled + X_detector[X_detector.shape[0] // 2, X_detector.shape[1] // 2]
            Y_ref = -1 * Y_input_grid_subsampled + Y_detector[Y_detector.shape[0] // 2, Y_detector.shape[1] // 2]

            dist = np.sqrt((X_detector - X_ref) ** 2 + (Y_detector - Y_ref) ** 2)


            ax = axs[i]

            scatter_new = ax.scatter(X_detector, Y_detector, c=dist, cmap='viridis',label=f'accurate model')
            cbar = fig.colorbar(scatter_new, ax=ax)  # Use the new figure for the colorbar
            cbar.set_label(f'Distorsion at {int(wavelength[0, 0])} nm [um]')
            scatter_trad = ax.scatter(X_ref, Y_ref,alpha=0.1,label='classical model')

            ax.legend()

        # Update the canvas with the new figure
        self.canvas_distorsion.figure = self.figure_distorsion
        self.canvas_distorsion.draw()


class Worker(QThread):
    finished_define_mask_grid = pyqtSignal(tuple)
    finished_propagate_mask_grid = pyqtSignal(tuple)
    finished_distorsion = pyqtSignal(tuple)

    def __init__(self, cassi_system,system_config):
        super().__init__()
        self.cassi_system = cassi_system
        self.system_config = system_config


    def run(self):
        # Put your analysis here
        self.cassi_system.update_config(self.system_config)

        cassi_system = self.cassi_system

        self.finished_define_mask_grid.emit((cassi_system.X_dmd_grid, cassi_system.Y_dmd_grid))  # Emit a tuple of arrays

        list_X_detector, list_Y_detector, list_wavelengths = cassi_system.propagate_mask_grid([self.system_config["spectral range"]["wavelength min"],
                                                                                               self.system_config["spectral range"]["wavelength max"]],
                                                                                              self.system_config["spectral range"]["number of spectral samples"])
        self.finished_propagate_mask_grid.emit((list_X_detector, list_Y_detector,list_wavelengths))


        self.finished_distorsion.emit((cassi_system.X_dmd_grid, cassi_system.Y_dmd_grid,list_X_detector, list_Y_detector, list_wavelengths))



class OpticsConfigEditor(QWidget):
    def __init__(self,initial_config_file=None):
        super().__init__()

        self.initial_config_file = initial_config_file
        # Create a QScrollArea
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        # Create a widget for the scroll area
        scroll_widget = QWidget()

        # Add your optics parameters here
        self.results_directory = QLineEdit()

        self.spectral_samples = QSpinBox()
        self.spectral_samples.setMinimum(1)
        self.spectral_samples.setMaximum(500)

        self.wavelength_min = QSpinBox()
        self.wavelength_min.setMinimum(200)
        self.wavelength_min.setMaximum(5000)

        self.wavelength_max = QSpinBox()
        self.wavelength_max.setMinimum(200)
        self.wavelength_max.setMaximum(5000)

        self.sampling_across_X = QSpinBox()
        self.sampling_across_Y = QSpinBox()

        self.delta_X = QLineEdit()
        self.delta_Y = QLineEdit()

        # Create form layout for wavelengths and add widgets
        wavelength_layout = QFormLayout()
        wavelength_layout.addRow("Wavelength min", self.wavelength_min)
        wavelength_layout.addRow("Wavelength max", self.wavelength_max)

        # Create form layout for sampling and deltas and add widgets
        sampling_layout = QFormLayout()
        sampling_layout.addRow("Sampling across X", self.sampling_across_X)
        sampling_layout.addRow("Sampling across Y", self.sampling_across_Y)
        sampling_layout.addRow("Delta X", self.delta_X)
        sampling_layout.addRow("Delta Y", self.delta_Y)

        general_layout = QFormLayout()
        general_layout.addRow("results directory", self.results_directory)
        general_layout.addRow("spectral samples", self.spectral_samples)

        # Create group boxes and set layouts
        wavelength_group = QGroupBox("Spectral Range")
        wavelength_group.setLayout(wavelength_layout)

        sampling_group = QGroupBox("Input Grid Sampling")
        sampling_group.setLayout(sampling_layout)



        general_group = QGroupBox("General settings")
        general_group.setLayout(general_layout)

        # Load config button
        self.load_config_button = QPushButton("Load Config")
        self.load_config_button.clicked.connect(self.on_load_config_clicked)

        # Create main layout and add widgets
        main_layout = QVBoxLayout()
        main_layout.addWidget(wavelength_group)
        main_layout.addWidget(sampling_group)
        main_layout.addWidget(general_group)
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

        self.wavelength_min.setValue(self.config['spectral range']['wavelength min'])
        self.wavelength_max.setValue(self.config['spectral range']['wavelength max'])

        self.sampling_across_X.setValue(self.config['input grid sampling']['sampling across X'])
        self.sampling_across_Y.setValue(self.config['input grid sampling']['sampling across Y'])
        self.delta_X.setText(str(self.config['input grid sampling']['delta X']))
        self.delta_Y.setText(str(self.config['input grid sampling']['delta Y']))

        self.spectral_samples.setValue(self.config['number of spectral samples'])

    def get_config(self):
        return {
            "infos": {
                "results directory": self.results_directory.text()
            },
            "spectral range": {
                "wavelength min": self.wavelength_min.value(),
                "wavelength max": self.wavelength_max.value()
            },
            "input grid sampling": {
                "sampling across X": self.sampling_across_X.value(),
                "sampling across Y": self.sampling_across_Y.value(),
                "delta X": int(self.delta_X.text()),
                "delta Y": int(self.delta_Y.text())
            },
            "number of spectral samples": self.spectral_samples.value()
        }

class OpticsWidget(QWidget):
    def __init__(self, cassi_system=None,editor_system_config=None,optics_config_path=None):
        super().__init__()

        self.cassi_system = cassi_system
        self.editor_system_config = editor_system_config

        self.layout = QHBoxLayout()

        # Create the result display widget (tab widget in this case)
        self.result_display_widget = QTabWidget()

        # Create the result displays and store them as attributes
        self.camera_result_display = InputGridDisplay()
        self.dmd_result_display = InputGridPropagationDisplay()
        self.distorsion_result_display = DistorsionResultDisplay()

        # Add the result displays to the tab widget
        self.result_display_widget.addTab(self.camera_result_display, "Mask Grid")
        self.result_display_widget.addTab(self.dmd_result_display, "Propagated Grids")
        self.result_display_widget.addTab(self.distorsion_result_display, "Distortion Maps")

        # Create the run button
        self.run_button = QPushButton('Run Simulation')
        self.run_button.setStyleSheet('QPushButton {background-color: blue; color: white;}')        # Connect the button to the run_optics method
        self.run_button.clicked.connect(self.run_optics)

        # Create a group box for the run button
        self.run_button_group_box = QGroupBox()
        run_button_group_layout = QVBoxLayout()

        run_button_group_layout.addWidget(self.run_button)
        run_button_group_layout.addWidget(self.result_display_widget)

        self.run_button_group_box.setLayout(run_button_group_layout)

        # Add the optics configuration editor, the result display widget, and the run button to the layout
        # self.layout.addWidget(self.optics_config_editor)
        self.layout.addWidget(self.run_button_group_box)

        self.layout.setStretchFactor(self.run_button_group_box, 1)
        self.layout.setStretchFactor(self.result_display_widget, 2)

        # Set the layout on the widget
        self.setLayout(self.layout)


    def run_optics(self):
        # Get the configs from the editors


        system_config = self.editor_system_config.get_config()
        cassi_system = self.cassi_system
        # optics_config = self.optics_config_editor.get_config()

        self.worker = Worker(cassi_system,system_config)
        self.worker.finished_define_mask_grid.connect(self.display_mask_grid)
        self.worker.finished_propagate_mask_grid.connect(self.display_mask_propagation)
        self.worker.finished_distorsion.connect(self.display_results_distorsion)
        self.worker.start()



    @pyqtSlot(tuple)
    def display_mask_grid(self, arrays_input_grid):
        X_input_grid, Y_input_grid = arrays_input_grid  # Unpack the tuple
        self.camera_result_display.display_mask_grid(X_input_grid, Y_input_grid)

    @pyqtSlot(tuple)
    def display_mask_propagation(self, arrays_propagated_grid):
        list_X_detector, list_Y_detector, list_wavelengths = arrays_propagated_grid  # Unpack the tuple
        self.dmd_result_display.display_mask_propagation(list_X_detector, list_Y_detector, list_wavelengths)
    @pyqtSlot(tuple)
    def display_results_distorsion(self, arrays_dmd_and_cam):
        X_cam, Y_cam,list_X_detector, list_Y_detector, list_wavelengths = arrays_dmd_and_cam  # Unpack the tuple
        self.distorsion_result_display.display_results_distorsion(X_cam, Y_cam,list_X_detector, list_Y_detector, list_wavelengths)
