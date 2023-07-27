from PyQt5.QtWidgets import (QTabWidget,QHBoxLayout, QPushButton, QFileDialog,
                             QLineEdit, QComboBox,QFormLayout, QGroupBox, QScrollArea,
                             QVBoxLayout, QSlider, QLabel, QWidget)
from PyQt5.QtCore import Qt,QThread, pyqtSignal, pyqtSlot

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import pyqtgraph as pg
import numpy as np
import yaml
from cassi_systems import CassiSystem



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
        self.label.setText("wavelength: " + str(int(self.list_wavelengths[slice_index])) +" nm")

        # Display the slice
        self.imageView.setImage(np.rot90(self.data[:,:,slice_index]), levels=(0, 255))

class Worker(QThread):
    finished_define_mask_grid = pyqtSignal(np.ndarray)
    finished_propagate_mask_grid = pyqtSignal(np.ndarray,np.ndarray)

    def __init__(self, cassi_system,system_config,simulation_config):
        super().__init__()
        self.cassi_system = cassi_system
        self.system_config = system_config
        self.simulation_config = simulation_config

    def run(self):
        # Put your analysis here
        self.cassi_system.update_config(self.system_config)
        self.cassi_system.propagate_mask_grid()
        self.cassi_system.generate_filtering_cube()
        self.finished_propagate_mask_grid.emit(self.cassi_system.filtering_cube,self.cassi_system.system_wavelengths)  # Emit a tuple of arrays


class FilteringCubeWidgetEditor(QWidget):
    def __init__(self,filtering_widget,initial_config_file=None):
        super().__init__()

        self.filtering_widget = filtering_widget
        self.initial_config_file = initial_config_file


        # Create a QScrollArea
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        # Create a widget for the scroll area
        scroll_widget = QWidget()

        # Add your dimensioning parameters here
        self.results_directory = QLineEdit()

        self.mask_type = QComboBox()
        self.mask_type.addItems(["slit","random","blue","custom h5 mask"])
        self.mask_type.currentTextChanged.connect(self.on_mask_type_changed)

        self.slit_position_slider = QSlider(Qt.Horizontal)
        self.slit_position_slider.setMinimum(-200)
        self.slit_position_slider.setMaximum(200)  # Adjust as needed
        self.slit_position_slider.valueChanged.connect(self.on_slit_position_changed)

        self.slit_width_slider = QSlider(Qt.Horizontal)
        self.slit_width_slider.setMinimum(1)
        self.slit_width_slider.setMaximum(30)  # Adjust as needed
        self.slit_width_slider.valueChanged.connect(self.on_slit_width_changed)

        self.random_ROM = QLineEdit()
        self.random_ROM.setText("0.5")



        self.general_layout = QFormLayout()
        self.general_layout.addRow("mask type", self.mask_type)
        self.general_layout.addRow("slit position", self.slit_position_slider)
        self.general_layout.addRow("slit width", self.slit_width_slider)



        general_group = QGroupBox("Settings")
        general_group.setLayout(self.general_layout)


        # Load config button
        self.load_config_button = QPushButton("Load Config")
        self.load_config_button.clicked.connect(self.on_load_config_clicked)
        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.clicked.connect(self.save_config)

        # Create main layout and add widgets
        main_layout = QVBoxLayout()
        main_layout.addWidget(general_group)
        main_layout.addWidget(self.load_config_button)
        main_layout.addWidget(self.save_config_button)

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

    def save_config(self):
        if hasattr(self, 'config'):
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Config", "",
                                                       "YAML Files (*.yml *.yaml);;All Files (*)", options=options)
            if file_name:
                # Update the config from the current input fields
                self.config = self.get_config()
                with open(file_name, 'w') as file:
                    yaml.safe_dump(self.config, file, default_flow_style=False)


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
        self.mask_type.setCurrentText(self.config['mask']['type'])


        if self.config['mask']['type'] == 'slit':
            self.slit_position_slider.setValue(self.config['mask']['slit position'])
            self.slit_width_slider.setValue(self.config['mask']['slit width'])
        elif self.config['mask']['type'] == 'random':
            self.random_ROM.setText(str(self.config['mask']['ROM']))



    def on_mask_type_changed(self, mask_type):

        self.filtering_widget.enable_generate_mask_button()

        for i in range(self.general_layout.rowCount() - 1, 0, -1):  # start from last row, stop at 2 (exclusive), step backwards
            # Remove row at index i from the layout
            self.general_layout.removeRow(i)

        if mask_type == "random":
            self.random_ROM = QLineEdit()
            self.random_ROM.setText("0.5")
            self.general_layout.addRow("ROM", self.random_ROM)

            self.random_ROM.textChanged.connect(self.filtering_widget.enable_generate_mask_button)
            self.random_ROM.textChanged.connect(self.filtering_widget.enable_generate_mask_button)

        if mask_type == "slit":
            self.slit_position_slider = QSlider(Qt.Horizontal)
            self.slit_position_slider.setMinimum(-200)
            self.slit_position_slider.setMaximum(200)  # Adjust as needed
            self.slit_position_slider.valueChanged.connect(self.on_slit_position_changed)

            self.slit_width_slider = QSlider(Qt.Horizontal)
            self.slit_width_slider.setMinimum(1)
            self.slit_width_slider.setMaximum(30)  # Adjust as needed
            self.slit_width_slider.valueChanged.connect(self.on_slit_width_changed)

            self.slit_position_slider.valueChanged.connect(self.filtering_widget.enable_generate_mask_button)
            self.slit_width_slider.valueChanged.connect(self.filtering_widget.enable_generate_mask_button)
            self.general_layout.addRow("slit position", self.slit_position_slider)
            self.general_layout.addRow("slit width", self.slit_width_slider)


        if mask_type == "custom h5 mask":

            self.file_path = QLineEdit()

            self.browse_button = QPushButton("Browse")
            self.browse_button.clicked.connect(self.browse_file)

            self.file_path.textChanged.connect(self.filtering_widget.enable_generate_mask_button)
            self.general_layout.addRow("File Path", self.file_path)
            self.general_layout.addRow("", self.browse_button)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select h5 file", "", "H5 Files (*.h5)")
        if file_path:
            self.file_path.setText(file_path)
    def on_slit_position_changed(self, position):
        # Update the slit position in your config
        self.config['mask']['slit position'] = position

    def on_slit_width_changed(self, width):
        # Update the slit width in your config
        self.config['mask']['slit width'] = width

    def get_config(self):
        config = {
            "mask": {
                "type": self.mask_type.currentText()
            },
        }

        if config['mask']['type'] == "slit":
            config['mask']['slit position'] = self.slit_position_slider.value()
            config['mask']['slit width'] = self.slit_width_slider.value()

        elif config['mask']['type'] == "custom h5 mask":
            config['mask']['file path'] = self.file_path.text()
        elif config['mask']['type'] == "random":
            config['mask']['ROM'] = float(self.random_ROM.text())
        else :
            config['mask']['slit position'] = None
            config['mask']['slit width'] = None

        return config

class FilteringCubeWidget(QWidget):

    maskGenerated = pyqtSignal(np.ndarray)
    def __init__(self, cassi_system=None,system_editor=None,filtering_config_path=None):
        super().__init__()

        self.cassi_system = cassi_system
        self.system_editor = system_editor


        self.layout = QHBoxLayout()

        # Create the dimensioning configuration editor
        self.filtering_config_editor = FilteringCubeWidgetEditor(self,initial_config_file=filtering_config_path)

        # Create the result display widget (tab widget in this case)
        self.result_display_widget = QTabWidget()

        # Create the result displays and store them as attributes
        self.camera_result_display = MaskGridDisplay()
        self.propagated_mask_display = PropagatedMaskGridDisplay()


        # Add the result displays to the tab widget
        self.result_display_widget.addTab(self.camera_result_display, "Mask Grid")
        self.result_display_widget.addTab(self.propagated_mask_display, "Filtering cube, slide by slide")

        # Create the generate mask button
        self.generate_mask_button = QPushButton('Generate mask')
        # self.generate_mask_button.setStyleSheet('QPushButton {background-color: red; color: white;}')        # Connect the button to the run_dimensioning method
        self.generate_mask_button.clicked.connect(self.generate_mask)


        # Create the run button
        self.run_button = QPushButton('Generate Filtering Cube')
        # self.run_button.setStyleSheet('QPushButton {background-color: red; color: white;}')        # Connect the button to the run_dimensioning method
        self.run_button.clicked.connect(self.run_dimensioning)

        # Create a group box for the run button
        self.run_button_group_box = QGroupBox()
        run_button_group_layout = QVBoxLayout()

        run_button_group_layout.addWidget(self.generate_mask_button)
        run_button_group_layout.addWidget(self.run_button)
        run_button_group_layout.addWidget(self.result_display_widget)

        self.run_button_group_box.setLayout(run_button_group_layout)



        # Add the dimensioning configuration editor, the result display widget, and the run button to the layout
        self.layout.addWidget(self.filtering_config_editor)
        self.layout.addWidget(self.run_button_group_box)

        self.layout.setStretchFactor(self.run_button_group_box, 1)
        self.layout.setStretchFactor(self.result_display_widget, 2)

        # Set the layout on the widget
        self.setLayout(self.layout)


    def enable_generate_mask_button(self):
        # Enable the "Generate Mask" button
        self.generate_mask_button.setEnabled(True)
        self.run_button.setEnabled(True)


    def run_dimensioning(self):
        # Get the configs from the editors

        system_config = self.system_editor.get_config()
        config_filtering = self.filtering_config_editor.get_config()
        self.run_button.setDisabled(True)

        self.worker = Worker(self.cassi_system,system_config, config_filtering)
        self.worker.finished_define_mask_grid.connect(self.display_mask_grid)
        self.worker.finished_propagate_mask_grid.connect(self.display_propagated_masks)
        self.worker.start()

    def generate_mask(self):
        system_config = self.system_editor.get_config()
        config_filtering = self.filtering_config_editor.get_config()

        self.maskGenerated.connect(self.display_mask_grid)

        self.cassi_system.update_config(system_config)
        self.cassi_system.generate_2D_mask(config_filtering)

        self.generate_mask_button.setDisabled(True)
        self.maskGenerated.emit(self.cassi_system.mask)

    @pyqtSlot(np.ndarray)
    def display_mask_grid(self, mask):
        self.camera_result_display.display_mask_grid(mask)

    @pyqtSlot(np.ndarray,np.ndarray)
    def display_propagated_masks(self, filtering_cube,np_of_wavelengths):
        self.cassi_system.filtering_cube = filtering_cube
        self.cassi_system.system_wavelengths = np_of_wavelengths
        self.propagated_mask_display.display_propagated_mask_grid(filtering_cube,np_of_wavelengths)
