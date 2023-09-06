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
from simca import CassiSystem



class CodedApertureGridDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.figure_cam = plt.figure()
        self.canvas_cam = FigureCanvas(self.figure_cam)
        self.toolbar_cam = NavigationToolbar(self.canvas_cam, self)

        self.layout.addWidget(self.toolbar_cam)
        self.layout.addWidget(self.canvas_cam)

    def display_pattern_grid(self, pattern):
        self.figure_cam.clear()

        ax = self.figure_cam.add_subplot(111)
        scatter = ax.imshow(pattern,cmap="viridis",interpolation='none',vmin=0,vmax=1)

        # Set labels with LaTeX font.
        ax.set_xlabel(f'X coded aperture grid [um]', fontsize=12)
        ax.set_ylabel(f'Y coded aperture grid[um]', fontsize=12)
        ax.set_title(f'coded aperture Grid', fontsize=12)

        self.canvas_cam.draw()






class PropagatedcodedapertureGridDisplay(QWidget):
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


    def display_propagated_pattern_grid(self, filtering_cube,list_wavelengths):
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
    finished_define_pattern = pyqtSignal(np.ndarray)
    finished_propagate_pattern = pyqtSignal(np.ndarray,np.ndarray)

    def __init__(self, cassi_system,system_config,simulation_config):
        super().__init__()
        self.cassi_system = cassi_system
        self.system_config = system_config
        self.simulation_config = simulation_config

    def run(self):
        # Put your analysis here
        self.cassi_system.update_config(system_config=self.system_config)
        self.cassi_system.propagate_coded_aperture_grid()
        self.cassi_system.generate_filtering_cube()
        self.finished_propagate_pattern.emit(self.cassi_system.filtering_cube,self.cassi_system.optical_model.system_wavelengths)  # Emit a tuple of arrays


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

        self.pattern_type = QComboBox()
        self.pattern_type.addItems(["slit","random","blue-noise type 1","custom h5 pattern"])
        self.pattern_type.currentTextChanged.connect(self.on_pattern_type_changed)

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
        self.general_layout.addRow("pattern type", self.pattern_type)
        self.general_layout.addRow("slit position", self.slit_position_slider)
        self.general_layout.addRow("slit width", self.slit_width_slider)



        general_group = QGroupBox("Settings")
        general_group.setLayout(self.general_layout)


        # Load configs button
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
        if hasattr(self, 'configs'):
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Config", "",
                                                       "YAML Files (*.yml *.yaml);;All Files (*)", options=options)
            if file_name:
                # Update the configs from the current input fields
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
        # Call a method to update the GUI with the loaded configs
        self.update_config()

    def update_config(self):
        # This method should update your QLineEdit and QSpinBox widgets with the loaded configs.

        self.results_directory.setText(self.config['infos']['results directory'])
        self.pattern_type.setCurrentText(self.config['pattern']['type'])


        if self.config['pattern']['type'] == 'slit':
            self.slit_position_slider.setValue(self.config['pattern']['slit position'])
            self.slit_width_slider.setValue(self.config['pattern']['slit width'])
        elif self.config['pattern']['type'] == 'random':
            self.random_ROM.setText(str(self.config['pattern']['ROM']))



    def on_pattern_type_changed(self, pattern_type):



        for i in range(self.general_layout.rowCount() - 1, 0, -1):  # start from last row, stop at 2 (exclusive), step backwards
            # Remove row at index i from the layout
            self.general_layout.removeRow(i)

        if pattern_type == "random":
            self.random_ROM = QLineEdit()
            self.random_ROM.setText("0.5")
            self.general_layout.addRow("ROM", self.random_ROM)

        if pattern_type == "slit":
            self.slit_position_slider = QSlider(Qt.Horizontal)
            self.slit_position_slider.setMinimum(-200)
            self.slit_position_slider.setMaximum(200)  # Adjust as needed
            self.slit_position_slider.valueChanged.connect(self.on_slit_position_changed)

            self.slit_width_slider = QSlider(Qt.Horizontal)
            self.slit_width_slider.setMinimum(1)
            self.slit_width_slider.setMaximum(30)  # Adjust as needed
            self.slit_width_slider.valueChanged.connect(self.on_slit_width_changed)

            self.general_layout.addRow("slit position", self.slit_position_slider)
            self.general_layout.addRow("slit width", self.slit_width_slider)


        if pattern_type == "custom h5 pattern":

            self.file_path = QLineEdit()

            self.browse_button = QPushButton("Browse")
            self.browse_button.clicked.connect(self.browse_file)

            self.general_layout.addRow("File Path", self.file_path)
            self.general_layout.addRow("", self.browse_button)


    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select h5 file", "", "H5 Files (*.h5)")
        if file_path:
            self.file_path.setText(file_path)
    def on_slit_position_changed(self, position):
        # Update the slit position in your configs
        self.config['pattern']['slit position'] = position

    def on_slit_width_changed(self, width):
        # Update the slit width in your configs
        self.config['pattern']['slit width'] = width

    def get_config(self):
        config = {
            "pattern": {
                "type": self.pattern_type.currentText()
            },
        }

        if config['pattern']['type'] == "slit":
            config['pattern']['slit position'] = self.slit_position_slider.value()
            config['pattern']['slit width'] = self.slit_width_slider.value()

        elif config['pattern']['type'] == "custom h5 pattern":
            config['pattern']['file path'] = self.file_path.text()
        elif config['pattern']['type'] == "random":
            config['pattern']['ROM'] = float(self.random_ROM.text())
        else :
            config['pattern']['slit position'] = None
            config['pattern']['slit width'] = None

        return config

class FilteringCubeWidget(QWidget):

    patternGenerated = pyqtSignal(np.ndarray)
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
        self.camera_result_display = CodedApertureGridDisplay()
        self.propagated_pattern_display = PropagatedcodedapertureGridDisplay()


        # Add the result displays to the tab widget
        self.result_display_widget.addTab(self.camera_result_display, "Patterns")
        self.result_display_widget.addTab(self.propagated_pattern_display, "Filtering cube, slide by slide")

        # Create the generate pattern button
        self.generate_pattern_button = QPushButton('Generate pattern')
        self.generate_pattern_button.clicked.connect(self.generate_pattern)


        # Create the run button
        self.run_button = QPushButton('Generate Filtering Cube')
        # self.run_button.setStyleSheet('QPushButton {background-color: red; color: white;}')        # Connect the button to the run_dimensioning method
        self.run_button.clicked.connect(self.run_dimensioning)

        # Create a group box for the run button
        self.run_button_group_box = QGroupBox()
        run_button_group_layout = QVBoxLayout()

        run_button_group_layout.addWidget(self.generate_pattern_button)
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



    def run_dimensioning(self):
        # Get the configs from the editors

        system_config = self.system_editor.get_config()
        config_filtering = self.filtering_config_editor.get_config()
        self.run_button.setDisabled(True)

        self.worker = Worker(self.cassi_system,system_config, config_filtering)
        self.worker.finished_define_pattern.connect(self.display_pattern_grid)
        self.worker.finished_propagate_pattern.connect(self.display_propagated_patterns)
        self.worker.start()

    def generate_pattern(self):
        system_config = self.system_editor.get_config()
        config_filtering = self.filtering_config_editor.get_config()

        self.patternGenerated.connect(self.display_pattern_grid)

        self.cassi_system.update_config(system_config=system_config)
        self.cassi_system.generate_2D_pattern(config_filtering)

        self.run_button.setEnabled(True)
        self.patternGenerated.emit(self.cassi_system.pattern)

    @pyqtSlot(np.ndarray)
    def display_pattern_grid(self, pattern):
        self.camera_result_display.display_pattern_grid(pattern)

    @pyqtSlot(np.ndarray,np.ndarray)
    def display_propagated_patterns(self, filtering_cube,np_of_wavelengths):
        self.cassi_system.filtering_cube = filtering_cube
        self.cassi_system.optical_model.system_wavelengths = np_of_wavelengths
        self.propagated_pattern_display.display_propagated_pattern_grid(filtering_cube,np_of_wavelengths)
