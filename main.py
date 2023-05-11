import sys
import yaml
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QDockWidget,QHBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QWidget, QFormLayout, QScrollArea, QGroupBox,QRadioButton, QButtonGroup,QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from Dimensionner import Dimensionner

from PyQt5.QtWidgets import QVBoxLayout, QToolBar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib
import numpy as np



class EditorSystemConfig(QWidget):

    def __init__(self):
        super().__init__()

        self.init_ui()
        self.load_config("init_config.yml")  # Add this line


    def init_ui(self):
        # self.setWindowTitle('Editor for system config')

        # Create layout and widgets
        main_layout = QVBoxLayout()
        buttons_layout = QHBoxLayout()

        open_button = QPushButton('Open')
        open_button.clicked.connect(self.open_config)
        save_button = QPushButton('Save')
        save_button.clicked.connect(self.save_config)
        buttons_layout.addWidget(open_button)
        buttons_layout.addWidget(save_button)

        self.group_layout = QVBoxLayout()

        scroll_widget = QWidget()
        scroll_widget.setLayout(self.group_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)

        main_layout.addLayout(buttons_layout)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)  # Set the layout for the current widget


    def load_config(self, file_name):
        if file_name:
            with open(file_name, 'r') as file:
                self.config = yaml.safe_load(file)
                self.show_config()

    def open_config(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Config", "", "YAML Files (*.yml *.yaml);;All Files (*)",
                                                   options=options)

        if file_name:
            self.load_config(file_name)


    def show_config(self):
        self.input_fields = {}
        self.input_labels = {}
        self.input_groups = {}
        for key, section in self.config.items():
            group_box = QGroupBox(key)
            form_layout = QFormLayout()
            group_box.setLayout(form_layout)

            for sub_key, value in section.items():
                full_key = f"{key}_{sub_key}"
                label = QLabel(sub_key)

                if sub_key in ['dispersive_element_1', 'dispersive_element_2']:
                    sub_group_box = QGroupBox(sub_key)
                    sub_form_layout = QFormLayout()
                    sub_group_box.setLayout(sub_form_layout)
                    self.input_groups[full_key] = sub_group_box  # Save a reference to the QGroupBox

                    self.input_fields[sub_key] = {}  # Store the fields in a separate dictionary
                    self.input_labels[sub_key] = {}  # Store the labels in a separate dictionary

                    for nested_sub_key, nested_value in value.items():
                        nested_full_key = f"{full_key}_{nested_sub_key}"
                        nested_label = QLabel(nested_sub_key)

                        # input_field = QLineEdit(str(nested_value))
                        # sub_form_layout.addRow(nested_label, input_field)
                        # self.input_fields[sub_key][nested_full_key] = input_field  # Store the field in the new dictionary
                        # self.input_labels[sub_key][nested_full_key] = nested_label  # Store the label in the new dictionary

                        if nested_sub_key == "type":
                            input_field = QComboBox(self)
                            input_field.addItem("prism")
                            input_field.addItem("grating")
                            input_field.setCurrentText(nested_value)
                            input_field.currentTextChanged.connect(
                                lambda value, full_key=full_key: self.toggle_dispersive_element_fields(value, full_key))
                        else:
                            input_field = QLineEdit(str(nested_value))
                        sub_form_layout.addRow(nested_label, input_field)
                        self.input_fields[nested_full_key] = input_field

                    form_layout.addRow(sub_group_box)

                else:
                    full_key = f"{key}_{sub_key}"
                    label = QLabel(sub_key)

                    if full_key == "system architecture_type":
                        input_field = QComboBox(self)
                        input_field.addItem("SD-CASSI")
                        input_field.addItem("DD-CASSI")
                        input_field.setCurrentText(value)
                        input_field.currentTextChanged.connect(self.toggle_system_architecture_fields)
                        form_layout.addRow(QLabel(sub_key), input_field)
                        self.input_fields[full_key] = input_field  # This line is added to store the QComboBox in input_fields

                    else:
                        label = QLabel(sub_key)
                        input_field = QLineEdit(str(value))
                        form_layout.addRow(label, input_field)
                        self.input_fields[full_key] = input_field
                        self.input_labels[full_key] = label

            self.group_layout.addWidget(group_box)

            # Simulate the change events
            if key == 'system architecture':
                system_architecture_type = section.get('type')
                if system_architecture_type:
                    self.toggle_system_architecture_fields(system_architecture_type)

                dispersive_element_1_type = section.get('dispersive_element_1', {}).get('type')
                if dispersive_element_1_type:
                    self.toggle_dispersive_element_fields(dispersive_element_1_type, f"{key}_dispersive_element_1")

                dispersive_element_2_type = section.get('dispersive_element_2', {}).get('type')
                if dispersive_element_2_type:
                    self.toggle_dispersive_element_fields(dispersive_element_2_type, f"{key}_dispersive_element_2")

    def toggle_fields(self, keys, enable_fields, input_fields_dict, labels_dict):
        for key in keys:
            if key in input_fields_dict:
                input_fields_dict[key].setEnabled(enable_fields)
                if enable_fields:
                    input_fields_dict[key].show()
                else:
                    input_fields_dict[key].hide()

            if key in labels_dict:  # Add this check to prevent KeyError
                labels_dict[key].setVisible(enable_fields)

    # Modify this method
    def toggle_system_architecture_fields(self, system_type):
        lens_keys = ["system architecture_focal_lens_3", "system architecture_focal_lens_4"]
        dispersive_element_2_keys = [f"system architecture_dispersive_element_2_{key}" for key in
                                     self.config['system architecture']['dispersive_element_2'].keys()]

        enable_fields = system_type == "DD-CASSI"
        self.toggle_fields(lens_keys, enable_fields, self.input_fields, self.input_labels)
        self.toggle_fields(dispersive_element_2_keys, enable_fields, self.input_fields, self.input_labels)



        # If system name is SD-CASSI, set all dispersive_element_2 values to None
        if system_type == "SD-CASSI":
            self.input_groups["system architecture_dispersive_element_2"].hide()
        else:
            self.input_groups["system architecture_dispersive_element_2"].show()  # Add this line

    def toggle_dispersive_element_fields(self, dispersive_element_type, dispersive_element_key):
        prism_keys = [f"{dispersive_element_key}_A"]
        grating_keys = [f"{dispersive_element_key}_m", f"{dispersive_element_key}_G"]

        if dispersive_element_type == "prism":
            self.toggle_fields(prism_keys, True, self.input_fields, self.input_labels)
            self.toggle_fields(grating_keys, False, self.input_fields, self.input_labels)
        else:
            self.toggle_fields(prism_keys, False, self.input_fields, self.input_labels)
            self.toggle_fields(grating_keys, True, self.input_fields, self.input_labels)


    def save_config(self):
        if hasattr(self, 'config'):
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "YAML Files (*.yml *.yaml);;All Files (*)", options=options)

            if file_name:
                self.update_config()
                with open(file_name, 'w') as file:
                    yaml.dump(self.config, file)
        else:
            pass

    def get_config(self):
        config = {}  # Create an empty dictionary
        print(self.input_fields.items())
        # Loop over all input fields and add their values to the config
        for key, input_field in self.input_fields.items():
            # Split the key into section and sub_key
            section, sub_key = key.split("_", 1)
            if section not in config:
                config[section] = {}

            # Get the value from the input field
            if isinstance(input_field, QLineEdit):
                value = input_field.text()
            elif isinstance(input_field, QComboBox):
                value = input_field.currentText()
            else:
                continue

            # If the value can be converted to a number, do so
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

            # Add the value to the config
            config[section][sub_key] = value

        return config

    def update_config(self):
        for key, section in self.config.items():
            for sub_key, value in section.items():
                if isinstance(value, dict):
                    for nested_sub_key, _ in value.items():
                        full_key = f"{key}_{sub_key}_{nested_sub_key}"
                        if full_key in self.input_fields:
                            input_field = self.input_fields[full_key]
                            if input_field.isHidden():
                                new_value = None
                            else:
                                if isinstance(input_field, QComboBox):
                                    new_value = input_field.currentText()  # Changed line
                                else:
                                    new_value = input_field.text()
                                try:
                                    new_value = int(new_value)
                                except ValueError:
                                    try:
                                        new_value = float(new_value)
                                    except ValueError:
                                        pass
                            self.config[key][sub_key][nested_sub_key] = new_value
                else:
                    full_key = f"{key}_{sub_key}"
                    if full_key in self.input_fields:
                        if full_key == "system_architecture_name" or full_key == "FOV_field_of_view_mode":
                            input_field = self.input_fields[full_key]
                            new_value = input_field.currentText()
                        else:
                            input_field = self.input_fields[full_key]
                            if input_field.isHidden():
                                new_value = None
                            else:
                                if isinstance(input_field, QComboBox):
                                    new_value = input_field.currentText()  # Changed line
                                else:
                                    new_value = input_field.text()
                                try:
                                    new_value = int(new_value)
                                except ValueError:
                                    try:
                                        new_value = float(new_value)
                                    except ValueError:
                                        pass
                        self.config[key][sub_key] = new_value


class ResultDisplay(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.figure_cam = plt.figure()
        self.canvas_cam = FigureCanvas(self.figure_cam)
        self.toolbar_cam = NavigationToolbar(self.canvas_cam, self)

        self.figure_dmd = plt.figure()
        self.canvas_dmd = FigureCanvas(self.figure_dmd)
        self.toolbar_dmd = NavigationToolbar(self.canvas_dmd, self)

        layout.addWidget(self.toolbar_cam)
        layout.addWidget(self.canvas_cam)
        layout.addWidget(self.toolbar_dmd)
        layout.addWidget(self.canvas_dmd)

    def display_results_cam(self, X_cam, Y_cam):
        self.figure_cam.clear()

        ax = self.figure_cam.add_subplot(111)
        scatter = ax.scatter(X_cam, Y_cam)

        # Set labels with LaTeX font.
        ax.set_xlabel(f'X_cam', fontsize=12)
        ax.set_ylabel(f'Y_cam', fontsize=12)
        ax.set_title(f'Camera sampling', fontsize=12)

        self.canvas_cam.draw()


    def display_results_dmd(self, X_dmd, Y_dmd):
        self.figure_dmd.clear()

        ax = self.figure_dmd.add_subplot(111)

        # Draw a grid using the plot method
        for i in range(X_dmd.shape[0]):
            ax.plot(X_dmd[i, :], Y_dmd[i, :], color='k')

        for j in range(X_dmd.shape[1]):
            ax.plot(X_dmd[:, j], Y_dmd[:, j], color='k')

        # Set labels with LaTeX font.
        ax.set_xlabel(f'X_dmd', fontsize=12)
        ax.set_ylabel(f'Y_dmd', fontsize=12)
        ax.set_title(f'DMD sampling', fontsize=12)


        self.canvas_dmd.draw()

class Worker(QThread):
    finished_cam = pyqtSignal(tuple)  # For camera sampling results
    finished_dmd = pyqtSignal(tuple)  # For DMD sampling results

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        # Put your analysis here
        dimensioner = Dimensionner(config=self.config)

        X_cam, Y_cam = dimensioner.define_camera_sampling()
        self.finished_cam.emit((X_cam, Y_cam))  # Emit a tuple of arrays
        # X_dmd, Y_dmd = dimensioner.define_DMD_sampling()
        X_dmd, Y_dmd = dimensioner.retropropagate()
        self.finished_dmd.emit((X_dmd, Y_dmd))



class MainWindow(QMainWindow):
    # This is your new main window class

    def __init__(self):
        super().__init__()

        self.setWindowTitle('App')

        # Create the ConfigEditor and add it as a dock widget
        self.editor_system_config = EditorSystemConfig()
        self.config_dock = QDockWidget("Editor for system config")
        self.config_dock.setWidget(self.editor_system_config)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.config_dock)

        # Create the ResultDisplay and set it as the central widget
        self.result_display = ResultDisplay()
        self.setCentralWidget(self.result_display)

        # Create the "Run Dimensioning" button and add it to the toolbar
        run_button = QPushButton('Run Dimensioning')
        run_button.clicked.connect(self.run_dimensioning)
        self.toolbar = self.addToolBar("Run Dimensioning")
        self.toolbar.addWidget(run_button)

    def run_dimensioning(self):
        # Get the config from the editor
        config = self.editor_system_config.get_config()

        import pprint
        pprint.pprint(config)

        # Create a new worker thread and start it
        self.worker = Worker(config)
        self.worker.finished_cam.connect(self.display_results_cam)
        self.worker.finished_dmd.connect(self.display_results_dmd)
        self.worker.start()
    @pyqtSlot(tuple)
    def display_results_cam(self, arrays):
        X_cam, Y_cam = arrays  # Unpack the tuple
        self.result_display.display_results_cam(X_cam, Y_cam)

    @pyqtSlot(tuple)
    def display_results_dmd(self, arrays):
        X_dmd, Y_dmd = arrays  # Unpack the tuple
        self.result_display.display_results_dmd(X_dmd, Y_dmd)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


