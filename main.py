import sys
import yaml
from PyQt5.QtWidgets import (QApplication, QGridLayout,QMainWindow, QSpinBox,QVBoxLayout, QDockWidget,QHBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QWidget, QFormLayout, QScrollArea, QGroupBox,QRadioButton, QButtonGroup,QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize

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
        # buttons_layout = QHBoxLayout()
        #
        # open_button = QPushButton('Open')
        # open_button.clicked.connect(self.open_config)
        # save_button = QPushButton('Save')
        # save_button.clicked.connect(self.save_config)
        # buttons_layout.addWidget(open_button)
        # buttons_layout.addWidget(save_button)

        self.group_layout = QVBoxLayout()

        scroll_widget = QWidget()
        scroll_widget.setLayout(self.group_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)

        # main_layout.addLayout(buttons_layout)
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


    # def save_config(self):
    #     if hasattr(self, 'config'):
    #         options = QFileDialog.Options()
    #         file_name, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "YAML Files (*.yml *.yaml);;All Files (*)", options=options)
    #
    #         system_config = self.get_config()
    #         self.config = system_config
    #         if file_name:
    #             # self.update_config()
    #             with open(file_name, 'w') as file:
    #                 yaml.dump(self.config, file)
    #     else:
    #         pass

    def get_config(self):
        config = {}  # Create an empty dictionary
        # print(self.input_fields.items())
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

    # def update_config(self):
    #     for key, section in self.config.items():
    #         for sub_key, value in section.items():
    #             if isinstance(value, dict):
    #                 for nested_sub_key, _ in value.items():
    #                     full_key = f"{key}_{sub_key}_{nested_sub_key}"
    #                     if full_key in self.input_fields:
    #                         input_field = self.input_fields[full_key]
    #                         if input_field.isHidden():
    #                             new_value = None
    #                         else:
    #                             if isinstance(input_field, QComboBox):
    #                                 new_value = input_field.currentText()  # Changed line
    #                             else:
    #                                 new_value = input_field.text()
    #                             try:
    #                                 new_value = int(new_value)
    #                             except ValueError:
    #                                 try:
    #                                     new_value = float(new_value)
    #                                 except ValueError:
    #                                     pass
    #                         self.config[key][sub_key][nested_sub_key] = new_value
    #             else:
    #                 full_key = f"{key}_{sub_key}"
    #                 if full_key in self.input_fields:
    #                     if full_key == "system_architecture_name" or full_key == "FOV_field_of_view_mode":
    #                         input_field = self.input_fields[full_key]
    #                         new_value = input_field.currentText()
    #                     else:
    #                         input_field = self.input_fields[full_key]
    #                         if input_field.isHidden():
    #                             new_value = None
    #                         else:
    #                             if isinstance(input_field, QComboBox):
    #                                 new_value = input_field.currentText()  # Changed line
    #                             else:
    #                                 new_value = input_field.text()
    #                             try:
    #                                 new_value = int(new_value)
    #                             except ValueError:
    #                                 try:
    #                                     new_value = float(new_value)
    #                                 except ValueError:
    #                                     pass
    #                     self.config[key][sub_key] = new_value

class SquareLayout(QVBoxLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setGeometry(self, rect):
        square_size = min(rect.width(), rect.height())
        rect.setSize(QSize(square_size, square_size))
        super().setGeometry(rect)
class CameraResultDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = SquareLayout()
        self.setLayout(self.layout)

        self.figure_cam = plt.figure()
        self.canvas_cam = FigureCanvas(self.figure_cam)
        self.toolbar_cam = NavigationToolbar(self.canvas_cam, self)

        self.layout.addWidget(self.toolbar_cam)
        self.layout.addWidget(self.canvas_cam)

    def display_results_cam(self, X_cam, Y_cam):
        self.figure_cam.clear()

        ax = self.figure_cam.add_subplot(111)
        scatter = ax.scatter(X_cam, Y_cam)

        # Set labels with LaTeX font.
        ax.set_xlabel(f'X_cam', fontsize=12)
        ax.set_ylabel(f'Y_cam', fontsize=12)
        ax.set_title(f'Camera Pixelization', fontsize=12)

        self.canvas_cam.draw()

class DMDResultDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = SquareLayout()
        self.setLayout(self.layout)

        self.figure_dmd = plt.figure()
        self.canvas_dmd = FigureCanvas(self.figure_dmd)
        self.toolbar_dmd = NavigationToolbar(self.canvas_dmd, self)

        self.layout .addWidget(self.toolbar_dmd)
        self.layout .addWidget(self.canvas_dmd)

    def display_results_dmd(self, list_X_dmd, list_Y_dmd,list_wavelengths):

        self.figure_dmd.clear()

        ax = self.figure_dmd.add_subplot(111)

        for idx in range(len(list_X_dmd)):

           if idx ==0 or idx == len(list_X_dmd)//2  or idx == len(list_X_dmd)-1:
            X_dmd = list_X_dmd[idx]
            Y_dmd = list_Y_dmd[idx]
            wavelength = list_wavelengths[idx]

            X_dmd = X_dmd.reshape(-1,1)
            Y_dmd = Y_dmd.reshape(-1,1)


            scatter = ax.scatter(X_dmd, Y_dmd,label=f'{int(wavelength[0,0])} nm')


        ax.set_xlabel(f'X_dmd', fontsize=12)
        ax.set_ylabel(f'Y_dmd', fontsize=12)
        ax.set_title(f'Retropropagation to DMD -- mapping', fontsize=12)
        ax.legend()

        self.canvas_dmd.draw()

class DispersionResultDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = SquareLayout()
        self.setLayout(self.layout)

        self.figure_dispersion= plt.figure()
        self.canvas_dispersion = FigureCanvas(self.figure_dispersion)
        self.toolbar_dispersion = NavigationToolbar(self.canvas_dispersion, self)

        self.layout.addWidget(self.toolbar_dispersion)
        self.layout.addWidget(self.canvas_dispersion)

    def display_results_dispersion(self, list_X_dmd, list_Y_dmd,list_wavelengths):
        self.figure_dispersion.clear()

        ax_center = self.figure_dispersion.add_subplot(111)

        list_wavelength_center = list()
        list_X_dmd_center = list()
        list_Y_dmd_center = list()
        list_X_dmd_corner_hleft = list()
        list_Y_dmd_corner_hleft = list()
        list_X_dmd_corner_lright = list()
        list_Y_dmd_corner_lright = list()



        for wavelength_array in list_wavelengths:
            list_wavelength_center.append(wavelength_array[0,0])

        for idx in range(len(list_X_dmd)):
            X_dmd = list_X_dmd[idx]
            Y_dmd = list_Y_dmd[idx]

            X_dmd_center = X_dmd[X_dmd.shape[0]//2,X_dmd.shape[1]//2]
            Y_dmd_center = Y_dmd[X_dmd.shape[0]//2,X_dmd.shape[1]//2]

            X_dmd_corner_hleft = X_dmd[0,0]
            Y_dmd_corner_hleft = Y_dmd[0,0]

            X_dmd_corner_lright = X_dmd[X_dmd.shape[0]-1,X_dmd.shape[1]-1]
            Y_dmd_corner_lright = Y_dmd[X_dmd.shape[0]-1,X_dmd.shape[1]-1]

            list_X_dmd_center.append(X_dmd_center)
            list_Y_dmd_center.append(Y_dmd_center)
            list_X_dmd_corner_hleft.append(X_dmd_corner_hleft)
            list_Y_dmd_corner_hleft.append(Y_dmd_corner_hleft)
            list_X_dmd_corner_lright.append(X_dmd_corner_lright)
            list_Y_dmd_corner_lright.append(Y_dmd_corner_lright)

        for idx in range(len(list_X_dmd_center)):
            list_X_dmd_corner_hleft[idx] = list_X_dmd_corner_hleft[idx] + list_X_dmd_center[len(list_X_dmd_center)//2]
            list_Y_dmd_corner_hleft[idx] = list_Y_dmd_corner_hleft[idx] + list_Y_dmd_center[len(list_Y_dmd_center)//2]
            list_X_dmd_corner_lright[idx] = list_X_dmd_corner_lright[idx] + list_X_dmd_center[len(list_X_dmd_center)//2]
            list_Y_dmd_corner_lright[idx] = list_Y_dmd_corner_lright[idx] + list_Y_dmd_center[len(list_Y_dmd_center)//2]

        plot = ax_center.plot(list_X_dmd_center, list_wavelength_center, label='X_cam center')
        ax_center.legend()
        # plot = ax_center.plot(list_X_dmd_corner_hleft, list_wavelength_center, label='Corner up left')
        # plot = ax_center.plot(list_X_dmd_corner_lright, list_wavelength_center,label='Corner up left')

        # Set labels with LaTeX font.
        ax_center.set_xlabel(f'x_dmd', fontsize=12)
        ax_center.set_ylabel(f'Wavelength', fontsize=12)
        ax_center.set_title(f'Spectral Dispersion', fontsize=12)

        self.canvas_dispersion.draw()

class DistorsionResultDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = SquareLayout()
        self.setLayout(self.layout)

        self.figure_distorsion= plt.figure()
        self.canvas_distorsion = FigureCanvas(self.figure_distorsion)
        self.toolbar_distorsion = NavigationToolbar(self.canvas_distorsion, self)

        self.layout.addWidget(self.toolbar_distorsion)
        self.layout.addWidget(self.canvas_distorsion)

    def display_results_distorsion(self, X_cam, Y_cam,list_X_dmd, list_Y_dmd,list_wavelengths):

        self.figure_distorsion.clear()

        ax = self.figure_distorsion.add_subplot(111)

        for idx in range(len(list_X_dmd)):

           if idx ==0 :
            X_dmd = list_X_dmd[idx]
            Y_dmd = list_Y_dmd[idx]
            wavelength = list_wavelengths[idx]

            print(X_dmd[X_dmd.shape[0]//2,X_dmd.shape[1]//2])
            print(X_dmd[X_dmd.shape[0]//2,X_dmd.shape[1]//2],X_cam[X_dmd.shape[0]//2,X_dmd.shape[1]//2])

            X_ref = -1*X_cam + X_dmd[X_dmd.shape[0]//2,Y_dmd.shape[1]//2]
            Y_ref = -1*Y_cam + Y_dmd[Y_dmd.shape[0]//2,Y_dmd.shape[1]//2]

            dist = np.sqrt((X_dmd-X_ref)**2 + (Y_dmd-Y_ref)**2)

        print(X_dmd[X_dmd.shape[0]//2,X_dmd.shape[1]//2],Y_dmd[X_dmd.shape[0]//2,X_dmd.shape[1]//2],X_ref[X_dmd.shape[0]//2,X_dmd.shape[1]//2],Y_ref[X_dmd.shape[0]//2,X_dmd.shape[1]//2],dist[X_dmd.shape[0]//2,X_dmd.shape[1]//2])

        imshow = ax.imshow(dist,extent=[X_dmd.min(),X_dmd.max(),Y_dmd.min(),Y_dmd.max()],cmap='viridis')

        cbar = self.figure_distorsion.colorbar(imshow, ax=ax)
        cbar.set_label('Distorsion')

        ax.set_xlabel(f'X_dmd', fontsize=12)
        ax.set_ylabel(f'Y_dmd', fontsize=12)
        ax.set_title(f'Distorsion map', fontsize=12)
        ax.legend()

        self.canvas_distorsion.draw()

class Worker(QThread):
    finished_cam = pyqtSignal(tuple)
    finished_dmd = pyqtSignal(tuple)
    finished_dispersion = pyqtSignal(tuple)
    finished_distorsion = pyqtSignal(tuple)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        # Put your analysis here
        dimensioner = Dimensionner(config=self.config)

        X_cam, Y_cam = dimensioner.define_camera_sampling()
        self.finished_cam.emit((X_cam, Y_cam))  # Emit a tuple of arrays
        # X_dmd, Y_dmd = dimensioner.define_DMD_sampling()
        list_X_dmd, list_Y_dmd, list_wavelengths = dimensioner.propagate()
        self.finished_dmd.emit((list_X_dmd, list_Y_dmd,list_wavelengths))
        self.finished_dispersion.emit((list_X_dmd, list_Y_dmd, list_wavelengths))

        self.finished_distorsion.emit((X_cam, Y_cam,list_X_dmd, list_Y_dmd, list_wavelengths))


class DimensioningConfigEditor(QWidget):
    # A new class for editing dimensioning parameters
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()

        # Add your dimensioning parameters here
        self.results_directory = QLineEdit()
        self.spectral_samples = QSpinBox()


        layout.addWidget(QLabel("results directory"))
        layout.addWidget(self.results_directory)
        layout.addWidget(QLabel("number of spectral samples"))
        layout.addWidget(self.spectral_samples)

        self.setLayout(layout)

    def get_config(self):
        # This method returns the current settings
        return {
            'results_directory': self.results_directory.text(),
            'spectral_samples': self.spectral_samples.value(),
            # add more key-value pairs for other parameters...
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('SIMCA -- Dimensioning -- V0.1')

        self.editor_system_config = EditorSystemConfig()
        self.system_config_dock = QDockWidget("Editor for system config")
        self.system_config_dock.setWidget(self.editor_system_config)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.system_config_dock)

        self.dimensioning_config_editor = DimensioningConfigEditor()
        self.dimensioning_config_dock = QDockWidget("Dimensioning Config")
        self.dimensioning_config_dock.setWidget(self.dimensioning_config_editor)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dimensioning_config_dock)

        # Create the ResultDisplay widgets
        self.camera_result_display = CameraResultDisplay()
        self.dmd_result_display = DMDResultDisplay()
        self.dispersion_result_display = DispersionResultDisplay()
        self.distorsion_result_display = DistorsionResultDisplay()

        # Create a widget and grid layout to hold the result widgets
        self.result_widget = QWidget()
        self.result_layout = QGridLayout(self.result_widget)

        # Add the result widgets to the grid layout
        self.result_layout.addWidget(self.camera_result_display, 0, 0)
        self.result_layout.addWidget(self.dmd_result_display, 0, 1)
        self.result_layout.addWidget(self.dispersion_result_display, 1, 0)
        self.result_layout.addWidget(self.distorsion_result_display, 1, 1)

        # Set the grid layout as the central widget
        self.setCentralWidget(self.result_widget)

        run_button = QPushButton('Run Dimensioning')
        run_button.clicked.connect(self.run_dimensioning)
        self.toolbar = self.addToolBar("Run Dimensioning")
        self.toolbar.addWidget(run_button)
    def run_dimensioning(self):
        # Get the configs from the editors
        system_config = self.editor_system_config.get_config()
        dimensioning_config = self.dimensioning_config_editor.get_config()

        config = {**system_config, **dimensioning_config}  # Merge the configs

        # pprint.pprint(config)

        self.worker = Worker(config)
        self.worker.finished_cam.connect(self.display_results_cam)
        self.worker.finished_dmd.connect(self.display_results_dmd)
        self.worker.finished_dispersion.connect(self.display_results_dispersion)
        self.worker.finished_distorsion.connect(self.display_results_distorsion)
        self.worker.start()

    @pyqtSlot(tuple)
    def display_results_cam(self, arrays_cam):
        X_cam, Y_cam = arrays_cam  # Unpack the tuple
        self.camera_result_display.display_results_cam(X_cam, Y_cam)

    @pyqtSlot(tuple)
    def display_results_dmd(self, arrays_dmd):
        list_X_dmd, list_Y_dmd, list_wavelengths = arrays_dmd  # Unpack the tuple
        self.dmd_result_display.display_results_dmd(list_X_dmd, list_Y_dmd, list_wavelengths)

    @pyqtSlot(tuple)
    def display_results_dispersion(self, arrays_dmd):
        list_X_dmd, list_Y_dmd, list_wavelengths = arrays_dmd  # Unpack the tuple
        self.dispersion_result_display.display_results_dispersion(list_X_dmd, list_Y_dmd, list_wavelengths)

    @pyqtSlot(tuple)
    def display_results_distorsion(self, arrays_dmd_and_cam):
        X_cam, Y_cam,list_X_dmd, list_Y_dmd, list_wavelengths = arrays_dmd_and_cam  # Unpack the tuple
        self.distorsion_result_display.display_results_distorsion(X_cam, Y_cam,list_X_dmd, list_Y_dmd, list_wavelengths)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


