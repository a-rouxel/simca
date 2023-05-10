import sys
import yaml
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QWidget, QFormLayout, QScrollArea, QGroupBox)
from PyQt5.QtWidgets import QRadioButton, QButtonGroup,QComboBox


class ConfigEditor(QMainWindow):

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Config Editor')

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

        main_widget = QWidget()
        main_layout.addLayout(buttons_layout)
        main_layout.addWidget(scroll_area)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def open_config(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Config", "", "YAML Files (*.yml *.yaml);;All Files (*)", options=options)

        if file_name:
            with open(file_name, 'r') as file:
                self.config = yaml.safe_load(file)
                self.show_config()
                self.toggle_fov_fields(str(self.config['FOV']['field_of_view_mode']))

    def show_config(self):
        self.input_fields = {}
        for key, section in self.config.items():
            group_box = QGroupBox(key)
            form_layout = QFormLayout()
            group_box.setLayout(form_layout)

            for sub_key, value in section.items():
                if isinstance(value, dict):
                    for nested_sub_key, nested_value in value.items():
                        full_key = f"{key}_{sub_key}_{nested_sub_key}"
                        label = QLabel(f"{sub_key} - {nested_sub_key}")

                        input_field = QLineEdit(str(nested_value))
                        form_layout.addRow(label, input_field)
                        self.input_fields[full_key] = input_field
                else:
                    full_key = f"{key}_{sub_key}"
                    label = QLabel(sub_key)

                    if full_key == "FOV_field_of_view_mode":
                        input_field = QComboBox(self)
                        input_field.addItem("0")
                        input_field.addItem("1")
                        input_field.setCurrentText(str(value))
                        input_field.currentTextChanged.connect(self.toggle_fov_fields)
                        form_layout.addRow(QLabel(sub_key), input_field)

                    elif full_key == "system_architecture_name":
                        input_field = QComboBox(self)
                        input_field.addItem("SD-CASSI")
                        input_field.addItem("DD-CASSI")
                        input_field.setCurrentText(value)
                        input_field.currentTextChanged.connect(self.toggle_system_architecture_fields)
                        form_layout.addRow(QLabel(sub_key), input_field)
                    else:
                        label = QLabel(sub_key)
                        input_field = QLineEdit(str(value))
                        if key == "FOV" and self.config["FOV"]["field_of_view_mode"] == 1:
                            input_field.setEnabled(False)
                        form_layout.addRow(label, input_field)
                        self.input_fields[full_key] = input_field

            self.group_layout.addWidget(group_box)


    def toggle_system_architecture_fields(self, system_name):
        lens_keys = ["system_architecture_lens_focal_length_3", "system_architecture_lens_focal_length_4"]
        enable_fields = system_name == "DD-CASSI"
        for key in lens_keys:
            if key in self.input_fields:
                self.input_fields[key].setEnabled(enable_fields)

    def toggle_fov_fields(self, mode):
        fov_keys = ["FOV_FOV_ACT", "FOV_FOV_ALT", "FOV_FOV_center_ACT", "FOV_FOV_center_ALT"]
        enable_fields = mode == "0"
        for key in fov_keys:
            if key in self.input_fields:
                self.input_fields[key].setEnabled(enable_fields)

    def fov_mode_changed(self, button):
        mode = button.group().checkedId()
        for sub_key in ["FOV_ACT", "FOV_ALT", "FOV_center_ACT", "FOV_center_ALT"]:
            full_key = f"FOV_{sub_key}"
            input_field = self.input_fields.get(full_key, None)
            if input_field:
                input_field.setEnabled(mode == 0)
        self.config["FOV"]["field_of_view_mode"] = mode

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

    def update_config(self):
        for key, section in self.config.items():
            for sub_key, value in section.items():
                if isinstance(value, dict):
                    for nested_sub_key, _ in value.items():
                        full_key = f"{key}_{sub_key}_{nested_sub_key}"
                        if full_key in self.input_fields:
                            input_field = self.input_fields[full_key]
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
                            new_value = input_field.text()
                            try:
                                new_value = int(new_value)
                            except ValueError:
                                try:
                                    new_value = float(new_value)
                                except ValueError:
                                    pass
                        self.config[key][sub_key] = new_value



if __name__ == '__main__':
    app = QApplication(sys.argv)
    config_editor = ConfigEditor()
    config_editor.show()
    sys.exit(app.exec_())
