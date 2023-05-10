import sys
import yaml
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QWidget, QFormLayout, QScrollArea, QGroupBox)
from PyQt5.QtWidgets import QRadioButton, QButtonGroup


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

    def show_config(self):
        self.input_fields = {}
        for key, section in self.config.items():
            group_box = QGroupBox(key)
            form_layout = QFormLayout()
            group_box.setLayout(form_layout)

            for sub_key, value in section.items():
                full_key = f"{key}_{sub_key}"
                label = QLabel(sub_key)

                if full_key == "FOV_field_of_view_mode":
                    input_field = QButtonGroup(self)
                    radio1 = QRadioButton("1")
                    radio2 = QRadioButton("0")
                    if value == 1:
                        radio1.setChecked(True)
                    else:
                        radio2.setChecked(True)

                    radio_layout = QHBoxLayout()
                    radio_layout.addWidget(radio1)
                    radio_layout.addWidget(radio2)
                    input_field.addButton(radio1, 1)
                    input_field.addButton(radio2, 0)
                    input_field.buttonClicked.connect(self.fov_mode_changed)

                    form_layout.addRow(label, radio_layout)
                else:
                    input_field = QLineEdit(str(value))
                    if key == "FOV" and self.config["FOV"]["field_of_view_mode"] == 1:
                        input_field.setEnabled(False)
                    form_layout.addRow(label, input_field)

                self.input_fields[full_key] = input_field

            self.group_layout.addWidget(group_box)

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
            for sub_key, _ in section.items():
                full_key = f"{key}_{sub_key}"
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
                    self.config[key][sub_key] = new_value



if __name__ == '__main__':
    app = QApplication(sys.argv)
    config_editor = ConfigEditor()
    config_editor.show()
    sys.exit(app.exec_())
