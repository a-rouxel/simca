from PyQt5.QtWidgets import (QVBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QWidget, QFormLayout, QScrollArea, QGroupBox,QComboBox)
import yaml


class EditorSystemConfigWidget(QWidget):

    def __init__(self,initial_system_config_path=None):
        super().__init__()

        self.initial_system_config_path = initial_system_config_path
        self.init_ui()



        if self.initial_system_config_path is not None:
            self.load_config(initial_system_config_path)  # Add this line



    def init_ui(self):
        # self.setWindowTitle('Editor for system config')

        # Load config button
        self.load_config_button = QPushButton("Load Config")
        self.load_config_button.clicked.connect(self.on_load_config_clicked)

        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.clicked.connect(self.save_config)


        # Create layout and widgets
        main_layout = QVBoxLayout()

        self.group_layout = QVBoxLayout()

        scroll_widget = QWidget()
        scroll_widget.setLayout(self.group_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)

        # main_layout.addLayout(buttons_layout)
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(self.load_config_button)
        main_layout.addWidget(self.save_config_button)
        self.setLayout(main_layout)  # Set the layout for the current widget


    def load_config(self, file_name):
        if file_name:
            with open(file_name, 'r') as file:
                self.config = yaml.safe_load(file)
                self.show_config()

    def on_load_config_clicked(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open YAML", "", "YAML Files (*.yml)")
        if file_name:
            self.load_config(file_name)

    def load_config(self, file_name):
        # Check if 'input_fields' exists and then clear the old configuration
        if hasattr(self, 'input_fields'):
            for key, widget in self.input_fields.items():
                if isinstance(widget, dict):  # if widget is a dictionary, iterate over its items
                    for subwidget in widget.values():
                        subwidget.deleteLater()
                    widget.clear()  # clear the dictionary
                else:
                    widget.deleteLater()
            self.input_fields.clear()

        # Check if 'input_labels' exists and then delete all its widgets
        if hasattr(self, 'input_labels'):
            for key, widget in self.input_labels.items():
                if isinstance(widget, dict):  # if widget is a dictionary, iterate over its items
                    for subwidget in widget.values():
                        subwidget.deleteLater()
                    widget.clear()  # clear the dictionary
                else:
                    widget.deleteLater()
            self.input_labels.clear()

        # Check if 'input_groups' exists and then delete all its widgets
        if hasattr(self, 'input_groups'):
            for widget in self.input_groups.values():
                widget.deleteLater()
            self.input_groups.clear()

        # Clear the old group layout
        for i in reversed(range(self.group_layout.count())):
            widget = self.group_layout.itemAt(i).widget()
            if widget:  # additional check in case of None
                self.group_layout.removeWidget(widget)
                widget.deleteLater()

        # Load the new configuration
        with open(file_name, 'r') as file:
            self.config = yaml.safe_load(file)
        # Call a method to update the GUI with the loaded config
        self.show_config()


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

                if sub_key in ['dispersive element 1', 'dispersive element 2']:
                    sub_group_box = QGroupBox(sub_key)
                    sub_form_layout = QFormLayout()
                    sub_group_box.setLayout(sub_form_layout)
                    self.input_groups[full_key] = sub_group_box  # Save a reference to the QGroupBox

                    self.input_fields[sub_key] = {}  # Store the fields in a separate dictionary
                    self.input_labels[sub_key] = {}  # Store the labels in a separate dictionary

                    for nested_sub_key, nested_value in value.items():
                        nested_full_key = f"{full_key}_{nested_sub_key}"
                        nested_label = QLabel(nested_sub_key)

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
                    input_field = QLineEdit(str(value))
                    form_layout.addRow(label, input_field)
                    self.input_fields[full_key] = input_field
                    self.input_labels[full_key] = label

            self.group_layout.addWidget(group_box)

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
        # Loop over all input fields and add their values to the config
        for key, input_field in self.input_fields.items():
            # Split the key into parts
            key_parts = key.split("_")

            if len(key_parts) == 3:
                section, sub_section, sub_key = key_parts
                # Ensure the section exists in the config
                if section not in config:
                    config[section] = {}
                # Ensure the sub_section exists in the config
                if sub_section not in config[section]:
                    config[section][sub_section] = {}
                target_dict = config[section][sub_section]
            elif len(key_parts) == 2:
                section, sub_key = key_parts
                # Ensure the section exists in the config
                if section not in config:
                    config[section] = {}
                target_dict = config[section]
            else:
                continue  # Skip this key, it doesn't follow the expected structure

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
            target_dict[sub_key] = value


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
