import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget

from gui_elements import OpticsWidget
from gui_elements import EditorSystemConfigWidget
from gui_elements import FilteringCubeWidget
from gui_elements import SceneWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('SIMCA')

        self.editor_system_config = EditorSystemConfigWidget(initial_system_config_path="config/cassi_system.yml")
        self.system_config_dock = QDockWidget("Editor for system config")
        self.system_config_dock.setWidget(self.editor_system_config)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.system_config_dock)
        self.system_config_dock.setFixedWidth(int(self.width() / 2))

        self.optics_widget = OpticsWidget(self.editor_system_config,optics_config_path="config/optics.yml")
        self.optics_dock = QDockWidget("Optics")
        self.optics_dock.setWidget(self.optics_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.optics_dock)

        self.filtering_config_widget = FilteringCubeWidget(self.editor_system_config,filtering_config_path="config/filtering.yml")
        self.filtering_config_dock = QDockWidget("Filtering Cube")
        self.filtering_config_dock.setWidget(self.filtering_config_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filtering_config_dock)

        self.scene_widget = SceneWidget(scene_config_path="config/scene.yml")
        self.scene_dock = QDockWidget("Scene")
        self.scene_dock.setWidget(self.scene_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.scene_dock)


        self.tabifyDockWidget(self.optics_dock, self.filtering_config_dock)
        self.tabifyDockWidget(self.filtering_config_dock, self.scene_dock)
        # Connect the signal to the slot
        self.tabifiedDockWidgetActivated.connect(self.check_dock_visibility)

        self.optics_dock.raise_()

    def check_dock_visibility(self, dock_widget):
        # If the currently selected dock widget is the Scene dock, hide the system_config_dock
        if dock_widget is self.scene_dock or dock_widget is self.filtering_config_dock:
            self.system_config_dock.setVisible(False)
        else:
            self.system_config_dock.setVisible(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())