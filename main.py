import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget

from gui_elements import DimensioningWidget
from gui_elements import EditorSystemConfigWidget
from gui_elements import FilteringCubeWidget
from gui_elements import SceneWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('SIMCA')

        # Set the stylesheet for the QMainWindow

        self.editor_system_config = EditorSystemConfigWidget(initial_system_config_path="config/cassi_system.yml")
        self.system_config_dock = QDockWidget("Editor for system config")
        self.system_config_dock.setWidget(self.editor_system_config)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.system_config_dock)

        self.dimensioning_widget = DimensioningWidget(self.editor_system_config,dimensioning_config_path="config/dimensioning.yml")
        self.dimensioning_dock = QDockWidget("Optics")
        self.dimensioning_dock.setWidget(self.dimensioning_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dimensioning_dock)

        # Create the MaskConfigWidget
        self.filtering_config_widget = FilteringCubeWidget(self.editor_system_config,filtering_config_path="config/filtering.yml")
        self.filtering_config_dock = QDockWidget("Filtering Cube")
        self.filtering_config_dock.setWidget(self.filtering_config_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filtering_config_dock)

        # Create the new widget
        self.acquisition_widget = SceneWidget(scene_config_path="config/scene.yml")
        self.acquisition_dock = QDockWidget("Acquisition")
        self.acquisition_dock.setWidget(self.acquisition_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.acquisition_dock)

        # Set tabified dock widgets
        self.tabifyDockWidget(self.filtering_config_dock, self.dimensioning_dock)
        self.tabifyDockWidget(self.dimensioning_dock, self.acquisition_dock)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())