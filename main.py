import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication,QMainWindow, QDockWidget)

from gui_elements import DimensioningWidget
from gui_elements import EditorSystemConfigWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('SIMCA -- Dimensioning -- V0.1')

        self.editor_system_config = EditorSystemConfigWidget(initial_system_config_path="config/cassi_system.yml")
        self.system_config_dock = QDockWidget("Editor for system config")
        self.system_config_dock.setWidget(self.editor_system_config)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.system_config_dock)

        # Create the DimensioningWidget
        self.dimensioning_widget = DimensioningWidget(self.editor_system_config)
        # Add it as a dock widget
        self.dimensioning_dock = QDockWidget("Dimensioning")
        self.dimensioning_dock.setWidget(self.dimensioning_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dimensioning_dock)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


