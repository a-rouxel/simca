import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget

from gui_elements import DimensioningWidget
from gui_elements import EditorSystemConfigWidget
from gui_elements import AcquisitionWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('SIMCA')

        self.editor_system_config = EditorSystemConfigWidget(initial_system_config_path="config/cassi_system.yml")
        self.system_config_dock = QDockWidget("Editor for system config")
        self.system_config_dock.setWidget(self.editor_system_config)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.system_config_dock)

        self.dimensioning_widget = DimensioningWidget(self.editor_system_config)
        self.dimensioning_dock = QDockWidget("Dimensioning")
        self.dimensioning_dock.setWidget(self.dimensioning_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dimensioning_dock)


        # Create the MaskConfigWidget
        self.mask_config_widget = AcquisitionWidget(self.editor_system_config)
        self.mask_config_dock = QDockWidget("Mask Config")
        self.mask_config_dock.setWidget(self.mask_config_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.mask_config_dock)


        # Set tabified dock widgets
        self.tabifyDockWidget(self.mask_config_dock,self.dimensioning_dock)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
