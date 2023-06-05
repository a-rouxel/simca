import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget
from PyQt5.QtGui import QIcon
from gui_elements import OpticsWidget
from gui_elements import EditorSystemConfigWidget
from gui_elements import FilteringCubeWidget
from gui_elements import SceneWidget
from gui_elements import AcquisitionWidget

from CassiSystem import CassiSystem
import os
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowIcon(QIcon('logo_SIMCA.png'))
        # self.setWindowIcon(QIcon('Figure_1.ico'))
        self.setWindowTitle('SIMCA')

        self.scene_widget = SceneWidget(scene_config_path="config/scene.yml")
        self.scene_dock = QDockWidget("Scene")
        self.scene_dock.setWidget(self.scene_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.scene_dock)


        self.system_editor = EditorSystemConfigWidget(initial_system_config_path="config/cassi_system.yml")
        self.system_config_dock = QDockWidget("Editor Optical Design")
        self.system_config_dock.setWidget(self.system_editor)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.system_config_dock)
        self.system_config_dock.setVisible(False)
        self.system_config_dock.setFixedWidth(int(self.width() / 2))

        # --- Initiliaze CASSI SYSTEM -- #



        self.optics_widget = OpticsWidget(self.system_editor,optics_config_path="config/optics.yml")
        self.optics_dock = QDockWidget("Optical Design")
        self.optics_dock.setWidget(self.optics_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.optics_dock)

        self.filtering_widget = FilteringCubeWidget(self.system_editor,filtering_config_path="config/filtering.yml")
        self.filtering_dock = QDockWidget("Masks")
        self.filtering_dock.setWidget(self.filtering_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filtering_dock)

        self.acquisition_widget = AcquisitionWidget(self.system_editor,self.scene_widget, self.filtering_widget,acquisition_config_path="config/acquisition.yml")
        self.acquisition_dock = QDockWidget("Acquisition")
        self.acquisition_dock.setWidget(self.acquisition_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.acquisition_dock)

        self.tabifyDockWidget(self.scene_dock, self.optics_dock)
        self.tabifyDockWidget(self.optics_dock, self.filtering_dock)
        self.tabifyDockWidget(self.filtering_dock, self.acquisition_dock)

        self.scene_dock.raise_()
        # Connect the signal to the slot
        self.tabifiedDockWidgetActivated.connect(self.check_dock_visibility)

    def check_dock_visibility(self, dock_widget):
        # If the currently selected dock widget is the Scene dock, hide the system_config_dock
        if dock_widget is self.scene_dock or dock_widget is self.acquisition_dock:
            self.system_config_dock.setVisible(False)
        else:
            self.system_config_dock.setVisible(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('logo_SIMCA.png'))  # Add this line
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())