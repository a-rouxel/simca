import yaml
from PyQt5.QtWidgets import (QTabWidget, QSpinBox,QHBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit, QWidget, QFormLayout, QScrollArea, QGroupBox,QRadioButton, QButtonGroup,QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot,QCoreApplication

class AcquisitionWidget(QWidget):
    def __init__(self,system_config):
        super().__init__()
        pass