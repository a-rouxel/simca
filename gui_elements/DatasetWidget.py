import yaml
from PyQt5.QtWidgets import (QTabWidget,QHBoxLayout, QPushButton, QLineEdit,
                             QWidget, QFormLayout, QGroupBox,QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import  QSlider
from PyQt5.QtCore import Qt
from utils import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import os

class datasetContentDisplay(QWidget):

    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Create a label
        self.label = QLabel("Slice Number: ")
        self.layout.addWidget(self.label)

        # Create a slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.update_image)
        self.layout.addWidget(self.slider)

        # Create ImageView item with a PlotItem as its view box
        self.imageView = pg.ImageView(view=pg.PlotItem())
        self.layout.addWidget(self.imageView)

    def diplay_dataset_content(self, dataset, list_wavelengths):

        self.list_wavelengths = list_wavelengths

        # Normalize filtering_cube values to range [0, 255] for ImageView
        self.data = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset)) * 255
        self.data = self.data.astype(np.uint8)

        # Set slider maximum value
        self.slider.setMaximum(self.data.shape[2] - 1)

        # Display the first slice
        self.update_image(0)

    def update_image(self, slice_index):
        # Update the label
        self.label.setText("wavelength: " + str(int(self.list_wavelengths[slice_index])) + " nm")

        # Display the slice
        image = np.rot90(self.data[:, :, slice_index],1)
        self.imageView.setImage(np.flip(image, axis=0))

class datasetHistogram(QWidget):

    def __init__(self):
        super().__init__()

        # Create a QVBoxLayout for the widget
        self.layout = QVBoxLayout(self)

        # Create a figure and a canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

    def plot_label_histogram(self, ground_truth, label_values, ignored_labels,palette):
        """
        Plot a histogram with the occurrence of each data label in the dataset.

        :param ground_truth: np.array, ground truth labels
        :param label_values: list, label_values[i] = name of the class i
        :param palette: color palette to use, must be a list of colors where the index corresponds to the label
        :param ignored_labels: list of ignored labels (pixel with no label)
        """
        # Clear the figure
        self.figure.clear()

        # Create an axes
        ax = self.figure.add_subplot(111)

        # Compute the occurrence of each label
        values, counts = np.unique(ground_truth, return_counts=True)

        # Remove ignored labels
        valid_indices = [i for i, v in enumerate(values) if v not in ignored_labels]
        values = values[valid_indices]
        counts = counts[valid_indices]

        # Create a list of label names
        labels = [label_values[i] for i in values]

        # Get the colors from the palette and convert to the range [0, 1]
        colors = [tuple([x / 255 for x in palette[i]]) for i in values]

        # Plot the histogram
        ax.bar(labels, counts, color=colors)

        # Set the title and labels
        ax.set_title("Label occurrence")
        ax.set_xlabel("Label")
        ax.set_ylabel("Occurrence")

        # Rotate the x-axis labels
        ax.set_xticks(range(len(labels)))  # Add this line
        ax.set_xticklabels(labels, rotation=45)

        # Redraw the canvas
        self.canvas.draw()

from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel, QScrollArea
from PyQt5.QtGui import QColor, QPixmap
import pyqtgraph as pg
import numpy as np


class datasetLabelisation(QWidget):
    def __init__(self):
        super().__init__()

        # Create a QHBoxLayout for the widget
        self.layout = QHBoxLayout(self)

        # Create a PlotWidget (QGraphicsView)
        self.plot_widget = pg.PlotWidget()

        # Lock the aspect ratio
        self.plot_widget.getViewBox().setAspectLocked(True)

        # Create ImageItem and add it to PlotWidget
        self.image_item = pg.ImageItem(border='w')
        self.plot_widget.addItem(self.image_item)

        self.layout.addWidget(self.plot_widget)

        # Create a QScrollArea for the legend
        self.legend_scroll_area = QScrollArea(self)
        self.layout.addWidget(self.legend_scroll_area)


    def display_ground_truth(self, ground_truth, label_values, palette):
        """
        Display the ground truth with each color corresponding to a label_value.

        :param ground_truth: np.array, ground truth labels
        :param label_values: list, label_values[i] = name of the class i
        :param palette: color palette to use, must be a list of colors where the index corresponds to the label
        """
        # Convert the ground truth labels to colors using the palette
        image = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8)
        for i in range(len(label_values)):
            image[ground_truth == i] = palette[i]

        image = np.rot90(image, 3)
        # Display the image using ImageItem
        self.image_item.setImage(image)


        # Create a legend
        legend_widget = LegendWidget(label_values, palette, ground_truth)
        self.legend_scroll_area.setWidget(legend_widget)

class RGBdatasetDisplay(QWidget):
    def __init__(self):
        super().__init__()

        # Create a QVBoxLayout for the widget
        self.layout = QVBoxLayout(self)

        # Create a figure and a canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

    def display_rgb_img(self, dataset, rgb_bands):
        rgb_array = np.zeros((dataset.shape[0], dataset.shape[1], 3))

        rgb_array[:, :, 0] = dataset[:, :, rgb_bands[0]]
        rgb_array[:, :, 1] = dataset[:, :, rgb_bands[1]]
        rgb_array[:, :, 2] = dataset[:, :, rgb_bands[2]]

        rgb_array /= np.max(rgb_array)
        rgb_array = np.asarray(255 * rgb_array, dtype='uint8')

        # rgb_array[:, :, 0] = dataset[:, :, rgb_bands[0]]
        # rgb_array[:, :, 1] = dataset[:, :, rgb_bands[1]]
        # rgb_array[:, :, 2] = dataset[:, :, rgb_bands[2]]

        self.figure.clear()
        ax1 = self.figure.add_subplot(111)
        im = ax1.imshow(rgb_array)
        ax1.set_title('RGB image of the dataset')
        self.canvas.draw()

class LegendWidget(QWidget):
    def __init__(self, label_values, palette, ground_truth):
        super().__init__()

        self.layout = QVBoxLayout(self)

        # Create a colored square and a label for each label_value
        for i in range(len(label_values)):
            # Create a QHBoxLayout for the colored square and the label
            hbox = QHBoxLayout()

            # Create a QPixmap (colored square) for the label
            pixmap = QPixmap(20, 20)
            pixmap.fill(QColor(*palette[i]))
            pixmap_label = QLabel()
            pixmap_label.setPixmap(pixmap)
            hbox.addWidget(pixmap_label)

            # Create a QLabel for the label text
            label_text = f"{label_values[i]}: {np.sum(ground_truth == i)}"
            text_label = QLabel(label_text)
            hbox.addWidget(text_label)

            self.layout.addLayout(hbox)




class SpectralDataDisplay(QWidget):
    def __init__(self):
        super().__init__()

        # Create a QVBoxLayout for the widget
        self.layout = QVBoxLayout(self)

        # Create a figure and a canvas
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

    def plot_spectrums(self, spectrums, std_spectrums, palette, label_values):
        """Plot mean and intra-class covariance spectrum for each class on a single graph

        Args:
            spectrums: dictionary (name -> spectrum) of spectrums to plot
            std_spectrums: dictionary (name -> std of spectrum) of spectrums to plot
            palette: dictionary, color palette
            label_values: list, label_values[i] = name of the ith class
        """
        # Clear previous figure
        self.figure.clear()

        # Create an axes
        ax = self.figure.add_subplot(111)

        for k, v in spectrums.items():
            std_spec = std_spectrums[k]
            up_spec = v + std_spec
            low_spec = v - std_spec
            x = np.arange(len(v))
            i = label_values.index(k)

            # Convert RGB to normalized tuple
            color = tuple([x / 255 for x in palette[i]])

            ax.fill_between(x, up_spec, low_spec, color=color, alpha=0.3)

        for k, v in spectrums.items():
            x = np.arange(len(v))
            i = label_values.index(k)

            # Convert RGB to normalized tuple
            color = tuple([x / 255 for x in palette[i]])

            ax.plot(x, v, color=color, label=k)

        ax.set_title("Mean spectrum per class")
        ax.legend()

        # Draw the figure on the canvas
        self.canvas.draw()



    def display_spectral_data(self, mean_spectrums, std_spectrums, palette, label_values):
        # Call the plot function
        self.plot_spectrums(mean_spectrums, std_spectrums, palette, label_values)




class datasetConfigEditor(QWidget):

    dataset_loaded = pyqtSignal(int,int,int,float,float)
    def __init__(self,initial_config_file=None):
        super().__init__()
        self.initial_config_file = initial_config_file

        # Create a QScrollArea
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        # Create a widget for the scroll area
        scroll_widget = QWidget()

        self.datasets_directory = QLineEdit()


        self.directories_combo = QComboBox()




        self.dataset_dimension_x = QLineEdit()
        self.dataset_dimension_x.setReadOnly(True)  # The dimensions should be read-only
        self.dataset_dimension_y = QLineEdit()
        self.dataset_dimension_y.setReadOnly(True)  # The dimensions should be read-only
        self.dataset_nb_of_spectral_bands = QLineEdit()
        self.dataset_nb_of_spectral_bands.setReadOnly(True)  # The dimensions should be read-only
        self.minimum_wavelengths = QLineEdit()
        self.minimum_wavelengths.setReadOnly(True)  # The dimensions should be read-only
        self.maximum_wavelengths = QLineEdit()
        self.maximum_wavelengths.setReadOnly(True)  # The dimensions should be read-only

        self.dataset_loaded.connect(self.update_dataset_dimensions)

        # Add the dimensioning configuration editor, the result display widget, and the run button to the layout


        dataset_layout = QFormLayout()
        dataset_layout.addRow("datasets directory", self.datasets_directory)

        self.reload_datasets_button = QPushButton('reload datasets')
        self.reload_datasets_button.clicked.connect(self.load_datasets)

        dataset_layout.addWidget(self.reload_datasets_button)

        chosen_dataset = QFormLayout()



        chosen_dataset.addRow("dataset name",self.directories_combo)
        chosen_dataset.addRow("dimension along X", self.dataset_dimension_x)
        chosen_dataset.addRow("dimension along Y", self.dataset_dimension_y)
        chosen_dataset.addRow("number of spectral bands", self.dataset_nb_of_spectral_bands)
        chosen_dataset.addRow("minimum wavelength", self.minimum_wavelengths)
        chosen_dataset.addRow("maximum wavelength", self.maximum_wavelengths)

        chosen_dataset_group = QGroupBox("Chosen dataset")
        chosen_dataset_group.setLayout(chosen_dataset)

        # dataset_layout.addRow("chosen dataset", chosen_dataset)


        dataset_group = QGroupBox("Settings")
        dataset_group.setLayout(dataset_layout)




        main_layout = QVBoxLayout()
        main_layout.addWidget(dataset_group)
        main_layout.addWidget(chosen_dataset_group)



        # Set the layout on the widget within the scroll area
        scroll_widget.setLayout(main_layout)

        # Set the widget for the scroll area
        scroll.setWidget(scroll_widget)

        # Create a layout for the current widget and add the scroll area
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

        # Load the initial configuration file if one was provided
        if self.initial_config_file is not None:
            self.load_config(initial_config_file)

        self.load_datasets()

    def load_datasets(self):

        dataset_dir = self.datasets_directory.text()
        self.directories_combo.clear()
        if os.path.isdir(dataset_dir):
            sub_dirs = [name for name in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, name))]
            self.directories_combo.clear()
            self.directories_combo.addItems(sub_dirs)

    def load_config(self, file_name):
        with open(file_name, 'r') as file:
            self.config = yaml.safe_load(file)
        # Call a method to update the GUI with the loaded configs
        self.update_config()
    def update_config(self):
        # This method should update your QLineEdit and QSpinBox widgets with the loaded configs.
        self.datasets_directory.setText(self.config['datasets directory'])

    @pyqtSlot(int,int,int,float,float)
    def update_dataset_dimensions(self, x_dim,y_dim,wav_dim,min_wav,max_wav):
        # Update the GUI in this slot function, which is called from the main thread
        self.dataset_dimension_x.setText(str(x_dim))
        self.dataset_dimension_y.setText(str(y_dim))
        self.dataset_nb_of_spectral_bands.setText(str(wav_dim))
        self.minimum_wavelengths.setText(str(min_wav))
        self.maximum_wavelengths.setText(str(max_wav))


class Worker(QThread):

    finished_load_dataset = pyqtSignal(np.ndarray,list)
    finished_rgb_dataset = pyqtSignal(np.ndarray,tuple)
    finished_explore_dataset = pyqtSignal(dict,dict,list)
    finished_dataset_labelisation = pyqtSignal(np.ndarray, list, dict)
    finished_dataset_label_histogram = pyqtSignal(np.ndarray, list, list,dict)

    def __init__(self,cassi_system,dataset_config_editor):
        super().__init__()

        self.cassi_system = cassi_system
        self.dataset_config_editor = dataset_config_editor

    def run(self):
        self.dataset_config_editor.update_config()
        self.cassi_system.load_dataset(self.dataset_config_editor.directories_combo.currentText(),self.dataset_config_editor.datasets_directory.text())

        self.dataset_config_editor.dataset_loaded.emit(self.cassi_system.dataset.shape[1],self.cassi_system.dataset.shape[0],self.cassi_system.dataset.shape[2],
                                                   self.cassi_system.list_dataset_wavelengths[0],self.cassi_system.list_dataset_wavelengths[-1])

        self.stats_per_class = explore_spectrums(self.cassi_system.dataset, self.cassi_system.dataset_gt, self.cassi_system.dataset_label_values,self.cassi_system.dataset_ignored_labels)

        self.finished_load_dataset.emit(self.cassi_system.dataset,self.cassi_system.list_dataset_wavelengths)  # Emit a tuple of arrays
        self.finished_rgb_dataset.emit(self.cassi_system.dataset,self.cassi_system.dataset_rgb_bands)  # Emit a tuple of arrays
        self.finished_explore_dataset.emit(self.stats_per_class,self.cassi_system.dataset_palette,self.cassi_system.dataset_label_values)
        self.finished_dataset_labelisation.emit(self.cassi_system.dataset_gt,self.cassi_system.dataset_label_values,self.cassi_system.dataset_palette)# Emit a tuple of arrays
        self.finished_dataset_label_histogram.emit(self.cassi_system.dataset_gt,self.cassi_system.dataset_label_values,self.cassi_system.dataset_ignored_labels,self.cassi_system.dataset_palette)
class DatasetWidget(QWidget):
    def __init__(self,cassi_system=None,dataset_config_path="configs/dataset.yml"):
        super().__init__()

        self.cassi_system = cassi_system

        self.layout = QHBoxLayout()

        self.result_display_widget = QTabWidget()

        self.dataset_content_display = datasetContentDisplay()
        self.rgb_dataset_display = RGBdatasetDisplay()
        self.dataset_spectral_data_display = SpectralDataDisplay()
        self.dataset_labelisation_display = datasetLabelisation()
        self.dataset_label_histogram = datasetHistogram()

        self.result_display_widget.addTab(self.dataset_content_display, "dataset Content")
        self.result_display_widget.addTab(self.rgb_dataset_display, "dataset RGB")
        self.result_display_widget.addTab(self.dataset_spectral_data_display, "dataset Caracterization")
        self.result_display_widget.addTab(self.dataset_labelisation_display, "dataset Labelisation")
        self.result_display_widget.addTab(self.dataset_label_histogram, "dataset Labels Histogram")


        self.run_button = QPushButton('Load dataset')
        # self.run_button.setStyleSheet('QPushButton {background-color: green; color: white;}')        # Connect the button to the run_dimensioning method
        self.run_button.clicked.connect(self.run_load_dataset)

        # Create a group box for the run button
        self.run_button_group_box = QGroupBox()
        run_button_group_layout = QVBoxLayout()

        if dataset_config_path is not None:
            self.dataset_config_editor = datasetConfigEditor(dataset_config_path)

        self.layout.addWidget(self.dataset_config_editor)

        run_button_group_layout.addWidget(self.run_button)
        run_button_group_layout.addWidget(self.result_display_widget)

        self.run_button_group_box.setLayout(run_button_group_layout)
        self.layout.addWidget(self.run_button_group_box)


        self.layout.setStretchFactor(self.run_button_group_box, 1)
        self.layout.setStretchFactor(self.result_display_widget, 3)
        self.setLayout(self.layout)
    #
    def run_load_dataset(self):
        # Get the configs from the editors


        self.worker = Worker(self.cassi_system,self.dataset_config_editor)
        self.worker.finished_load_dataset.connect(self.display_dataset_content)
        self.worker.finished_rgb_dataset.connect(self.display_rgb_dataset)
        self.worker.finished_explore_dataset.connect(self.display_spectral_data)
        self.worker.finished_dataset_labelisation.connect(self.display_ground_truth)
        self.worker.finished_dataset_label_histogram.connect(self.dataset_label_histogram.plot_label_histogram)
        self.worker.start()

    @pyqtSlot(np.ndarray,list)
    def display_dataset_content(self, dataset,list_wavelengths):
        self.dataset = dataset
        self.dataset_content_display.diplay_dataset_content(dataset,list_wavelengths)

    @pyqtSlot(np.ndarray,tuple)
    def display_rgb_dataset(self, dataset,rgb_bands):
        print(rgb_bands)
        self.rgb_dataset_display.display_rgb_img(dataset,rgb_bands)

    @pyqtSlot(dict,dict,list)
    def display_spectral_data(self,stats_class,palette,label_values):
        mean_spectrums, std_spectrums= stats_class["mean_spectrums"], stats_class["std_spectrums"]
        self.dataset_spectral_data_display.display_spectral_data(mean_spectrums, std_spectrums,palette, label_values)

    @pyqtSlot(np.ndarray, list, dict)
    def display_ground_truth(self,ground_truth, label_values, palette):
        self.dataset_labelisation_display.display_ground_truth(ground_truth, label_values, palette)

    @pyqtSlot(np.ndarray, list, list,dict)
    def display_plot_label_histogram(self, ground_truth, label_values, ignored_labels,palette):
        self.dataset_label_histogram.plot_label_histogram(ground_truth, label_values, ignored_labels,palette)
