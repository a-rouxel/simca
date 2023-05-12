# SIMCA -- Dimensioning -- V0.1

This Python/Qt application allows users to run dimensioning analysis on various datasets.

## Features

The application provides the following features:

1. Allows users to specify various parameters related to the dimensioning analysis.
2. Displays results of the analysis in different forms, such as camera pixelization, retropropagation to DMD -- mapping, spectral dispersion, and distorsion map.

## Installation

To install and run the application, you will need to have Python installed on your computer. You can download Python from [here](https://www.python.org/downloads/).

After installing Python, you will also need to install PyQt and other dependencies. This can be done by running the following commands in your terminal:

```bash
pip install pyqt5 matplotlib numpy
```

After installing the dependencies, you can run the application by navigating to the directory containing the python script and running:

```bash
python main.py
```

## Usage

The application provides a graphical interface for running dimensioning analysis.

- The application window has two main sections: The left section for inputting parameters, and the right section for displaying the results of the analysis.

- You can input the system configuration and dimensioning parameters on the left section of the window.

- After you have input the parameters, you can click on the "Run Dimensioning" button to start the analysis.

- The results of the analysis will be displayed in the form of graphs on the right section of the window.

### Input Parameters

- **Results directory:** This is the directory where the results of the analysis will be saved.
- **Number of spectral samples:** This is the number of spectral samples to be used in the analysis.

### Output Results

- **Camera Pixelization:** This is a graph showing the pixelization of the camera.
- **Retropropagation to DMD -- mapping:** This is a graph showing the mapping of the retropropagation to the DMD.
- **Spectral Dispersion:** This graph shows the spectral dispersion.
- **Distorsion map:** This is a graph showing the distorsion map.

## Contributing

If you would like to contribute to this project, please feel free to fork the repository, make your changes and then create a pull request. If you have any questions, please feel free to contact the maintainers of this project.

## License

This project is licensed under the MIT License. Please see the LICENSE file for more details.
