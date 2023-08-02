---
title: 'SIMCA : a simulator for Coded-Aperture Spectral Snapshot Imaging (CASSI)'
tags:
  - computational imaging
  - hyperspectral
  - optics
  - ray-tracing
  - coded-aperture
  - snapshot imaging
  - optical design
  - python
authors:
  - name: Antoine Rouxel
    orcid: 0000-0003-4784-7244
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Léo Paillet
    affiliation: 1
  - name: Valentin Portmann
    affiliation: 1,2
affiliations:
 - name: LAAS-CNRS, Université de Toulouse, CNRS, Toulouse, France
   index: 1
 - name: IRAP, Université de Toulouse, CNRS, CNES, Toulouse, France
   index: 2
date: 21 July 2023
bibliography: paper.bib

---

# Summary

The image formation in coded-aperture spectral imagers is a key information to process the acquired compressed data, and the optical system design and calibration of these instruments require great care. 
`SIMCA` is a python-based tool built upon ray-tracing equations of each optical component to produce realist measurements of various CASSI systems.
The underlying model takes into account spatial filtering, spectral dispersion, optical distortions, PSF, sampling effects and optical misalignments.

# Statement of need
![Working principle of a Double-Disperser CASSI.\label{fig:DD-CASSI}](DD-CASSI.png){width="70%"}


Spectral imaging has been considerably renewed by the advent of coded aperture imagers, known as CASSI ("Coded-aperture Spectral Snapshot Imager" \cite{Wagadarikar2008}). 
These computational imaging systems reduce the number of acquisitions required to interpret High-Dimensional data such as hyperspectral scenes, they remove the need for mechanical scanning, and they allow for exploration of optimal optical processing in regard of a given task (classification, unmixing, target detection, reconstruction, etc...) 


In order to interpret these multiplexed measurements, various algorithms have been proposed, mainly focusing on classification (\cite{NN with coded aperture optim., adaptive baysian approach, CSSNET}) and reconstruction (\cite{GAP-TV, Twist, SA, lambda-net, PnP}) of the observed scene.
All these algorithms rely on a precise knowledge of the spatial and spectral information contained in the compressed measurements, which is directly related to the optical system caracteristics and calibration.
However, the image formation process is complex, and the calibration of these instruments is a tedious process.

A ray-tracing based parametric propagation model, associated to interpolation methods whose relevance was underlined in \cite{Arguello2013}, is used to simulate the image formation process in CASSI systems.
Our tool allows for a comparison of various system architectures, optical components and coded-aperture patterns.
In addition, this same model can utilized to calibrate a real instrument, build a virtual prototype for testing and training, and explore the optical processing of the compressed measurements in regard of a given task.

Our contribution is motivated by the lack of tools to simulate CASSI systems acquisitions precisely, and the lack of knowledge on the optical caracteristics and the coded aperture needed to interpret a hyperspectral scene.


# Brief software description

`SIMCA` is a simulation tool to generate realistic Coded-Aperture Spectral Snapshot Imagers measurements.
The repository contains an application programming interface (API) and a graphical user interface (GUI) built in PyQt5 to analyze hyperspectral scenes and interact with the API.

Tutorials for using the API and GUI are available [here](https://arouxel.gitlab.io/simca-documentation/).


## Optical model
The core of the code is a ray-tracing model of the optical system allowing for precise simulation of light propagation depending on the caracteristics and types of optical components.
It includes various optical that are often neglected in the literature, such as optical distortions, optical misalignments, etc...

Available propagation models are:
* Higher-Order from \cite{Arguello2013}
* Ray-tracing (first implementation in \cite{Hemsley2020})

Available system architectures are:
* Single-Disperser CASSI
* Double-Disperser CASSI

Available optical components and related caracteristics are:
* Lens (params: focal length)
* Prism (params : apex angle, glass type, orientation misalignments)
* Grating (params : groove density, orientation misalignments)

## Coded-Aperture generation

The generation of coded-apertures is a key step in the design of CASSI systems.
We provide a sample of coded-aperture generation methods, including:
* Random with various ratios of open/closed coded aperture pixels
* two types of blue-noise
* LN-Random (\cite{Ibrahim, Hemsley2022})

None of these methods require prior-knowledge of the scene.


## Compressed image formation
  
We provide a way of harnessing this optics-related data to generate realistic measurements of the scene.
Depending on the system architecture, the compressed image formation varies.
However for each cases it requires multiple projections of the coded-aperture pattern onto the detector after propagation through the optical system, each projection correponds to a specific wavelength.
Interpolation between these multiple projections (none-structured) and the detector pixels grid (structured) is performed, then all these projections are summed to obtain the compressed measurement.




# Acknowledgements

We acknowledge multiple discussions, testing and critiques from Maud Biquard, Hervé Carfantan, Valentin Portmann, , Trung Tin DINH


# References
