---
title: 'SIMCA : a simulator for Coded-Aperture Spectral Snapshot Imaging (CASSI)'
tags:
  - Python
  - optics
  - ray-tracing
  - coded-aperture
  - hyperspectral
  - snapshot imaging
  - optical design
authors:
  - name: Antoine Rouxel
    orcid: 0000-0003-4784-7244
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Valentin Portmann
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Antoine Monmayrant
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - name: Simon Lacroix
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 4
  - name: Hervé Carfanta
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 4
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 21 July 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The image formation in coded aperture spectral imagers is a key information to process the acquired compress data, and the optical system design and calibration of these instruments require great care. 
`SIMCA` is a python-based tool built upon ray-tracing equations of each optical component to produce realist measurements of various CASSI systems.
The underlying model takes into account spatial filtering, spectral dispersion, optical distortions, PSF, sampling effects and optical misalignments.

# Statement of need
![Working principle of a Double-Disperser CASSI.\label{fig:DD-CASSI}](DD-CASSI.png){width="70%"}


Spectral imaging has been considerably renewed by the advent of coded aperture imagers, known as CASSI ("Coded-aperture Spectral Snapshot Imager" \cite{Wagadarikar2008}). 
These systems are designed to acquire combinations of spatio-spectral voxels (elements of the hyperspectral cube) on each pixel of the detector.
To interpret these multiplexed measurements, various reconstruction \cite{GAP-TV, Twist, SA, lambda-net, PnP} and classification (\cite{adaptive baysian approach, CSSNET}) algorithms have been developed.
These computational imaging systems are used to reduce the number of acquisitions required to interpret High-Dimensional Data such as hyperspectral images.

 
* Advantages of CASSI systems in general : 
    * optical processing : exploring how to acquire spectral information in regard of a given task (classification, unmixing, ttarget detection, reconstruction, etc.)
    * lower number of acquisitions (factor 10) required 
    * lower amount of data to store (factor 10)
    * snapshot imaging : no need for mechanical scanning
    * adaptive perception in some systems
* No open-source tool to simulate CASSI systems precisely
* Virtual prototyping for testing and training
* Lack of knowledge on the optical caracteristics and the coded aperture needed to reconstruct and/or classify scenes.


# Brief software description

`SIMCA` is a simulation tool to generate realistic Coded-Aperture Spectral Snapshot Imagers measurements.
The repository contains an application programming interface (API) and a graphical user interface (GUI) built in PyQt5 to analyze hyperspectral scenes and interact with the API.
Tutorials for using the GUI are available [here](https://arouxel.gitlab.io/simca-documentation/).


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



# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements



# References
