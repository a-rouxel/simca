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
The underlying model takes into account spatial filtering, spectral dispersion, optical distortions, sampling effects and optical misalignments.
The performances of the instrument can then be evaluated by analyzing raw optical data or the reconstruction of the hyperspectral scene.

# Statement of need

* Imaging has been considerably renewed by the advent of coded aperture imagers, known as CASSI systems ("Coded-aperture Spectral Snapshot Imager" \cite{Wagadarikar2008}). 
* Advantages of CASSI systems in general : 
    * optical processing : exploring how to acquire spectral information in regard of a given task (classification, unmixing, ttarget detection, reconstruction, etc.)
    * lower number of acquisitions (factor 10) required 
    * lower amount of data to store (factor 10)
    * snapshot imaging : no need for mechanical scanning
    * adaptive perception in some systems
* No open-source tool to simulate CASSI systems precisely
* Lack of knowledge on the optical caracteristics and the coded aperture needed to reconstruct and/or classify scenes.


# Proposal

An end-to-end simulation tool to evaluate CASSI system performances based on optical analysis and reconstruction accuracy.
Tutorials are available [here](https://arouxel.gitlab.io/simca-documentation/)  
  
## Optical model
First, the core of the tool is a ray-tracing model of the optical system allowing for precise simulation of light propagation depending on the caracteristics and types of optical components.
  
## Compressed image formation
  
Second, we provide a way of harnessing this propagation data to generate realistic measurements of the scene, including spatial filtering, spectral dispersion, optical distortions and sampling effects.
  
## Performances analysis

Third, a draft of a CASSI system evaluation tool to analyze optical and task-specific performances. 



# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements



# References
