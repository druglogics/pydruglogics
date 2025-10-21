---
title: 'PyDrugLogics: A Python Package for Predicting Drug Synergies Using Boolean Models'
tags:
  - Python
  - boolean models
  - boolean networks
  - logical modeling
  - genetic algorithm
  - drug synergies
  - bliss
  - hsa
authors:
  - name: Laura Szekeres
    orcid: 0009-0002-4141-0500
    affiliation: 1
  - name: John Zobolas
    orcid: 0000-0002-3609-8674
    affiliation: 1
  - name: Åsmund Flobak
    orcid: 0000-0002-3357-425X
    affiliation: 2
affiliations:
  - name: Department of Cancer Genetics, Institute for Cancer Research, Oslo University Hospital, Oslo, Norway
    index: 1
    ror: 00j9c2840
  - name: Department of Clinical and Molecular Medicine, Norwegian University of Science and Technology, Trondheim, Norway
    index: 2
    ror: 05xg72x27 
date: 19 December 2024
bibliography: paper.bib
---

# Summary

For complex diseases such as cancer, combined drug therapies can enhance the efficacy of personalized treatment and 
minimize side effects [@jinRationalCombinationsTargeted2023]. Combined drug therapies may have a stronger effect than 
single-drug treatments, a concept referred to as synergy [@berenbaumWhatSynergy1989]. However, finding synergistic drug 
combinations is a challenging area of modern medicine. The vast number of possible drug combinations, even for small 
sets of drugs, makes it a complex problem due to the combinatorial growth in possibilities. This fact necessitates longer 
testing time in the laboratory experiments and a significant amount of experimental costs.

PyDrugLogics is a Python package that generates optimized Boolean models that represent a biological system and performs 
in-silico perturbations of these models to predict synergistic drug combinations. The implemented method is derived from 
the pipeline published by @flobakFineTuningLogical2023.

# Statement of Need

Logical modeling is a powerful tool that can be used to reduce the costs of identifying synergistic drug combinations. 
By formalizing biological networks into logical models, we are able to simulate the behaviour of large-scale signaling 
networks and predict responses to perturbations (@eduatiPatientspecificLogicModels2020, 
@niederdorferStrategiesEnhanceLogic2020, @bealPersonalizedLogicalModels2021). 

Within this approach, the modeling process constructs a Boolean network to simulate the biological system, such as a 
cancer cell, and identify stable states that reflect the system’s long-term behaviour. During the optimization process, 
the network’s rules and topology are systematically adjusted to match the experimental steady states 
(e.g., protein activities). Multiple Boolean models are generated from this calibration process. This model ensemble is 
used to simulate the in-silico effects of drug perturbations and predict synergy scores for each combination. These 
predicted synergy scores are validated using experimentally measured outcomes, ensuring their predictive accuracy.

Several previous tools have addressed the challenges of modeling biological networks with logical modeling approaches. 
For instance, CellNOptR [@terfveCellNOptRFlexibleToolkit2012] trains protein signaling networks to experimental data 
using multiple logic formalisms. In contrast, our tool focuses on optimizing Boolean models for predicting synergy 
scores. Another pipeline that was introduced in @dorierBooleanRegulatoryNetwork2016 uses Boolean models 
and a genetic algorithm for network construction and perturbation analysis, focusing on attractor identification and 
implemented as a command-line tool. PyDrugLogics builds on similar principles by leveraging advanced computational 
libraries such as MPBNs [@chatainMostPermissiveSemantics2018] and PyBoolNet [@klarnerPyBoolNetPythonPackage2017] to 
calculate stable states and trap spaces, with the added benefit of a Python-based implementation.

The DrugLogics software pipeline was an important step forward for simulating biological networks with logical models 
and predicting synergy scores [@dl-doc]. Originally, structured as three separate Java packages, it preserved modular 
clarity. The Java implementation was well-designed and robust; nevertheless, the Java and Maven environment presented 
challenges in terms of maintainability, installation, and integrability with other community 
tools [@naldiCoLoMoToInteractiveNotebook2018].

Noticing these limitations, PyDrugLogics provides a practical solution that not only retains the core functionality of 
the Java-based pipelines, but also significantly boosts it by reducing the code complexity, improving the execution time, 
and introducing new features that expand its capabilities. By unifying the functionality of the three Java packages into 
a single Python package, PyDrugLogics simplifies the installation and software maintenance. Example improvements include 
the use of a standardized format, BoolNet [@musselBoolNetPackageGeneration2010], for loading the models, as well as 
visualization options (e.g., precision-recall (PR) and receiver operating characteristic (ROC) curves) and statistical 
analyses (e.g., repeated subsampling of the ensemble Boolean models) for robust evaluation of prediction performance.
Additional examples and comparisons of the Python and Java pipelines are available on the project wiki [@pydl-wiki].
PyDrugLogics provides an easy-to-use, flexible, and up-to-date solution for simulating Boolean networks and predicting 
synergistic drug combinations for the prioritization of follow-up lab experiments.

There has been a growing focus on developing tools that prioritize accessibility, reproducibility, and seamless 
integration in the logical modeling community. In particular, the CoLoMoTo Interactive Notebook aims to simplify 
integration and enables faster collaboration [@naldiCoLoMoToInteractiveNotebook2018]. PyDrugLogics adopts this approach 
by integrating into the CoLoMoTo Docker, enabling compatibility with other tools so that researchers can combine 
methodologies and share results more effectively. The comprehensive documentation for PyDrugLogics is provided 
on the package website [@pydl-doc], along with a detailed tutorial [@pydl-tutorial]. The package is available on 
PyPI [@pydl-pypi], offering a simple installation process and integration into Python workflows.

# Brief Overview

The PyDrugLogics pipeline involves two main stages: calibration and prediction. 

The calibration (`train`) function is responsible for loading a Boolean model for a particular biological system 
(e.g., a cancer cell) or the interactions for automatically constructing such a model. The next step is the optimization 
process that uses the PyGAD Genetic Algorithm [@gadPyGADIntuitiveGenetic2021] for finding the best set of Boolean models 
that fit the training data (e.g., protein measurements of the cancerous cell). The optimization process changes the 
model's operators and topology to ensure its behaviour fits the training data. 

In the prediction (`predict`) function, the calibrated models are used to perform in-silico perturbations to simulate 
the effect of various drug treatments and their combinations. The perturbations represent changes to the system to mimic 
drug effects such as inhibition or activation of the affected proteins. The results of the perturbations are analyzed, 
and the predicted viability scores, which represent the system’s response to drug treatments, are computed. Synergy 
scores are then derived from these viability scores, quantifying drug interactions and classifying combinations by 
synergistic potential. To verify the accuracy of the predictions, the pipeline requires the knowledge of the observed 
synergies, which serve as the ground truth, and they are typically derived from experimental datasets or literature 
sources. Using the observed synergies (binary labels: 0 for non-synergistic and 1 for synergistic) as the ground truth, 
binary classification metrics such as the ROC and PR AUC (area under the curve) are generated to evaluate how well 
the predicted synergy scores distinguish between synergistic and non-synergistic drug combinations.

# Acknowledgements

JZ received funding from Astri og Birger Torsteds Legater for cancer research, which contributed to the completion of 
this work.

The following people have contributed code to this package or provided help with technical and scientific questions 
(in alphabetical order): 

- Daniel Nebdal

- Eirini Tsirvouli

- Marco Fariñas

Special thanks are due for their valuable support and contributions.

# References
