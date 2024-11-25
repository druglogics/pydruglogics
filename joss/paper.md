---
title: 'PyDrugLogics'
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
    equal-contrib: true
    affiliation: 1
  - name: John Zobolas
    orcid: 0000-0002-3609-8674
    affiliation: 1
affiliations:
 - name: Department of Cancer Genetics, Institute for Cancer Research, Oslo University Hospital, Oslo, Norway
   index: 1
   ror: 00j9c2840
date: 23 November 2024
bibliography: paper.bib

---

# Summary


The identification of effective drug combinations is a key challenge in modern medicine, particularly for diseases 
like cancer, where synergistic therapies can improve treatment outcomes. By combining drugs that work together to 
achieve greater effects than their individual components, it becomes possible to enhance therapeutic efficacy while 
minimizing side effects. However, the lerge amoutn of possible drug combinations, even for modest drug panels, makes 
experimental testing prohibitively time-consuming and expensive.

# Statement of Need



# Running PyDrugLogics


Running an analysis with PyDrugLogics involves three main stages: input preparation, Boolean model optimization, 
and synergy evaluation. The first step is to prepare the input files, which include a network file describing 
biological interactions (in formats such as `.sif` or `.bnet`), a list of drugs with their targets, and a set 
of training data observations. These inputs must adhere to specific formats to ensure compatibility. 
If needed, pre-processing scripts can be used to convert data into the required format, simplifying the 
setup process for researchers.


The pipeline begins with the **Gitsbe** module, which constructs and optimizes Boolean models. Boolean equations 
are generated based on the interactions specified in the input network file. For example, the activity of a target 
node may depend on activating and inhibitory regulators, expressed as:
\[
\text{Target} \; *= \; (\text{A or B}) \; \text{and not} \; (\text{C or D})
\]
where \(A\) and \(B\) are activators, and \(C\) and \(D\) are inhibitors. Gitsbe uses a genetic algorithm to refine 
these models, introducing random mutations to logical equations and selecting the models that best fit the training 
data. Fitness scores are calculated by comparing the model outputs with the observations, and the process continues 
through multiple generations, with crossover and mutation phases improving the model population. The optimization 
halts when either a specified fitness threshold is reached or a maximum number of generations has been completed.



# Citations

- Fine tuning a logical model of cancer cells to predict drug synergies: combining manual curation and automated parameterization
https://www.frontiersin.org/journals/systems-biology/articles/10.3389/fsysb.2023.1252961/full

- Concepts in Boolean network modeling: What do they all mean?
https://www.csbj.org/article/S2001-0370(19)30460-X/fulltext

- Boolean modeling in systems biology: an overview of methodology and applications
https://iopscience.iop.org/article/10.1088/1478-3975/9/5/055001

- PyGAD
https://link.springer.com/article/10.1007/s11042-023-17167-y#author-information

- Pyboolnet
https://academic.oup.com/bioinformatics/article/33/5/770/2725550?login=false

- mpbn
https://arxiv.org/abs/2403.06255

- colomoto
https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2018.00680/full

- bliss
https://onlinelibrary.wiley.com/doi/10.1111/j.1744-7348.1939.tb06990.x

- hsa


Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

??????????????????????????????

# References

- Fine tuning a logical model of cancer cells to predict drug synergies: combining manual curation and automated parameterization
https://www.frontiersin.org/journals/systems-biology/articles/10.3389/fsysb.2023.1252961/full

- Concepts in Boolean network modeling: What do they all mean?
https://www.csbj.org/article/S2001-0370(19)30460-X/fulltext

- Boolean modeling in systems biology: an overview of methodology and applications
https://iopscience.iop.org/article/10.1088/1478-3975/9/5/055001

- PyGAD
https://link.springer.com/article/10.1007/s11042-023-17167-y#author-information

- Pyboolnet
https://academic.oup.com/bioinformatics/article/33/5/770/2725550?login=false

- mpbn
https://arxiv.org/abs/2403.06255

- colomoto
https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2018.00680/full

- bliss
https://onlinelibrary.wiley.com/doi/10.1111/j.1744-7348.1939.tb06990.x

- hsa

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.