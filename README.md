# PyDrugLogics

![PyDrugLogics Logo](https://raw.githubusercontent.com/druglogics/pydruglogics/main/logo.png)

[![PyPI version](https://img.shields.io/pypi/v/pydruglogics)](https://badge.fury.io/py/pydruglogics)
[![Test Status](https://github.com/druglogics/pydruglogics/actions/workflows/run-tests.yml/badge.svg)](https://github.com/druglogics/pydruglogics/actions/workflows/run-tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/druglogics/pydruglogics/blob/main/LICENSE)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://druglogics.github.io/pydruglogics/)


## Overview

PyDrugLogics is a Python package designed for constructing, optimizing Boolean Models and performs in-silico perturbations of the models.
### Core Features
- Construct Boolean model from `.sif` file
- Load Boolean model from `.bnet` file
- Optimize Boolean model
- Generate perturbed models
- Evaluate drug synergies

## Installation

**PyDrugLogics** can be installed via **PyPi**, **Conda**, or **directly from the source**.
### Install PyDrugLogics from PyPI

The process involves two steps to install the PyDrugLogics core package and its necessary external dependencies.

#### 1. Install PyDrugLogics via pip

```bash
pip install pydruglogics
```
#### 2. Install External Dependency

```bash
pip install -r https://raw.githubusercontent.com/druglogics/pydruglogics/main/requirements.txt
```
This will install the PyDrugLogics package and handle all dependencies automatically.


### Install PyDrugLogics via conda

```bash
conda install szlaura::pydruglogics
```

### Install from Source

For the latest development version, you can clone the repository and install directly from the source:

```bash
git clone https://github.com/druglogics/pydruglogics.git
cd pydruglogics
pip install .
pip install -r requirements.txt
```

## CoLoMoTo Notebook environment
PyDrugLogics is available in the CoLoMoTo Docker and Notebook starting from version `2025-01-01`.

### Setup CoLoMoTo Docker and Notebook

1. Install the helper script in a terminal:

```bash
    pip install -U colomoto-docker
```
2. Start the CoLoMoTo Notebook (a specific tag can also be given):


```bash
    colomoto-docker    # or colomoto-docker -V 2025-01-01
```

3. Open the Jupiter Notebook and navigate to the `tutorials` folder to find the `PyDrugLogics` folder hosting the pydruglogics tutorial notebook.


See more about the CoLoMoTo Docker and Notebook in the [documentation](https://colomoto.github.io/colomoto-docker/README.html).<br/>

## Testing
1. To run all tests and check code coverage, you need to install test dependencies:
```bash
    pip install -r requirements.txt
    pip install -e .[test]
```

2. Then, from the repository root, run:

```bash
    pytest tests
```

You should see a coverage report at the end.

## Documentation

For full **PyDrugLogics** documentation, visit the [GitHub Documentation](https://druglogics.github.io/pydruglogics/).

## Quick Start Guide

Here's a simple example to get you started:

```python
from pydruglogics.model.BooleanModel import BooleanModel
from pydruglogics.input.TrainingData import TrainingData
from pydruglogics.input.Perturbations import Perturbation
from pydruglogics.input.ModelOutputs import ModelOutputs
from pydruglogics.execution.Executor import execute

# Initialize train and predict
model_outputs_dict = {
        "RSK_f": 1.0,
        "MYC": 1.0,
        "TCF7_f": 1.0
    }
model_outputs = ModelOutputs(input_dictionary=model_outputs_dict)

observations = [(["CASP3:0", "CASP8:0","CASP9:0","FOXO_f:0","RSK_f:1","CCND1:1"], 1.0)]
training_data = TrainingData(observations=observations)


drug_data = [['PI', 'PIK3CA', 'inhibits'],
            ['PD', 'MEK_f', 'activates'],
            ['CT','GSK3_f']]
perturbations = Perturbation(drug_data=drug_data)


boolean_model = BooleanModel(file='./ags_cascade_1.0/network.bnet', model_name='test', mutation_type='topology',
                                  attractor_tool='mpbn', attractor_type='trapspaces')

observed_synergy_scores = ["PI-PD", "PI-5Z", "PD-AK", "AK-5Z"]


ga_args = {
        'num_generations': 20,
        'num_parents_mating': 3,
        'mutation_num_genes': 3,
        'fitness_batch_size': 20
}

ev_args = {
        'num_best_solutions': 3,
        'num_of_runs': 30,
        'num_of_cores': 4
}


train_params = {
        'boolean_model': boolean_model,
        'model_outputs': model_outputs,
        'training_data': training_data,
        'ga_args': ga_args,
        'ev_args': ev_args
}

predict_params = {
        'perturbations': perturbations,
        'model_outputs': model_outputs,
        'observed_synergy_scores': observed_synergy_scores,
        'synergy_method': 'bliss'
}

# run train and predict
execute(train_params=train_params, predict_params=predict_params)
```

For a more detailed tutorial, please visit the [documentation](https://druglogics.github.io/pydruglogics/) or the [tutorial](https://github.com/druglogics/pydruglogics/blob/2400a153f15a884222f6fdabe705df1a5981ef54/tutorials/pydruglogics_tutorial.ipynb).

## Citing PyDrugLogics

If you use PyDrugLogics, please cite the paper:

*Flobak, Å., Zobolas, J. et al. (2023): Fine tuning a logical model of cancer cells to predict drug synergies: combining manual curation and automated parameterization. [DOI: 10.3389/fsysb.2023.1252961](https://www.frontiersin.org/journals/systems-biology/articles/10.3389/fsysb.2023.1252961/full)*

```bibtex
@Article{druglogics2023,
  title = {Fine tuning a logical model of cancer cells to predict drug synergies: combining manual curation and automated parameterization},
  author = {Flobak, Å., Zobolas, J. and Other Authors},
  journal = {Frontiers},
  year = {2023},
  month = {nov},
  doi = {10.3389/fsysb.2023.1252961},
  url = {https://www.frontiersin.org/journals/systems-biology/articles/10.3389/fsysb.2023.1252961/full},
}