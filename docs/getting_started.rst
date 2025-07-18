Getting Started
===============


.. _overview:

Overview
--------

.. image:: https://raw.githubusercontent.com/druglogics/pydruglogics/main/logo.png
   :alt: PyDrugLogics Logo
   :align: center

.. raw:: html

    <br\>
    <p align="center">
    <a href="https://badge.fury.io/py/pydruglogics">
        <img src="https://img.shields.io/pypi/v/pydruglogics" alt="PyPI version">
    </a>
    <a href="https://github.com/druglogics/pydruglogics/actions/workflows/run-tests.yml">
        <img src="https://github.com/druglogics/pydruglogics/actions/workflows/run-tests.yml/badge.svg" alt="Test Status">
    </a>
    <a href="https://github.com/druglogics/pydruglogics/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3">
    </a>
    <a href="https://druglogics.github.io/pydruglogics/">
        <img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Documentation Status">
    </a>
    </p>

    <p>
        PyDrugLogics is a Python package designed for constructing, optimizing Boolean Models and performing in-silico perturbations of the models.
    </p>
    <h3>Core Features:</h3>
    <ul>
        <li>Construct Boolean model from <code>.sif</code> file</li>
        <li>Load Boolean model from <code>.bnet</code> file</li>
        <li>Optimize Boolean model</li>
        <li>Generate perturbed Boolean models</li>
        <li>Evaluate drug synergies</li>

    </ul>

.. _installation:

Installation
------------


**PyDrugLogics** can be installed via **PyPI**, **Conda**, or **directly from the source**.

Install PyDrugLogics from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The process involves two steps to install the PyDrugLogics core package and its necessary external dependencies.

1. **Install PyDrugLogics via pip**

.. code-block:: bash

   pip install pydruglogics

This will install the PyDrugLogics package and handle all dependencies automatically.

Install PyDrugLogics via conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    conda install szlaura::pydruglogics

Install from Source
~~~~~~~~~~~~~~~~~~~
For the latest development version, clone the repository and install it directly:

.. code-block:: bash

    git clone https://github.com/druglogics/pydruglogics.git
    cd pydruglogics
    pip install .

.. _colomoto_notebook_environment:

CoLoMoTo Notebook Environment
-----------------------------

PyDrugLogics is available in the CoLoMoTo Docker and Notebook starting from version **2025-01-01**.

Setup CoLoMoTo Docker and Notebook
~~~~~~~~~~~~~~~~~

1. Install the helper script in a terminal:

.. code-block:: bash

    pip install -U colomoto-docker

2. Start the CoLoMoTo Notebook (a specific tag can also be given):

.. code-block:: bash

    colomoto-docker    # or colomoto-docker -V 2025-01-01


3. Open the Jupiter Notebook and navigate to the **tutorials** folder to find the **PyDrugLogics** folder hosting the pydruglogics tutorial notebook.

See more about the CoLoMoTo Docker and Notebook in the `documentation <https://colomoto.github.io/colomoto-docker/README.html>`_.


