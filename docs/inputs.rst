
Inputs
======

PyDrugLogics requires various inputs to train Boolean Models and predict drug synergies.
These inputs define the interaction model, training data, model outputs, perturbations, and observed synergy scores,
each playing an essential role in the software pipeline.

To see the full Jupyter Notebook tutorial click `here <https://github.com/druglogics/pydruglogics/blob/main/tutorials/pydruglogics_tutorial.ipynb>`_.

Below is a detailed guide to each input type and its structure.

.. _input_files:

Input Files
-----------

PyDrugLogics supports the following input files to load or construct Boolean Models:

Boolean Model from .sif File
~~~~~~~~~~

The `.sif` file format represents network interactions using activation and inhibition relationships.
Each row defines an interaction between nodes in the network.


**Key notations:**

- `->`: Activation
- `-|`: Inhibition



**Example:**

.. code-block:: text
   :class: copybutton

   A -> B
   C -> A
   B -| C

**Code Example for Loading Interactions for construction a Boolean Model**

.. code-block:: python
   :class: copybutton

   # Initialize InteractonModel
   from pydruglogics.model.InteractionModel import InteractionModel

   interaction_model = InteractionModel(interactions_file='./path/to/network.sif', model_name='',
   remove_self_regulated_interactions=False, remove_inputs=False, remove_outputs=False)

   # Initialize BooleanModel
   from pydruglogics.model.BooleanModel import BooleanModel

   boolean_model = BooleanModel(model=model, model_name='test', mutation_type='balanced',
   attractor_tool='mpbn', attractor_type='trapspaces')

.. note::

   A `.sif` file defines one Boolean Model.

Boolean Model from .bnet File
~~~~~~~~~~~

The `.bnet` format is used for defining a Boolean network, where nodes represent variables,
and their activation expressions define relationships and dependencies among them. Each node's state is determined
by logical expressions.
Logical operators are used to specify relationships:

- `&`: Conjunction
- `|`: Disjunction
- `!`: Negation


**Example:**

.. code-block:: text
   :class: copybutton

   A, (B) & !(C)
   B, ((A) | C)
   C, (B)

**Code Example for Loading Model from .bnet**

.. code-block:: python
   :class: copybutton

   from pydruglogics.model.BooleanModel import BooleanModel

   model = BooleanModel(file='./path/to/network.bnet', model_name='test', mutation_type='balanced',
   attractor_tool='mpbn', attractor_type='stable_states')

.. note::

   A `.bnet` file defines one Boolean Model.

.. _training_data:

Training Data
-------------

The training data file contains condition-response pairs, and a weight that are essential for
evaluating the performance of Boolean models during the genetic algorithm's evolutionary process.
A fitness score is calculated for the condition(s)-response(s), reflecting how well (0-worst, 1-best) the model aligns with the training data.

Format
~~~~~~~

Each observation consists of:

1. **Condition**: -
2. **Response**: Specifies the node activity levels as a tab-separated list.
3. **Weight**: Once the fitness values have been calculated, this value is used to weight each condition-response pair
and calculate the overall weighted average fitness score of the model fitted to the training data

Types of Training Data
~~~~~~~~~~~~~~~~~~~~~~


1. **Unperturbed Condition - Steady State Response**

This training type describes the system's steady state, where activity values are assigned to nodes in the range [0, 1].
These values represent the observed state of the system and are compared against the model's attractors to calculate fitness.

Example:

.. code-block:: text
  :class: copybutton

  Condition
  -
  Response
  A:0 B:1 C:0 D:0.453
  Weight:1

2. **Unperturbed Condition - Global Output Response**

This training type specifies the system's behavior under no perturbation, typically used for studying proliferation in the networks.
The response is defined as `globaloutput:<value>` in the range [0, 1], with fitness calculated based on how close
the predicted global output is to the observed value.

Example:

.. code-block:: text
  :class: copybutton

  Condition
  -
  Response
  globaloutput:1
  Weight:1


Initialization Options
~~~~~~~~~~~~~~~~~~~~~~

**1. Load Training Data from File**

This method allows loading training data directly from a file. The file can be in a format such as `training_data.tab`
or `training_data`, containing input in a format like this:

.. code-block:: text

    # training data
    Condition
    -
    Response
    A:0	B:0	C:0
    Weight:1

Where the responses are tab-separated.

Example:

.. code-block:: python
   :class: copybutton

   from pydruglogics.input.TrainingData import TrainingData

   training_data = TrainingData(input_file='./path/to/training')

**2. Direct Initialization**

This method initializes the training data using Python data structures. The responses and weights are provided as
a list of tuples.

Example:

.. code-block:: python
   :class: copybutton

   from pydruglogics.input.TrainingData import TrainingData

   observations = [(["A:1", "B:0", "C:0.5"], 1)]
   training_data = TrainingData(observations=observations)

.. _model_outputs:

Model Outputs
-------------

The `model outputs` defines network nodes and their integer weights, determining their contribution to global
signaling outputs (e.g., cell proliferation or death).

Format
~~~~~~~

Each model output contains:

1. **Node name**: string value.
2. **Weight** (positive for proliferation, negative for death): continuous numeric value.


Initialization Options
~~~~~~~~~~~~~~~~~~~~~~

**1. Load Model Outputs from File**

This method allows loading model outputs directly from a file. The file can be in a format such as `modeloutputs.tab`
or `modeloutputs`, containing input in a format like this:

.. code-block:: text
   :class: copybutton

   # Name   Weight
   A  1.0
   B  -1.0
   C  -2.0

Where the names and weights are tab-separated.

Example:

.. code-block:: python
   :class: copybutton

   from pydruglogics.input.ModelOutputs import ModelOutputs

   model_outputs = ModelOutputs(input_file='./path/to/modeloutputs')

**2. Direct Initialization**

This method initializes the model outputs using Python data structures. Outputs are provided as a dictionary, with keys representing node names and values representing their corresponding output values.

Example:

.. code-block:: python
  :class: copybutton

  from pydruglogics.input.ModelOutputs import ModelOutputs

  model_outputs_dict = {
      "A": 1.0,
      "B": -1.0,
      "C": -2.0
  }
  model_outputs = ModelOutputs(input_dictionary=model_outputs_dict)

.. _perturbations:

Perturbations
-------------

Perturbations combine drugs applied to the system. The perturbations list contains all drug combinations to be tested.
The drug data contains the effect of each drug on the nodes.

.. note::

   Only 1- and 2-drug perturbations are allowed. Perturbations with more than two drugs are not supported.

Initialization From Dictionary
~~~~~~~~~~~~~~~~~~~~~~

You can define both `drug_data` and `perturbation_data`, or just `drug_data`:

1. **Define `drug_data` and `perturbation_data`:**
Provide a list of drugs, where each drug entry specifies:

a. **Drug data:**

- **Drug name**: Unique name of the drug.
- **Target(s)**: The node(s) in the network affected by the drug.
- **Effect**: This specifies how the drug influences the target and can take the following values:
    - `activates`: The drug increases the target's activity.
    - `inhibits`: The drug decreases the target's activity (this is the default if no effect is specified).

b. **Perturbation data:**

- **Perturbations**: One or two-drug combinations. The pipeline handles only single and tro-drug combinations.


If both `drug_data` and `perturbation_data` are defined, the explicitly provided perturbations will be used.

Example:

.. code-block:: python
   :class: copybutton

   # Define drug_data
   drug_data = [
       ['PI', 'A', 'inhibits'],     # PI inhibits target A
       ['PD', 'B', 'activates'],    # PD activates target B
       ['CT', 'C, D, E'],           # CT inhibits targets C, D, and E
       ['BI', 'F, G'],              # BI inhibits targets F and G
       ['PK', 'H'],                 # PK inhibits target H
       ['AK', 'I']                  # Ak inhibits target I
   ]

   # Define perturbation_data
   perturbation_data = [
       ['PI'],
       ['PD'],
       ['CT'],
       ['BI'],
       ['PK'],
       ['AK'],
       ['PI', 'PD'],
       ['PD', 'PK'],
       ['CT', 'AK'],
       ['BI', 'PK'],
       ['BI', 'AK']

   perturbations = Perturbation(drug_data=drug_data, perturbation_data=perturbation_data)

2. **Define only `drug_data`:**

If no `perturbation_data` is provided, the pipeline will automatically generate all possible
two-drug combinations from the `drug_data`.

Example:

.. code-block:: python
   :class: copybutton

   drug_data = [
       ['PI', 'A', 'inhibits'],     # PI inhibits target A
       ['PD', 'B', 'activates'],    # PD activates target B
       ['CT', 'C, D, E'],           # CT inhibits targets C, D, and E
       ['BI', 'F, G'],              # BI inhibits targets F and G
       ['PK', 'H'],                 # PK inhibits target H
       ['AK', 'I']                  # Ak inhibits target I
   ]

   perturbations = Perturbation(drug_data=drug_data)

.. note::

    - If `perturbation_data` is not provided, it will be automatically calculated to include all drug combinations from the `drug_data`.
    - The `effect` field in `drug_data` is optional. If omitted, the pipeline assumes the effect is `inhibits`.
    - Multiple targets can be specified for a single drug by listing them in the `Target(s)` field, separated by commas.
    - Valid options for the `effect` field are: `activates` and `inhibits`.


.. _observed_synergy_scores:

Observed Synergy Scores
------------------------

Observed synergy scores are ground truth data used to evaluate model predictions. They are typically derived from experimental datasets or literature sources.

Example:

.. code-block:: python
   :class: copybutton


   observed_synergy_scores = ["PI-PD", "PD-AK"]

