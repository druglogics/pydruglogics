Train and Predict
=================

The PyDrugLogics package provides methods to train Boolean models and predict drug synergies.

To see the full Jupyter Notebook tutorial click `here <https://github.com/druglogics/pydruglogics/blob/main/tutorials/pydruglogics_tutorial.ipynb>`_.

This page details the main functions for executing training and prediction tasks: `train`, `predict`, and `execute`.

.. _train:

Train
-----

The `train` function optimizes a Boolean model using a Genetic Algorithm.

Training involves optimalization of the Boolean models
The process uses a **genetic algorithm** to iteratively refine the models, ensuring alignment with observations such as steady states or global outputs.

The pipeline uses the **PyGAD** Genetic Algoritm.
For more information about the **PyGAD.GA** click `here <https://pygad.readthedocs.io/en/latest/>`_.

The Genetic Algorithm
~~~~~~~~~~~~~~~~~~~~~~

1. **Initial Model Generation**: Boolean models are generated from the initial Boolean model by applying mutations on it.

2. **Fitness Calculation**: Each model is evaluated and fitness score is calculated. Fitness scores reflect how well a model's predictions match to the provided data.

3. **Selection**: Models with higher fitness scores are selected for further refinement. These models are combined (via crossover) to create a new generation of models.

4. **Crossover and Mutation**:

There are 3 possible mutation types:

a. **Balanced**: Only the link operator is mutated.

Example:

    .. raw:: html

        A, ((B) | C) <b style="color: red;">|</b> !(D) => A, ((B) | C) <b style="color: red;">&</b> !(D)



b. **Topology**: One or more nodes are removed or added. Minimum 1 regulator always stays in the equation. There can be no more regulator than was in the original equation.

Example:

    .. raw:: html

        A, ((B) | C) | <b style="color: green;">!(D)</b> => A, ((B) | C)

c. **Mixed**: Combines the balanced and topology mutations.

Example:

    .. raw:: html

        A, ((B) |  <b style="color: green;">C</b>) <b style="color: red;">|</b> !(D) => A, (B) <b style="color: red;">&</b> !(D)

5. **Stopping Conditions**: The process continues until either a specified fitness threshold is reached or the maximum number of generations is completed.

Attractor Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~

In Boolean networks, attractors represent the long-term behaviors of the system, including fixed points (stable states) or cyclic patterns.

Stable states are fixed-point attractors where the system remains indefinitely, representing steady biological conditions.

Trap spaces are subsets of states the system cannot leave, encompassing stable states and recurring cycles, and provide insight into network behaviors.

In the pipeline,  `MPBN <https://mpbn.readthedocs.io/>`_ and `PyBoolNet <https://pyboolnet.readthedocs.io/en/master/>`_ packages are used to compute stable states and trap spaces.

**Example of adding the attractor calculation methods:**

.. code-block:: python
   :class: copybutton

    # MPBN, trapspaces
    model = BooleanModel(file='./path/to/network.bnet', model_name='test', mutation_type='balanced',
    attractor_tool='mpbn', attractor_type='trapspaces')

    # PyBoolNet, stable_states
    model = BooleanModel(file='./path/to/network.bnet', model_name='test', mutation_type='topology',
    attractor_tool='pyboolnet', attractor_type='stable_states')

GA Fitness Score Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Steady State Responses**

Fitness is calculated by comparing the model's attractors to the observed steady states. The formula is:

.. math::

    \text{fitness} = \frac{\sum \text{matches}}{\#\text{responses}}

For example, an observed response:

.. code-block:: text

    A: 0   B: 1   C: 0

If the model predicts an attractor:

.. code-block:: text

    A: 0   B: 1   C: *

The fitness score is calculated as:

.. math::

    \text{fitness} = \frac{1 + 1 + (1 - |0 - 0.5|))}{3} = \frac{2.5}{3} \approx 0.8333

**2. Global Output Responses**

When evaluating global outputs, the fitness measures the deviation between predicted and observed outputs. The formula is:

.. math::

    \text{fitness} = 1 - |\text{globaloutput}_{\text{obs}} - \text{globaloutput}_{\text{pred}}|

Where:

- :math:`\text{globaloutput}_{\text{obs}}`: Observed global output from the training data:

- :math:`\text{globaloutput}_{\text{pred}}`: Predicted global output from the model:


To calculate the predicted global output (:math:`\text{globaloutput}_{\text{pred}}`), the following steps are needed:

First, compute the weighted average score (:math:`\text{globaloutput}_{\text{pred}}`) across all attractors:


.. math::

    \text{globaloutput}_{\text{pred}} = \frac{\sum_{j=1}^{k} \sum_{i=1}^{n} \text{ss}_{ij} \times w_{i}}{k}


Where:

- :math:`k`: Number of attractors of the model.
- :math:`n`: Number of nodes defined in the model outputs.
- :math:`\text{ss}_{ij}`: The state of node :math:`i` in the :math:`j`-th attractor (values can be 0, 1, or 0.5).
- :math:`w_{i}`: The weight associated with each node.

Next, normalize the global output (:math:`\text{globaloutput}_{\text{norm}}`) to the :math:`[0, 1]` range using the following equation:

.. math::

    \text{globaloutput}_{\text{norm}} = \frac{\text{globaloutput}_{\text{pred}} - \text{min}(\text{gl})}{\text{max}(\text{gl}) - \text{min}(\text{gl})} \quad

Where:

- :math:`\text{max}(\text{gl}) = \sum_{w_i > 0} w_i`: The sum of all positive weights.
- :math:`\text{min}(\text{gl}) = \sum_{w_i < 0} w_i`: The sum of all negative weights.

For example, consider a Boolean model with one attractor where the modeloutput nodes have the following states:

.. code-block:: text

    A: 0   B: 1   C:*

Model Ouputs:

.. code-block:: text

    A: 1   B: 1   C: -1

The fitness score is calculated as:

.. math::

    \text{globaloutput}_{\text{pred}} = (0 \times 1) + (1 \times 1) + (0.5 \times -1) = 0.5


Next, normalize the predicted global output using the following values:

.. math::

    \text{min}(\text{gl}) = (-1) \quad \text{max}(\text{gl}) = (+1)

Normalize the global output:

.. math::

    \text{globaloutput}_{\text{norm}} = \frac{\text{globaloutput}_{\text{pred}} - \text{min}(\text{gl})}{\text{max}(\text{gl}) - \text{min}(\text{gl})}

Substitute the values:

.. math::

    \text{globaloutput}_{\text{norm}} = \frac{0.5 - (-1)}{1 - (-1)} = \frac{1.5}{2} = 0.75

Finally, calculate the fitness score using the observed global output (:math:`\text{globaloutput}_{\text{obs}} = 1`):

.. math::

    \text{fitness} = 1 - |\text{globaloutput}_{\text{obs}} - \text{globaloutput}_{\text{norm}}|

Substitute the values:

.. math::

    \text{fitness} = 1 - |1 - 0.75| = 1 - 0.25 = 0.75

Therefore, the fitness score is 0.75, reflecting the degree of alignment between the model's predicted and observed global outputs.


Arguments
~~~~~~~~~~

**1. Arguments for the Genetic Algorithm (`ga_args`)**

The pipeline uses the `PyGAD` Genetic Algoritm.
For more information about the `PyGAD.GA` initialization click `here <https://pygad.readthedocs.io/en/latest/pygad.html#init>`_.

.. code-block:: python

    ga_args = {
        'num_generations': 20,
        'num_parents_mating': 3,
        'mutation_num_genes': 10,
        'fitness_batch_size': 20, # should be the same as the num_generations
        'crossover_type': 'single_point',
        'mutation_type': 'random',
        'keep_elitism': 6,
        # 'stop_criteria': 'reach_95'
    }

**2. Arguments for the Evolution (`ev_args`)**

- `num_best_solutions`: Number of the best solutions per Evolution run.
- `num_of_runs`: Number of running the Evolution
- `num_of_cores`: Maximum number of cores for calculations
- `num_of_init_mutation`: Number of mutated genes in the initial population.


.. code-block:: python

    ev_args = {
        'num_best_solutions': 3,
        'num_of_runs': 50,
        'num_of_cores': 4,
        'num_of_init_mutation': 12
    }


Code Example
~~~~~~~~~~~~

There are 2 possible ways to initialize run train on a Boolean model:

1. Initialize the **train** function

.. code-block:: python

    from pydruglogics.execution.Executor import train

    best_boolean_models = train(boolean_model=boolean_model_bnet, model_outputs=model_outputs,
    training_data=training_data, ga_args=ga_args, ev_args=ev_args)

2. Initialize the **executor** function

.. code-block:: python

    from pydruglogics.execution.Executor import execute

    train_params = {
        'boolean_model': boolean_model_bnet,
        'model_outputs': model_outputs,
        'training_data': training_data,
        'ga_args': ga_args,
        'ev_args': ev_args,
        'save_best_models': True,
        'save_path': './models'
    }
    execute(train_params=train_params)

.. note::

   If the Training Data is not provided, the **train** function will calculate the fitness using the Global Output
   Response, assuming a default globaloutput value of 1.

Example Results
~~~~~~~~~~~~~~~
Saved .bnet File:

.. code-block:: text

    # 2024_11_10, 2019
    # Evolution: 1 Solution: 1
    # Fitness Score: 0.998
    A, ((B) | C)
    B, !(D)
    C, (E) & !(F)
    D, (A) | !(C)

Output
~~~~~~
- **Optimized Boolean model**: The Boolean model fitted to the training data.

.. _predict:

Predict
-------

The `predict` function evaluates the trained Boolean model against drug perturbations to predict synergy scores.

The prediction process uses an ensemble of trained Boolean models and a list of drug perturbations to evaluate their
effects on a biological network. For each perturbation, the Boolean models are modified to reflect the drug's impact
on target nodes (e.g., inhibiting or activating specific proteins). The perturbed models are then simulated to compute
their responses.

Steps in Prediction
~~~~~~~~~~~~~~~~~~

1. **Apply Perturbations**: Perturbations (from the drug panel) are applied to each Boolean model. This modifies the
Boolean equations of target nodes to simulate the effects of the drugs.

- If the drug inhibits the target node, the equation will be set to: A, 0.
- If the drug activates the target node, the equation will be B, 1.

2. **Simulate Responses**: Attractors are calculated for the perturbed models, and global output response values are computed using the model outputs.

3. **Evaluate Synergy Scores**:

   - The global output responses are used to assess whether drug combinations are synergistic or antagonistic.
   - The pipeline uses **Bliss Independence** or **Highest Single Agent (HSA)**
   - The predicted synergy scores are compared with the observed synergy scores. This determines how accurate the predicted synergy scores are.



Code Example
~~~~~~~~~~~~
There are 2 possible ways to initialize run predict:

1. Initialize the **predict** function:

.. code-block:: python

    from pydruglogics.execution.Executor import predict

    predict(best_boolean_models=best_boolean_models, model_outputs=model_outputs, perturbations=perturbations,
                 observed_synergy_scores=observed_synergy_scores, synergy_method='bliss', run_parallel= True,
                 plot_roc_pr_curves=True, save_predictions=False, cores=4)


2. Initialize the **executor** function:

.. code-block:: python

    predict_params = {
        'perturbations': perturbations,
        'model_outputs': model_outputs,
        'observed_synergy_scores': observed_synergy_scores,
        'synergy_method': 'hsa',
        'plot_roc_pr_curves': True,
        'save_predictions': False,
        # 'cores': 4,
        # 'save_path': './predictions',
        # 'model_directory': './models/example_models',
        # 'attractor_tool': 'mpbn',
        # 'attractor_type':  'stable_states'
    }

    from pydruglogics.execution.Executor import execute

    execute(predict_params=predict_params)


Example Results
~~~~~~~~~~~~~~~

Response Matrix

.. code-block:: text

            e1_s1   e1_s2   e1_s3   e2_s1
    PI-PD    NA   -1.0   NA    3.0
    PI-CT    3.0   2.0   NA    3.0

Predicted Synergies

.. code-block:: text

    perturbation_name   synergy_score
    PI-PD              -0.158
    PI-CT              0.003

**ROC an PR Curves**

.. image:: /images/predict_roc_pr.png
   :alt: Results After Execution
   :width: 100%
   :align: center

Output
~~~~~~
- **Predicted synergy scores**: A list of synergy scores for each perturbation.
- **Response Matrix:** A matrix containing the resonses for each perturben Boolean model and the prerturbation.

.. _execute:

Execute
-------

The `execute` method is a streamlined function that combines training and prediction in one step. It takes `train_params` and `predict_params` as arguments to define the configuration for training and prediction.

Arguments
~~~~~~~~~
1. **train_params**: A dictionary specifying the parameters for training.
2. **predict_params**: A dictionary specifying the parameters for prediction.

Code Example
~~~~~~~~~~~~
.. code-block:: python

    ga_args = {
        'num_generations': 20,
        'num_parents_mating': 3,
        'mutation_num_genes': 10,
        'fitness_batch_size': 20, # should be the same as the num_generations
        'crossover_type': 'single_point',
        'mutation_type': 'random',
        'keep_elitism': 6,
        # 'stop_criteria': 'reach_95'
    }

    ev_args = {
        'num_best_solutions': 3,
        'num_of_runs': 50,
        'num_of_cores': 4,
        'num_of_init_mutation': 12
    }

    train_params = {
        'boolean_model': boolean_model_bnet,
        'model_outputs': model_outputs,
        'training_data': training_data,
        'ga_args': ga_args,
        'ev_args': ev_args,
        'save_best_models': False,
        # 'save_path': './models'
    }
    predict_params = {
        'perturbations': perturbations,
        'model_outputs': model_outputs,
        'observed_synergy_scores': observed_synergy_scores,
        'synergy_method': 'bliss',
        'plot_roc_pr_curves': True,
        'save_predictions': False,
        # 'cores': 4,
        # 'save_path': './predictions',
        # 'model_directory': './models/example_models',
        # 'attractor_tool': 'mpbn',
        # 'attractor_type':  'stable_states'
      }

    from pydruglogics.execution.Executor import execute
    execute(train_params=train_params, predict_params=predict_params)

Example Results
~~~~~~~~~~~~~~~
The `execute` method produces both the outputs of the `train` and `predict` functions.

**Example Outputs from Train**

.. code-block:: text

    # 2024_11_10, 2019
    # Evolution: 1 Solution: 1
    # Fitness Score: 0.998
    A, ((B) | C)
    B, !(D)
    C, (E) & !(F)
    D, (A) | !(C)


**Example Outputs from Predict**

Response Matrix

.. code-block:: text

            e1_s1   e1_s2   e1_s3   e2_s1
    PI-PD    NA   -1.0   NA    3.0
    PI-CT    3.0   2.0   NA    3.0

Predicted Synergies

.. code-block:: text

    perturbation_name   synergy_score
    PI-PD              -0.158
    PI-CT              0.003

**ROC an PR Curves**

.. image:: /images/predict_roc_pr.png
   :alt: Results After Execution
   :width: 100%
   :align: center
