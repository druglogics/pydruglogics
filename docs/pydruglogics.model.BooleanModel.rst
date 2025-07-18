pydruglogics.model.BooleanModel
===============================

A Boolean network model with support for mutation, attractor calculation, perturbations, and input/output conversion.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.calculate_attractors

   Calculate attractors based on the chosen attractor_tool and attractor_type.

   **Parameters**

   - ``attractor_tool (str)``: Tool for attractor calculation. Possible values: ``'mpbn'``, ``'pyboolnet'``.
   - ``attractor_type (str)``: Type of attractor. Possible values: ``'stable_states'``, ``'trapspaces'``.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.calculate_global_output

   Calculates the (normalized) global output of the model.

   **Parameters**

   - ``model_outputs (ModelOutputs)``: Node weights.
   - ``normalized (bool, optional)``: Whether to normalize. Default: True.

   **Returns**

   - float: The (normalized) global output.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.from_binary

   Updates the Boolean Equations from a binary representation.

   **Parameters**

   - ``binary_representation (list)``: The binary representation of the Boolean Equations.
   - ``mutation_type (str)``: Mutation type: ``'topology'``, ``'balanced'``, or ``'mixed'``.

   **Returns**

   - list: Updated Boolean Equations.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.to_binary

   Converts the Boolean Equations to a binary representation.

   **Parameters**

   - ``mutation_type (str)``: The type of mutation.

   **Returns**

   - list: Binary representation of Boolean Equations.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.add_perturbations

   Adds perturbations to the Boolean Model.

   **Parameters**

   - ``perturbations (list)``: List of perturbations.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.print

    Print all Boolean equations as readable logic statements.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.clone

    Return a deep copy of the BooleanModel instance.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.reset_attractors

    Reset the attractor list.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.has_attractors

    Return True if attractors have been found.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.has_stable_states

    Return True if stable states exist.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.has_global_output

    Return True if the model has a computed global output.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.get_stable_states

    Return only the stable states from the attractors.

.. automethod:: pydruglogics.model.BooleanModel.BooleanModel.num_outputs

    Return the number of outputs (nodes).

.. autoattribute:: pydruglogics.model.BooleanModel.BooleanModel.model_name

    Get or set the model name.

.. autoattribute:: pydruglogics.model.BooleanModel.BooleanModel.boolean_equations

    Get or set the model's Boolean equations.

.. autoattribute:: pydruglogics.model.BooleanModel.BooleanModel.binary_boolean_equations

    Get or set the model's binary Boolean equations.

.. autoattribute:: pydruglogics.model.BooleanModel.BooleanModel.mutation_type

    The type of mutation used for the model.

.. autoattribute:: pydruglogics.model.BooleanModel.BooleanModel.global_output

    The current global output value.

.. autoattribute:: pydruglogics.model.BooleanModel.BooleanModel.attractors

    The current list of attractors.

.. autoattribute:: pydruglogics.model.BooleanModel.BooleanModel.attractor_tool

    The tool used for attractor calculation.

.. autoattribute:: pydruglogics.model.BooleanModel.BooleanModel.attractor_type

    The type of attractor calculation.
