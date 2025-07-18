pydruglogics.model.InteractionModel
===================================

Loads, parses, and manages network interactions, allowing for construction of Boolean models from `.sif` files,
with utilities for filtering and extracting regulators/targets.


.. automethod:: pydruglogics.model.InteractionModel.InteractionModel.size

Return the number of interactions.


.. automethod:: pydruglogics.model.InteractionModel.InteractionModel.print

Print the current interaction model in a human-readable form.


.. automethod:: pydruglogics.model.InteractionModel.InteractionModel.get_interaction

Return the dictionary of the interaction at a given index.


.. automethod:: pydruglogics.model.InteractionModel.InteractionModel.get_target

Return the target node of the interaction at the given index.


.. automethod:: pydruglogics.model.InteractionModel.InteractionModel.get_activating_regulators

Return the activating regulators for the interaction at the given index.


.. automethod:: pydruglogics.model.InteractionModel.InteractionModel.get_inhibitory_regulators

Return the inhibitory regulators for the interaction at the given index.


.. automethod:: pydruglogics.model.InteractionModel.InteractionModel.model_name

Set the model name.

