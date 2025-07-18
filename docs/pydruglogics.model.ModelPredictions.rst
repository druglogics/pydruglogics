pydruglogics.model.ModelPredictions
===================================

Simulates Boolean models under perturbations, computes responses, and calculates synergy scores using HSA or Bliss methods.


.. automethod:: pydruglogics.model.ModelPredictions.ModelPredictions.run_simulations

    Runs simulations on the Boolean Models with the perturbations, either in parallel or serially.

    **Parameters**

    - ``parallel (bool, optional)``: Whether to run simulations in parallel. Default: True.
    - ``cores (int, optional)``: Number of CPU cores to use. Default: 4.

.. automethod:: pydruglogics.model.ModelPredictions.ModelPredictions.save_to_file_predictions

    Save the prediction matrix and synergy results to disk.

    **Parameters**

    - ``base_folder (str, optional)``: Directory to save results.

.. automethod:: pydruglogics.model.ModelPredictions.ModelPredictions.get_prediction_matrix

    Print or retrieve the prediction matrix (perturbed responses).
