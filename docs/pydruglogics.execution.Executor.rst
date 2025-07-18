pydruglogics.execution.Executor
===============================

Provides training and prediction functions for Boolean models.

.. autofunction:: pydruglogics.execution.Executor.train

    Train a Boolean Model using genetic algorithm and evolution strategy.
    Finds the models with the best fitness score.

    **Parameters**

    - ``boolean_model (BooleanModel)``: The model to be trained.
    - ``model_outputs (ModelOutputs)``: The outputs to be optimized.
    - ``ga_args (dict)``: Genetic algorithm parameters.
    - ``ev_args (dict)``: Evolution strategy parameters.
    - ``training_data (TrainingData, optional)``: Training data for the model.
    - ``save_best_models (bool)``: Whether to save the best models.
    - ``save_path (str)``: Path to save models.

    **Returns**

    - ``List[BooleanModel]``: List of models with the best fitness.


.. autofunction:: pydruglogics.execution.Executor.predict

    Predict model outcomes and plot the results.

    **Parameters**

    - ``best_boolean_models (list of BooleanModel, optional)``: Models to use for prediction.
    - ``model_outputs (ModelOutputs)``: The outputs to be predicted.
    - ``perturbations (Perturbation)``: Perturbation scenarios.
    - ``observed_synergy_scores (list of str)``: Observed synergy scores.
    - ``synergy_method (str)``: Method for calculating synergy ('bliss', etc.).
    - ``run_parallel (bool)``: Whether to run in parallel.
    - ``plot_roc_pr_curves (bool)``: Whether to plot ROC and PR curves.
    - ``save_predictions (bool)``: Whether to save predictions.
    - ``save_path (str)``: Path to save predictions.
    - ``model_directory (str)``: Directory to load models from if not provided.
    - ``attractor_tool (str)``: Tool for attractor analysis.
    - ``attractor_type (str)``: Type of attractor analysis.
    - ``cores (int)``: Number of CPU cores to use.


.. autofunction:: pydruglogics.execution.Executor.execute
    Runs the train and predict functions.

    **Parameters**

    - ``train_params (dict, optional)``: Parameters for training.
    - ``predict_params (dict, optional)``: Parameters for prediction.

