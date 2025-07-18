pydruglogics.model.Statistics
=============================

Provides statistical and evaluation functions for analyzing the performance of Boolean model predictions.
Supports PR curve calculation, confidence intervals, bootstrapping, and comparison between model runs.

.. automethod:: pydruglogics.statistics.Statistics.sampling_with_ci

   Performs sampling with confidence interval calculation and plots the PR curve.

   **Parameters**

   - ``boolean_models (list)``: BooleanModel instances.
   - ``observed_synergy_scores (list)``: Observed synergy scores.
   - ``model_outputs``: Model outputs.
   - ``perturbations (list)``: Perturbations to apply.
   - ``synergy_method (str, optional)``: 'hsa' or 'bliss'.
   - ``repeat_time (int, optional)``: Repeats. Default: 10.
   - ``sub_ratio (float, optional)``: Proportion to sample. Default: 0.8.
   - ``boot_n (int, optional)``: Bootstrap samples. Default: 1000.
   - ``confidence_level (float, optional)``: CI level. Default: 0.9.
   - ``plot (bool, optional)``: Plot the PR curve. Default: True.
   - ``plot_discrete (bool, optional)``: Discrete points. Default: False.
   - ``save_result (bool, optional)``: Save to .tab file. Default: True.
   - ``with_seeds (bool, optional)``: Use fixed seed. Default: True.
   - ``seeds (int, optional)``: Random seed. Default: 42.

.. automethod:: pydruglogics.statistics.Statistics.compare_two_simulations

    Compares ROC and PR curves for two sets of evolution results.

    **Parameters**

    - ``boolean_models1 (list)``: First set.
    - ``boolean_models2 (list)``: Second set.
    - ``observed_synergy_scores (list)``: Observed scores.
    - ``model_outputs``: Model outputs.
    - ``perturbations (list)``: Perturbations.
    - ``synergy_method (str, optional)``: 'hsa' or 'bliss'.
    - ``label1 (str, optional)``: Label for first.
    - ``label2 (str, optional)``: Label for second.
    - ``normalized (bool, optional)``: Normalize first. Default: True.
    - ``plot (bool, optional)``: Show ROC/PR. Default: True.
    - ``save_result (bool, optional)``: Save results. Default: True.
