pydruglogics.utils.PlotUtil
===========================

Plotting utilities for ROC and PR curves, including ensemble and confidence interval visualizations.

.. automethod:: pydruglogics.utils.PlotUtil.PlotUtil.plot_roc_and_pr_curve

    Plot the ROC and PR Curves for one or multiple sets of predicted synergy scores.
    Plots both ROC and PR for model predictions, with optional support for model ensembles.

    **Parameters**:

    - ``predicted_synergy_scores``: List of predictions or single set.
    - ``observed_synergy_scores``: Observed synergy scores.
    - ``synergy_method (str)``: Method used for scoring (plot titles).
    - ``labels (list, optional)``: Optional labels for each set.

.. automethod:: pydruglogics.utils.PlotUtil.PlotUtil.plot_pr_curve_with_ci

    Plot a Precision-Recall (PR) curve with confidence intervals computed from bootstrap sampling.
    Accepts a DataFrame of PR/confidence data and supports both continuous confidence bands
    or discrete error bars to visualize model uncertainty.

    **Parameters**

    - ``pr_df (DataFrame)``: PR/confidence data.
    - ``auc_pr (float)``: Area under PR.
    - ``boot_n (int)``: Bootstrap samples.
    - ``plot_discrete (bool)``: Discrete error bars or band.
