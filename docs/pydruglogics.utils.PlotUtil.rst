pydruglogics.utils.PlotUtil
===========================

Plotting utilities for ROC and PR curves, including ensemble and confidence interval visualizations.

.. autofunction::  pydruglogics.utils.PlotUtil.PlotUtil.plot_roc_and_pr_curve

  Plot the ROC and PR Curves for one or multiple sets of predicted synergy scores.
  Plots both ROC and PR for model predictions, with optional support for model ensembles.

.. autofunction::  pydruglogics.utils.PlotUtil.PlotUtil.plot_pr_curve_with_ci

  Plot a Precision-Recall (PR) curve with confidence intervals computed from bootstrap sampling.
  Accepts a DataFrame of PR/confidence data and supports both continuous confidence bands
  or discrete error bars to visualize model uncertainty.