import time
import logging
from typing import Any, Dict, List, Optional, Union
from pydruglogics.model.BooleanModel import BooleanModel
from pydruglogics.input.ModelOutputs import ModelOutputs
from pydruglogics.input.TrainingData import TrainingData
from pydruglogics.input.Perturbations import Perturbation
from pydruglogics.model.Evolution import Evolution
from pydruglogics.model.ModelPredictions import ModelPredictions
from pydruglogics.utils.PlotUtil import PlotUtil


def timed_execution(func):
    """
    Decorator to log the time taken by a function and provide detailed logging.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f'{func.__name__.capitalize()} started...')
        result = func(*args, **kwargs)
        logging.info(f'{func.__name__.capitalize()} completed in {time.time() - start_time:.2f} seconds.')
        return result
    return wrapper


@timed_execution
def train(boolean_model: BooleanModel, model_outputs: ModelOutputs,
          ga_args: Dict[str, Any], ev_args: Dict[str, Any], training_data: Optional[TrainingData] = None,
          save_best_models: bool = False, save_path: str = './results/models') -> List[BooleanModel]:
    """
    Train a Boolean Model using genetic algorithm and evolution strategy.
    Finds the models with the best fitness score.
    """
    evolution = Evolution(boolean_model=boolean_model, model_outputs=model_outputs,
                          training_data=training_data, ga_args=ga_args, ev_args=ev_args)
    best_boolean_models = evolution.run()
    if save_best_models:
        evolution.save_to_file_models(save_path)
    return best_boolean_models


@timed_execution
def predict(best_boolean_models: Optional[List[BooleanModel]], model_outputs: ModelOutputs,
            perturbations: Perturbation, observed_synergy_scores: List[str],
            synergy_method: str = 'bliss', run_parallel: bool = True, plot_roc_pr_curves: bool = True,
            save_predictions: bool = False, save_path: str = './results/predictions',
            model_directory: str = '', attractor_tool: str = '',
            attractor_type: str = '', cores: int = 4) -> None:
    """
    Predict model outcomes and plot the results. If best_boolean_models is None, loads models from model_directory.
    """
    if best_boolean_models is None and model_directory:
        logging.info(f"Loading models from directory: {model_directory}")
        model_predictions = ModelPredictions(
            model_outputs=model_outputs,
            perturbations=perturbations,
            model_directory=model_directory,
            attractor_tool=attractor_tool,
            attractor_type=attractor_type,
            synergy_method=synergy_method,
            boolean_models=None
        )
    else:
        model_predictions = ModelPredictions(
            boolean_models=best_boolean_models,
            perturbations=perturbations,
            model_outputs=model_outputs,
            synergy_method=synergy_method
        )

    model_predictions.run_simulations(run_parallel, cores)

    if plot_roc_pr_curves:
        PlotUtil.plot_roc_and_pr_curve(model_predictions.predicted_synergy_scores,
                                       observed_synergy_scores, synergy_method)
    if save_predictions:
        model_predictions.save_to_file_predictions(save_path)

def execute(train_params: Optional[Dict[str, Any]] = None,
            predict_params: Optional[Dict[str, Any]] = None) -> None:
    """
    Execute training and/or prediction based on the provided parameters.
    """
    start_time = time.time()
    best_boolean_models = None

    if train_params:
        best_boolean_models = train(**train_params)

    if predict_params and best_boolean_models:
        predict(best_boolean_models=best_boolean_models, **predict_params)

    if predict_params and not best_boolean_models:
        predict(best_boolean_models=None, **predict_params)

    if train_params and predict_params:
        total_time = time.time() - start_time
        logging.info(f'Total runtime for training and prediction: {total_time:.2f} seconds')
