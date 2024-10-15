import time
import logging
from typing import Any, Dict, List
from pydruglogics.model.BooleanModel import BooleanModel
from pydruglogics.input.ModelOutputs import ModelOutputs
from pydruglogics.input.TrainingData import TrainingData
from pydruglogics.input.Perturbations import Perturbation
from pydruglogics.model.Evolution import Evolution
from pydruglogics.model.ModelPredictions import ModelPredictions


class Executor:
    def __init__(self):
        self.best_boolean_models = None
        self._time_evolution = 0.0
        self._time_predictions = 0.0

    def run_evolution(self,
                      boolean_model: BooleanModel = None,
                      model_outputs: ModelOutputs = None,
                      ga_args: Dict[str, Any] = None,
                      ev_args: Dict[str, Any] = None,
                      training_data: TrainingData = None,
                      save_best_models: bool = False,
                      save_path: str = './models'):
        try:
            if not boolean_model:
                raise ValueError("Boolean model is required.")
            if not model_outputs:
                raise ValueError("Model outputs are required.")
            if not ga_args or not ev_args:
                raise ValueError("GA arguments and EV arguments must be provided.")
            start_time = time.time()
            logging.info('Starting evolution...')

            evolution = Evolution(
                boolean_model=boolean_model,
                model_outputs=model_outputs,
                training_data=training_data,
                ga_args=ga_args,
                ev_args=ev_args)
            self.best_boolean_models = evolution.run()
            if save_best_models:
                evolution.save_to_file_models(save_path)

            self._time_evolution = time.time() - start_time
            logging.info(f'Evolution runtime: {self._time_evolution:.2f} seconds')

        except ValueError as ve:
            logging.error(f"ValueError in evolution: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error during evolution: {str(e)}")
            raise

    def run_predictions(self,
                        best_boolean_models: List[BooleanModel] or None,
                        model_outputs: ModelOutputs = None,
                        perturbations: Perturbation = None,
                        observed_synergy_scores: List[str] = None,
                        run_parallel: bool = True,
                        save_predictions: bool = False,
                        save_path: str = './predictions',
                        synergy_method='hsa',
                        model_directory: str = '',
                        attractor_tool: str = ''):

        try:
            if not model_outputs:
                raise ValueError("Model outputs are required.")
            if not observed_synergy_scores:
                raise ValueError("Observed synergy scores are required.")

            start_time = time.time()
            logging.info('Starting predictions...')

            if self.best_boolean_models is None and best_boolean_models is None:
                model_predictions = ModelPredictions(
                    perturbations=perturbations,
                    model_outputs=model_outputs,
                    observed_synergy_scores=observed_synergy_scores,
                    model_directory=model_directory,
                    attractor_tool=attractor_tool,
                    synergy_method=synergy_method,
                )

            else:
                model_predictions = ModelPredictions(
                    boolean_models=self.best_boolean_models if self.best_boolean_models is not None else best_boolean_models,
                    perturbations=perturbations,
                    model_outputs=model_outputs,
                    observed_synergy_scores=observed_synergy_scores,
                    synergy_method=synergy_method,
                )

            model_predictions.run_simulations(run_parallel)
            model_predictions.plot_roc_and_pr_curve()
            if save_predictions:
                model_predictions.save_to_file_predictions(save_path)

            self._time_predictions = time.time() - start_time
            logging.info(f'Predictions runtime: {self._time_predictions:.2f} seconds')
        except ValueError as ve:
            logging.error(f"Value Errror in running predictions: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error during predictions: {str(e)}")
            raise

    def execute(self, run_evolution=True, run_predictions=True, evolution_params=None, prediction_params=None):
        if run_evolution and not run_predictions:
            if not evolution_params:
                raise ValueError("Missing parameters for running evolution.")
            self.run_evolution(**evolution_params)

        if run_predictions and not run_evolution:
            if not prediction_params:
                raise ValueError("Missing parameters for running predictions.")
            self.run_predictions(**prediction_params)
        else:
            if not prediction_params and not run_evolution:
                raise ValueError("Missing parameters for running evolution and predictions.")
            self.run_evolution(**evolution_params)
            self.run_predictions(**prediction_params)
            logging.info(f'Evolution runtime: {self._time_evolution:.2f}\n'
                       f'Total runtime: {self._time_evolution + self._time_predictions:.2f} seconds')

    def display_parameter_hints(self):
        hints = {
            'boolean_model': 'Instance of a BooleanModel, required for the evolution process.',
            'ga_args': 'Dictionary of genetic algorithm settings.',
            'model_outputs': 'Definitions of model outputs needed for evaluating the model predictions.',
            'training_data': 'Training data for optimizing the boolean model.',
            'perturbations': 'List of perturbations to apply for predictions, typically drug combinations.',
            'observed_synergy_scores': 'List of observed synergy scores for validating model predictions.',
            'model_directory': 'Directory path to save or load models.',
            'verbosity': 'Verbosity level for logging output (0-3: higher for more detailed logs).'
        }
        print('Configuration Parameters for Executor:')
        for param, desc in hints.items():
            print(f'- {param}: {desc}')
