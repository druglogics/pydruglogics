from pydruglogics.model.Evolution import Evolution
from pydruglogics.model.ModelPredictions import ModelPredictions


class Executor:
    def __init__(self, boolean_model=None, ga_args=None, model_outputs=None, training_data=None,
                 perturbations=None, observed_synergy_scores=None, verbosity=1):
        self.boolean_model = boolean_model
        self.ga_args = ga_args
        self.model_outputs = model_outputs
        self.training_data = training_data
        self.perturbations = perturbations
        self.observed_synergy_scores = observed_synergy_scores
        self.verbosity = verbosity

    def run(self, save_path='models', num_best_solutions=3, num_runs=50, num_cores=4, synergy_method='hsa'):
        evolution = Evolution(
            boolean_model=self.boolean_model,
            model_outputs=self.model_outputs,
            training_data=self.training_data,
            num_best_solutions=num_best_solutions,
            num_runs=num_runs,
            num_cores=num_cores,
            ga_args=self.ga_args
        )

        best_boolean_models = evolution.run()
        evolution.save_to_file_models(save_path)
        self.run_predictions(best_boolean_models, synergy_method)

    def run_predictions(self, best_boolean_models, synergy_method):
        model_predictions = ModelPredictions(
            boolean_models=best_boolean_models,
            perturbations=self.perturbations,
            model_outputs=self.model_outputs,
            observed_synergy_scores=self.observed_synergy_scores,
            synergy_method= synergy_method
        )
        model_predictions.run_simulations()
        model_predictions.plot_roc_and_pr_curve()
