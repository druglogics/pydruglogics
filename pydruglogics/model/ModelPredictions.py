import os
import datetime
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pydruglogics.model.BooleanModel import BooleanModel
from pydruglogics.utils.Logger import Logger


class ModelPredictions:
    def __init__(self, boolean_models=None, perturbations=None, model_outputs=None, observed_synergy_scores=None,
                 synergy_method='hsa', model_directory=None, attractor_tool=None, verbosity=2):
        """
        Initializes the ModelPredictions class.
        :param boolean_models: List of BooleanModel instances.
        :param perturbations: List of perturbations to apply.
        :param model_outputs: Model outputs for evaluating the predictions.
        :param observed_synergy_scores: List of observed synergy scores.
        :param synergy_method: Method to check for synergy ('hsa' or 'bliss').
        :param model_directory: Directory from which to load models. (Needed only when there is not Evolution result.)
        :param attractor_tool: Tool to calculate attractors in models. (Needed only when loads models from directory)
        :param verbosity: Verbosity level of logging.
        """
        self._boolean_models = boolean_models or []
        self._perturbations = perturbations or []
        self._model_outputs = model_outputs
        self._observed_synergy_scores = observed_synergy_scores
        self._synergy_method = synergy_method
        self._model_predictions = []
        self._predicted_synergy_scores = []
        self._prediction_matrix = {}
        self._logger = Logger(verbosity)

        if model_directory and not boolean_models:
            self._load_models_from_directory(directory=model_directory, attractor_tool=attractor_tool)
        if not model_directory and not boolean_models:
            raise ValueError('Please provide Boolean Models from file or list.')

    def _simulate_model_responses(self, model, perturbation):
        """
        Initializes a single perturbed Boolean model and simulates its response.
        :param model: The Boolean model to perturb.
        :param perturbation: The perturbation to apply to the model.
        :return: The perturbed model, its response, and the perturbation.
        """
        perturbed_model = model.clone()
        perturbed_model.add_perturbations(perturbation)
        self._logger.log(f"Added new perturbed model: {perturbed_model.model_name}", 2)
        perturbed_model.calculate_attractors(perturbed_model.attractor_tool)
        global_output = perturbed_model.calculate_global_output(self._model_outputs, False)
        self._logger.log(f"Adding predicted response for perturbation {perturbation}: {global_output}", 2)
        return perturbed_model, global_output, perturbation

    def _store_result_in_matrix(self, output_matrix, model_name, perturbation, response):
        perturbation_name = self._get_perturbation_name(perturbation)

        if perturbation_name not in output_matrix:
            output_matrix[perturbation_name] = {}

        output_matrix[perturbation_name][model_name] = response

    def _get_perturbation_name(self, perturbation):
        return "-".join(drug['name'] for drug in perturbation)

    def _calculate_mean_responses(self):
        mean_values = {}
        for perturbation, model_responses in self._prediction_matrix.items():
            values = [response for response in model_responses.values() if isinstance(response, (int, float))]
            mean_values[perturbation] = np.mean(values) if values else 0
        return mean_values

    def _calculate_synergy(self):
        """
        Calculate synergy scores for perturbations that contain two drugs based on
        the chosen synergy method (HSA or Bliss).
        """
        self._logger.log('\nCalculating synergies..', 3)
        mean_responses = self._calculate_mean_responses()
        self._logger.log(f"\nSynergy scores ({self._synergy_method}):", 0)
        for perturbation in self._perturbations.perturbations:
            perturbation_name = self._get_perturbation_name(perturbation)

            if '-' in perturbation_name:
                drug1, drug2 = perturbation_name.split('-')
                mean_drug1 = mean_responses.get(drug1, None)
                mean_drug2 = mean_responses.get(drug2, None)
                mean_combination = mean_responses.get(perturbation_name, None)

                if mean_drug1 is not None and mean_drug2 is not None and mean_combination is not None:
                    if self._synergy_method == 'hsa':
                        self._calculate_hsa_synergy(mean_combination, mean_drug1, mean_drug2, perturbation_name)
                    elif self._synergy_method == 'bliss':
                        self._calculate_bliss_synergy(mean_combination, mean_drug1, mean_drug2, perturbation_name)

    def _calculate_hsa_synergy(self, mean_combination, mean_drug1, mean_drug2, perturbation_name):
        min_single_drug_response = min(mean_drug1, mean_drug2)

        if mean_combination < min_single_drug_response:
            synergy_score = mean_combination - min_single_drug_response
        elif mean_combination > min_single_drug_response:
            synergy_score = mean_combination - min_single_drug_response
        else:
            synergy_score = 0

        self._predicted_synergy_scores.append((perturbation_name, synergy_score))
        self._logger.log(f"{perturbation_name}: {synergy_score}", 0)

    def _calculate_bliss_synergy(self, mean_combination, mean_drug1, mean_drug2, perturbation_name):
        drug1_response = ((mean_drug1 - self._model_outputs.min_output) /
                            (self._model_outputs.max_output - self._model_outputs.min_output))
        drug2_response = ((mean_drug2 - self._model_outputs.min_output) /
                          (self._model_outputs.max_output - self._model_outputs.min_output))
        expected_combination_response = 1.0 * drug1_response * drug2_response
        combination_response = ((mean_combination - self._model_outputs.min_output) /
                                (self._model_outputs.max_output - self._model_outputs.min_output))
        synergy_score = combination_response - expected_combination_response

        self._predicted_synergy_scores.append((perturbation_name, synergy_score))
        self._logger.log(f"{perturbation_name}: {synergy_score}", 2)

    def _load_models_from_directory(self, directory, attractor_tool):
        """Loads all .bnet files from the given directory into Boolean Models with attractors and global outputs."""
        for filename in os.listdir(directory):
            if filename.endswith('.bnet'):
                file_path = os.path.join(directory, filename)
                try:
                    model = BooleanModel(file=file_path, attractor_tool=attractor_tool)
                    model.calculate_attractors(attractor_tool)
                    model.calculate_global_output(self._model_outputs, False)
                    self._boolean_models.append(model)
                    self._logger.log(f"Loaded model from {file_path}", 2)
                except Exception as e:
                    print(f"Failed to load model from {file_path}: {str(e)}")

    def run_simulations(self, parallel=True):
        """
        Runs the model simulations in parallel or serially.
        :param parallel: Whether to run simulations in parallel.
        """
        self._model_predictions = []
        self._prediction_matrix = {}

        if parallel:
            self._logger.log('Running simulations in parallel.', 1)
            results = Parallel(n_jobs=-1)(
                delayed(self._simulate_model_responses)(model, perturbation)
                for model in self._boolean_models
                for perturbation in self._perturbations.perturbations
            )

            for model, global_output, perturbation in results:
                self._store_result_in_matrix(self._prediction_matrix, model.model_name, perturbation, global_output)
                self._model_predictions.append((model.model_name, global_output, perturbation))
        else:
            self._logger.log('Running simulations serially.', 1)
            for model in self._boolean_models:
                for perturbation in self._perturbations.perturbations:
                    model, global_output, perturbation = self._simulate_model_responses(model, perturbation)
                    self._store_result_in_matrix(self._prediction_matrix, model.model_name, perturbation, global_output)
                    self._model_predictions.append((model.model_name, global_output, perturbation))

        self.get_prediction_matrix()
        self._calculate_synergy()

    def get_prediction_matrix(self):
        filtered_matrix = {k: v for k, v in self._prediction_matrix.items() if k.count('-') == 1}
        response_matrix_df = pd.DataFrame.from_dict(filtered_matrix, orient='index').fillna('NA')

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        self._logger.log("Response Matrix:", 2)
        self._logger.log(response_matrix_df, 2)

    def plot_roc_and_pr_curve(self):
        """
        Plot the ROC and PR Curves using the predicted synergy scores and the observed synergy combinations.
        """
        df = pd.DataFrame(self._predicted_synergy_scores, columns=['perturbation', 'synergy_score'])
        df['observed'] = df['perturbation'].apply(lambda x: 1 if x in self._observed_synergy_scores else 0)
        df['synergy_score'] = df['synergy_score'] * -1
        df = df.sort_values(by='synergy_score', ascending=False).reset_index(drop=True)
        self._logger.log("\nPredicted Data with Observed Synergies:", 2)
        self._logger.log(df, 2)

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(df['observed'], df['synergy_score'])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC: {roc_auc:.2f} Calibrated")
        plt.plot([0, 1], [0, 1], color='lightgrey', lw=1.2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f"ROC Curve, Ensemble-wise synergies ({self._synergy_method})")
        plt.legend(loc="lower right")
        plt.grid(lw=0.5, color='lightgrey')
        plt.show()
        self._logger.log(f"ROC AUC: {roc_auc:.2f}", 3)

        # PR Curve
        precision, recall, thresholds = precision_recall_curve(df['observed'], df['synergy_score'])
        pr_auc = auc(recall, precision)

        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f"AUC: {pr_auc:.2f} Calibrated")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"PR Curve, Ensemble-wise synergies ({self._synergy_method})")
        plt.legend(loc="upper right")
        plt.grid(lw=0.5, color='lightgrey')
        plt.plot([0, 1], [sum(df['observed']) / len(df['observed'])] * 2, linestyle='--', color='grey',)
        plt.legend(loc="upper right")
        plt.show()
        self._logger.log(f"PR AUC: {pr_auc:.2f}", 3)

    def save_to_file_predictions(self, base_folder='./predictions'):
        time_now = datetime.datetime.now()
        current_date = time_now.strftime('%Y_%m_%d')
        current_time = time_now.strftime('%H%M')

        subfolder = os.path.join(base_folder, f"predictions_{current_date}_{current_time}")
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        with open(os.path.join(subfolder, "model_scores.tab"), "w") as file:
            filtered_matrix = {k: v for k, v in self._prediction_matrix.items() if k.count('-') == 1}
            response_matrix_df = pd.DataFrame.from_dict(filtered_matrix, orient='index').fillna('NA')

            file.write("# Perturbed scores\n")
            response_matrix_df.to_csv(file, sep='\t', mode='w')

            file.write("\n# Unperturbed scores\n")
            for model in self._boolean_models:
                file.write(f"{model.model_name}\t{model.global_output}\n")

        with open(os.path.join(subfolder, f"synergies_{self._synergy_method}.tab"), "w") as file:
            file.write(f"# Synergies ({self._synergy_method})\n")
            file.write("perturbation_name\tsynergy_score\n")
            for perturbation, score in self._predicted_synergy_scores:
                if perturbation.count('-') == 1:
                    file.write(f"{perturbation}\t{score}\n")

        self._logger.log(f"Predictions saved to {subfolder}", 2)

    @property
    def predicted_synergy_scores(self):
        return self._predicted_synergy_scores
