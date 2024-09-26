import concurrent.futures
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from pydruglogics.model.BooleanModel import BooleanModel
from pydruglogics.utils.Logger import Logger


class ModelPredictions:
    def __init__(self, boolean_models=None, perturbations=None, model_outputs=None, model_directory=None,
                 attractor_tool= None, observed_synergy_scores=None, synergy_method='hsa', verbosity = 2):
        """
        Initializes the ModelPredictions class.

        :param boolean_models: List of BooleanModel instances.
        :param perturbations: List of perturbations to apply.
        :param model_outputs: Model outputs for evaluating the predictions.
        :param observed_synergy_scores: Lost of observed synergy scores.
        :param synergy_method: The method used to check for synergy ('hsa' or 'bliss').
        """
        self._boolean_models = boolean_models or []
        self._perturbations = perturbations or []
        self._model_outputs = model_outputs
        self._synergy_method = synergy_method
        self._model_predictions = []
        self._predicted_synergy_scores = []
        self._observed_synergy_scores = observed_synergy_scores
        self._response_matrix = {}
        self._logger = Logger(verbosity)

        if model_directory and not boolean_models:
            self._load_models_from_directory(model_directory, attractor_tool=attractor_tool)
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
        perturbation_parts = [drug['name'] for drug in perturbation]
        return "-".join(perturbation_parts)

    def _print_response_matrix(self, response_matrix):
        response_matrix_df = pd.DataFrame.from_dict(response_matrix, orient='index').fillna('NA')
        response_matrix_df = response_matrix_df.map(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        self._logger.log("Response Matrix:", 2)
        self._logger.log(response_matrix_df, 2)

    def _calculate_mean_responses(self):
        mean_values = {}
        for perturbation, model_responses in self._response_matrix.items():
            values = [response for response in model_responses.values() if isinstance(response, (int, float))]
            mean_values[perturbation] = sum(values) / len(values) if values else 0
        return mean_values

    def _calculate_synergy(self, perturbations):
        """
        Calculate synergy scores for perturbations that contain two drugs based on
        the chosen synergy method (HSA or Bliss).
        """
        self._logger.log('Calculating synergies..', 1)
        mean_responses = self._calculate_mean_responses()

        for perturbation in perturbations:
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

        self._logger.log(f"HSA Synergy score for {perturbation_name}: {synergy_score}", 2)

    def _calculate_bliss_synergy(self, mean_combination, mean_drug1, mean_drug2, perturbation_name):
        bliss_expected_response = mean_drug1 * mean_drug2

        if mean_combination < bliss_expected_response:
            synergy_score = mean_combination - bliss_expected_response
        elif mean_combination > bliss_expected_response:
            synergy_score = mean_combination - bliss_expected_response
        else:
            synergy_score = 0

        self._logger.log(f"Bliss Synergy score for {perturbation_name}: {synergy_score}", 2)

    def _load_models_from_directory(self, directory, attractor_tool):
        """Loads all .bnet files from the specified directory into boolean models."""
        for filename in os.listdir(directory):
            if filename.endswith(".bnet"):
                file_path = os.path.join(directory, filename)
                try:
                    model = BooleanModel(file=file_path, attractor_tool=attractor_tool)
                    self._boolean_models.append(model)
                    self._logger.log(f"Loaded model from {file_path}", 2)
                except Exception as e:
                    print(f"Failed to load model from {file_path}: {str(e)}")

    def run_simulations(self, parallel=True):
        self._model_predictions = []
        self._response_matrix = {}

        if parallel:
            self._logger.log('Running simulations in paralell.', 1)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._simulate_model_responses, model, perturbation)
                    for model in self._boolean_models
                    for perturbation in self._perturbations.perturbations
                ]
                for future in concurrent.futures.as_completed(futures):
                    model, global_output, perturbation = future.result()
                    self._store_result_in_matrix(self._response_matrix, model.model_name, perturbation, global_output)
                    self._model_predictions.append((model.model_name, global_output, perturbation))
        else:
            self._logger.log('Running simulations serially.', 1)
            for model in self._boolean_models:
                for single_perturbation in self._perturbations.perturbations:
                    model, global_output, perturbation = self._simulate_model_responses(model, single_perturbation)
                    self._store_result_in_matrix(self._response_matrix, model.model_name, perturbation, global_output)
                    self._model_predictions.append((model.model_name, global_output, global_output))

        self._print_response_matrix(self._response_matrix)
        self._calculate_synergy(self._perturbations.perturbations)

    def plot_roc_and_pr_curve(self):
        """
        Plot the ROC and Precision-Recall curves using the predicted synergy scores
        and the observed synergy combinations.
        """
        predicted_data = []
        for perturbation, model_responses in self._response_matrix.items():
            if '-' in perturbation:
                drugs = perturbation.split('-')
                if len(drugs) == 2:
                    mean_drug1 = self._calculate_mean_responses().get(drugs[0])
                    mean_drug2 = self._calculate_mean_responses().get(drugs[1])
                    mean_combination = self._calculate_mean_responses().get(perturbation)

                    if mean_drug1 is not None and mean_drug2 is not None and mean_combination is not None:
                        if self._synergy_method == 'hsa':
                            min_single_agent_response = min(mean_drug1, mean_drug2)
                            synergy_score = mean_combination - min_single_agent_response
                        else:
                            bliss_expected_response = mean_drug1 * mean_drug2
                            synergy_score = mean_combination - bliss_expected_response

                        predicted_data.append((perturbation, synergy_score))



        predicted_df = pd.DataFrame(predicted_data, columns=['Combination', 'Score'])
        predicted_df['Combination'] = predicted_df['Combination'].apply(
            lambda x: x.replace('[', '').replace(']', '').replace('-', '~'))

        predicted_df['Observed'] = predicted_df['Combination'].apply(
            lambda x: 1 if x in self._observed_synergy_scores else 0)
        print(predicted_df.head())

        y_true = predicted_df['Observed']
        y_scores = predicted_df['Score']
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='red', label=f"ROC curve (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        plt.figure()
        plt.plot(recall, precision, color='green', label=f"PR curve (AUC={pr_auc:.2f})")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"Precision-Recall Curve, Method: {self._synergy_method}")
        plt.legend(loc='lower left')
        plt.show()
