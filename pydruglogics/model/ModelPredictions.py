import os
import datetime
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pydruglogics.model.BooleanModel import BooleanModel
import logging


class ModelPredictions:
    def __init__(self, boolean_models=None, perturbations=None, model_outputs=None,
                 synergy_method='bliss', model_directory=None, attractor_tool=None, attractor_type=None):
        """
        Initializes the ModelPredictions class.
        :param boolean_models: List of BooleanModel instances.
        :param perturbations: List of perturbations to apply.
        :param model_outputs: Model outputs for evaluating the predictions.
        :param observed_synergy_scores: List of observed synergy scores.
        :param synergy_method: Method to check for synergy. Possible values: 'hsa' or 'bliss'.
        :param model_directory: Directory from which to load models. (Needed only when there is no Evolution result.)
        :param attractor_tool: Tool to calculate attractors in models. (Needed only when loads models from directory.)
        :param attractor_type: Type to calculate attractors in models. (Needed only when loads models from directory.)
        """
        self._boolean_models = boolean_models or []
        self._perturbations = perturbations or []
        self._model_outputs = model_outputs
        self._synergy_method = synergy_method
        self._model_predictions = []
        self._predicted_synergy_scores = []
        self._prediction_matrix = {}

        if model_directory and not boolean_models:
            self._load_models_from_directory(directory=model_directory, attractor_tool=attractor_tool,
                                             attractor_type=attractor_type)
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
        logging.debug(f"Added new perturbed model: {perturbed_model.model_name}")
        perturbed_model.calculate_attractors(perturbed_model.attractor_tool, perturbed_model.attractor_type)
        global_output = perturbed_model.calculate_global_output(self._model_outputs, False)
        logging.debug(f"Adding predicted response for perturbation {perturbation}: {global_output}")
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
        logging.debug('\nCalculating synergies..')
        mean_responses = self._calculate_mean_responses()
        logging.info(f"\nSynergy scores ({self._synergy_method}):")
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
        logging.info(f"{perturbation_name}: {synergy_score}")

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
        logging.info(f"{perturbation_name}: {synergy_score}")

    def _load_models_from_directory(self, directory, attractor_tool, attractor_type):
        """
        Loads all .bnet files from the given directory into Boolean Models with attractors and global outputs.
        :param directory: The path to the directory containing '.bnet' files.
        :param attractor_tool: The tool used for attractor calculation. Possible values : 'mpbn', 'pyboolnet'.
        :param attractor_type: The type of attractor to be calculated. Possible values: 'stable_states', 'trapspaces'.
        :return: None
        """
        try:
            for filename in os.listdir(directory):
                if filename.endswith('.bnet'):
                    file_path = os.path.join(directory, filename)
                    try:
                        model = BooleanModel(file=file_path, attractor_tool=attractor_tool,
                                             attractor_type=attractor_type)
                        model.calculate_attractors(attractor_tool, attractor_type)
                        model.calculate_global_output(self._model_outputs, False)
                        self._boolean_models.append(model)
                        logging.debug(f"Model loaded from {file_path}")
                    except Exception as e:
                        print(f"Failed to load model from {file_path}: {str(e)}")
                        raise
        except Exception as e:
            logging.error(f"Error loading models from directory {directory}: {str(e)}")
            raise

    def run_simulations(self, parallel=True, cores=4):
        """
        Runs simulations on the Boolean Models with the perturbations, either in parallel or serially.
        :param parallel: Whether to run the simulations in parallel By default, True.
        :param cores: The number of CPU cores to use for parallel execution By default, 4.
        :return: None
        """
        self._model_predictions = []
        self._prediction_matrix = {}

        if parallel:
            logging.debug('Running simulations in parallel.')
            results = Parallel(n_jobs=cores, backend='loky')(
                delayed(self._simulate_model_responses)(model, perturbation)
                for model in self._boolean_models
                for perturbation in self._perturbations.perturbations
            )

            for model, global_output, perturbation in results:
                self._store_result_in_matrix(self._prediction_matrix, model.model_name, perturbation, global_output)
                self._model_predictions.append((model.model_name, global_output, perturbation))
        else:
            logging.debug('Running simulations serially.')
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
        logging.debug("\nResponse Matrix:")
        logging.debug(response_matrix_df)

    def save_to_file_predictions(self, base_folder='./results/predictions'):
        try:
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

            logging.info(f"Predictions saved to {subfolder}")

        except IOError as e:
            logging.error(f"File I/O error while saving predictions: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error saving predictions to file: {str(e)}")
            raise

    @property
    def predicted_synergy_scores(self):
        return self._predicted_synergy_scores
