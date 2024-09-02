import concurrent.futures
import pandas as pd
import itertools

class ModelPredictions:
    def __init__(self, boolean_models=None, perturbations=None, model_outputs=None, synergy_method='hsa'):
        """
        Initializes the ModelPredictions class.

        :param boolean_models: List of BooleanModel instances.
        :param perturbations: List of perturbations to apply.
        :param model_outputs: Model outputs for evaluating the predictions.
        :param synergy_method: The method used to check for synergy ('hsa' or 'bliss').
        """
        self._boolean_models = boolean_models or []
        self._perturbations = perturbations or []
        self._model_outputs = model_outputs
        self._synergy_method = synergy_method
        self._model_predictions = []
        self._perturbation_models = []

    def run_simulations(self, parallel=True):
        """
        Initializes perturbed Boolean models, simulates their responses, and stores the results.
        """
        self._model_predictions = []
        result_matrix = {}

        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._simulate_model_responses, model, perturbation, model_index,
                                    perturbation_index)
                    for model_index, model in enumerate(self._boolean_models)
                    for perturbation_index, perturbation in enumerate(self._perturbations.perturbations)
                    ]
                for future in concurrent.futures.as_completed(futures):
                    model, global_output, perturbation = future.result()
                    self._store_result_in_matrix(result_matrix, model.model_name, perturbation, global_output)
                    self._model_predictions.append((model.model_name, global_output, perturbation))
        else:
            for model_index, model in enumerate(self._boolean_models):
                for perturbation_index, perturbation in enumerate(self._perturbations.perturbations):
                    model, response, perturbation = self._simulate_model_responses(model, perturbation,
                                                                                   model_index, perturbation_index)
                    self._store_result_in_matrix(result_matrix, model, perturbation, response)
                    self._model_predictions.append((model.model_name, response, perturbation))

        self._print_result_matrix(result_matrix)

    def _simulate_model_responses(self, model, perturbation, model_index, perturbation_index):
        """
        Initializes a single perturbed Boolean model and simulates its response.

        :param model: The Boolean model to perturb.
        :param perturbation: The perturbation to apply to the model.
        :param model_index: Index of the model in the boolean_model_list.
        :param perturbation_index: Index of the perturbation in the perturbations list.
        :return: The perturbed model, its response, and the perturbation.
        """
        perturbed_model = model.clone()
        perturbed_model.apply_perturbations(perturbation)
        perturbed_model.model_name = f"perturbed_model_{model_index + 1}_{perturbation_index + 1}"
        self._perturbation_models.append((perturbed_model, perturbation))

        perturbed_model.calculate_attractors(perturbed_model.attractor_tool)
        global_output = perturbed_model.calculate_global_output(self._model_outputs)

        if len(perturbation) > 1:
            self._check_combination_model_for_synergy(perturbation)

        return perturbed_model, global_output, perturbation

    def _check_combination_model_for_synergy(self, perturbation):
        drug_combination = self.get_combination_name(perturbation)
        drug_comb_perturbation_model = self.get_model_for_perturbation(perturbation)

        if not drug_comb_perturbation_model:
            print(f"No model found for the combination: {drug_combination}")
            return

        drug_combination_subsets = self.get_combination_subsets(perturbation)
        is_model_has_global_output = drug_comb_perturbation_model.has_global_output()

        for drug in drug_combination_subsets:
            if not self.get_model_for_perturbation(drug).has_global_output():
                is_model_has_global_output = False

        if is_model_has_global_output:
            if self._synergy_method == "hsa":
                min_global_output = min(self.get_model_for_perturbation(subset).global_output for subset in
                                        drug_combination_subsets)
                if drug_comb_perturbation_model.global_output < min_global_output:
                    print(f"{drug_combination} is synergistic (HSA)")
                else:
                    print(f"{drug_combination} is NOT synergistic (HSA)")
            elif self._synergy_method == "bliss":
                expected_bliss_global_output = 1
                for drug in drug_combination_subsets:
                    expected_bliss_global_output *= self.get_model_for_perturbation(drug).global_output
                if drug_comb_perturbation_model.global_output < expected_bliss_global_output:
                    print(f"{drug_combination} is synergistic (Bliss)")
                else:
                    print(f"{drug_combination} is NOT synergistic (Bliss)")
        else:
            print(f"{drug_combination} cannot be evaluated for synergy (lacking attractors)")

    def get_combination_subsets(self, perturbation):
        return [list(combo) for i in range(1, len(perturbation)) for combo in itertools.combinations(perturbation, i)]

    def get_combination_name(self, perturbation):
        return "-".join(sorted([drug['name'] for drug in perturbation]))

    def get_model_for_perturbation(self, perturbation):
        """
        Retrieve the model associated with a specific perturbation.

        :param perturbation: List of drugs in the perturbation.
        :return: The BooleanModel associated with the perturbation.
        """
        return next((model for model, pert in self._perturbation_models if
                     self._perturbations_equal(pert, perturbation)), None)

    def _perturbations_equal(self, pert1, pert2):
        return sorted(pert1, key=lambda x: sorted(x.items())) == sorted(pert2, key=lambda x: sorted(x.items()))

    def get_combination_subsets(self, combination):
        """
        Given a list of drug combinations, generate all subsets of the combination that are one drug short.
        """
        drugs = []
        drugs_in_combination = len(combination)
        drugs_in_subset = drugs_in_combination - 1

        if drugs_in_subset < 1:
            return drugs

        for drug in itertools.combinations(combination, drugs_in_subset):
            drugs.append(list(drug))
        return drugs

    def _store_result_in_matrix(self, output_matrix, model_name, perturbation, response):
        perturbation_name = self.get_combination_name(perturbation)
        if model_name not in output_matrix:
            output_matrix[model_name] = {}
        output_matrix[model_name][perturbation_name] = response

    def _print_result_matrix(self, output_matrix):
        df = pd.DataFrame.from_dict(output_matrix, orient='index').fillna('NA')
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        print("Result Matrix:")
        print(df)
