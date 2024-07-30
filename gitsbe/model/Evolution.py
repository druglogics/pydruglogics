import numpy as np
import pygad
from gitsbe.model.BooleanModelOptimizer import BooleanModelOptimizer


class Evolution(BooleanModelOptimizer):
    def __init__(self, boolean_model=None, training_data=None, model_outputs=None, ga_args=None):
        """
        Initializes the Evolution class with a BooleanModel and genetic algorithm parameters.
        :param boolean_model: The boolean model to be evolved.
        :param ga_args: Dictionary containing all necessary arguments for pygad.
        """
        self._boolean_model = boolean_model
        self._ga_args = ga_args or {}
        self._best_models = []
        self.mutation_number = self._ga_args.get('number_of_mutations')
        self._initial_population = boolean_model.generate_mutated_lists(10, 3)
        self._mutation_type = 'topology'
        self._training_data = training_data
        self._model_outputs = model_outputs

        if self._training_data is None or self._model_outputs is None:
            raise ValueError('Please provide training data and model outputs.')

    # def select_mutation(self, mutation_type):
    #     if mutation_type == 'balanced':
    #         return self.balanced_mutation
    #     elif mutation_type == 'topology':
    #         return self.topology_mutation
    #     elif mutation_type == 'mixed':
    #         return self.mixed_mutation
    #     else:
    #         raise ValueError(f"Unknown mutation type: {mutation_type}. Possible values are 'balanced', 'topology', "
    #                          f"and 'mixed'")
    #

    def calculate_fitness(self, ga_instance, solutions, solutions_idx):
        """
        Calculate fitness of models by going through all the observations defined in the
        training data and computing individual fitness values for each one of them.
        """
        fitness_values = []

        for solution in solutions:
            self._boolean_model.from_binary(solution, self._mutation_type)
            self._boolean_model.calculate_attractors(self._boolean_model.attractor_tool)
            responses = self._training_data.responses

            if "globaloutput" in responses[0]:
                observed_global_output = float(responses[0].split(":")[1])
                predicted_global_output = self.calculate_global_output()
                condition_fitness = 1 - abs(predicted_global_output - observed_global_output)
                fitness_values.append(condition_fitness)

                print('\nCalculating fitness..')
                print(f"Scaled fitness [0..1] for solution {solutions_idx}:  {condition_fitness}")

            else:
                attractors = self._boolean_model.attractors
                if not attractors:
                    fitness_values.append(0)
                    continue

                total_matches = []

                for attractor in attractors:
                    matches = self._calculate_matches(attractor)
                    total_matches.append(matches)

                if len(total_matches) > 1:
                    average_matches = np.mean(total_matches)
                    fitness = average_matches / len(self._training_data.responses)
                else:
                    fitness = total_matches[0] / len(self._training_data.responses)
                fitness_values.append(fitness)

                print('\nCalculating fitness..')
                print(f"Scaled fitness [0..1] for solution {solutions_idx}:  {fitness}")

        return fitness_values

    def _calculate_matches(self, attractor):
        total_matches = 0
        responses = self._training_data.responses
        weights = self._training_data.weights

        match_score = 0
        for node_response in responses:
            node, expected_value = node_response.split(":")
            expected_value = float(expected_value)
            if node not in attractor:
                continue

            actual_value = attractor[node]
            if actual_value in [0, 1]:
                match_score += 1 - abs(expected_value - actual_value)
            else:
                match_score += 1 - abs(expected_value - 0.5)

        weighted_match_score = match_score * weights[0]
        total_matches += weighted_match_score

        return total_matches

    def calculate_global_output(self) -> float:
        """
        Use this function after you have calculated attractors with the calculate_attractors function
        in order to find the normalized globaloutput of the model, based on the weights of the nodes
        defined in the ModelOutputs class.
        :return: float
        """
        if not self._boolean_model.attractors:
            raise ValueError("No attractors found. Ensure calculate_attractors() has been called.")

        pred_global_output = 0.0

        for attractor in self._boolean_model.attractors:
            for node_name, node_weight in self._model_outputs.model_outputs.items():
                if node_name not in attractor:
                    continue
                node_state = attractor[node_name]
                state_value = int(node_state) if node_state in [0, 1] else 0.5
                pred_global_output += state_value * node_weight

        pred_global_output /= len(self._boolean_model.attractors)
        return (pred_global_output - self._model_outputs.min_output) / (
                    self._model_outputs.max_output - self._model_outputs.min_output)

    def balanced_mutation(self, offspring, ga_instance):
        for individual in offspring:
            self._boolean_model.update_boolean_model_balance(individual)
            self._boolean_model.balance_mutation(self._mutation_number)
            individual[:] = self._boolean_model.to_binary('balanced')
        return offspring

    def topology_mutation(self, offspring, ga_instance):
        for individual in offspring:
            self._boolean_model.update_boolean_model_topology(individual)
            self._boolean_model.topology_mutations(self._mutation_number)
            individual[:] = self._boolean_model.to_binary('topology')
        return offspring

    def mixed_mutation(self, offspring, ga_instance):
        for individual in offspring:
            self._boolean_model.update_boolean_model_both(individual)
            self._boolean_model.balance_mutation(self._mutation_number)
            self._boolean_model.topology_mutations(self._mutation_number)
            individual[:] = self._boolean_model.to_binary('mixed')
        return offspring

    def run(self):
        initial_mutated_population = self._initial_population
        ga_instance = pygad.GA(
            num_generations=self._ga_args.get('num_generations'),
            num_parents_mating=self._ga_args.get('num_parents_mating'),
            fitness_func=self.calculate_fitness,
            fitness_batch_size=self._ga_args.get('fitness_batch_size', len(initial_mutated_population)),
            sol_per_pop=self._ga_args.get('sol_per_pop'),
            gene_space=[0, 1],
            num_genes=len(initial_mutated_population),
            initial_population=initial_mutated_population,
            parent_selection_type=self._ga_args.get('parent_selection_type', "sss"),
            crossover_type=self._ga_args.get('crossover_type', "single_point"),
            mutation_type='random',
            mutation_percent_genes=self._ga_args.get('mutation_percent_genes'),
            parallel_processing=self._ga_args.get('parallel_processing')
        )

        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(f"Best solution: {solution}")
        print(f"Best solution fitness: {solution_fitness}")
