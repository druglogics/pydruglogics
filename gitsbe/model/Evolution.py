import numpy as np
import pygad
from gitsbe.model.BooleanModelOptimizer import BooleanModelOptimizer
import concurrent.futures
import multiprocessing


class Evolution(BooleanModelOptimizer):
    def __init__(self, boolean_model=None, training_data=None, model_outputs=None, ga_args=None,
                 num_best_solutions=3, num_mutations=3, num_mutations_per_eq=3, num_runs=10, num_cores=None):
        """
        Initializes the Evolution class with a BooleanModel and genetic algorithm parameters.
        :param boolean_model: The boolean model to be evolved.
        :param ga_args: Dictionary containing all necessary arguments for pygad.
        :param num_best_solutions: Number of the best solutions to track.
        :param num_mutations: Number of the initial mutated equations.
        :param num_mutations_per_eq: Number of the initial mutations per equation.
        :param num_runs: Number of times to run the genetic algorithm.
        :param num_cores: Number of cores to use for parallel execution.
        """
        self._boolean_model = boolean_model
        self._ga_args = ga_args or {}
        self._best_models = []
        self.mutation_number = self._ga_args.get('number_of_mutations')
        self._initial_population = boolean_model.generate_mutated_lists(num_mutations, num_mutations_per_eq)
        self._mutation_type = boolean_model.mutation_type
        self._training_data = training_data
        self._model_outputs = model_outputs
        self._num_best_solutions = num_best_solutions
        self._num_runs = num_runs
        self._num_cores = num_cores if num_cores is not None else multiprocessing.cpu_count()

        if self._training_data is None or self._model_outputs is None:
            raise ValueError('Please provide training data and model outputs.')

    def calculate_fitness(self, ga_instance, solutions, solutions_idx):
        """
        Calculate fitness of models by going through all the observations defined in the
        training data and computing individual fitness values for each one of them.
        """
        fitness_values = np.zeros(len(solutions))

        for idx, solution in enumerate(solutions):
            self._boolean_model.from_binary(solution, self._mutation_type)
            self._boolean_model.calculate_attractors(self._boolean_model.attractor_tool)
            responses = self._training_data.responses

            if "globaloutput" in responses[0]:
                observed_global_output = float(responses[0].split(":")[1])
                predicted_global_output = self.calculate_global_output()
                condition_fitness = 1 - abs(predicted_global_output - observed_global_output)
                fitness_values[idx] = condition_fitness
            else:
                attractors = self._boolean_model.attractors
                if not attractors:
                    continue

                total_matches = np.array([self._calculate_matches(attractor) for attractor in attractors])

                if len(total_matches) > 1:
                    average_matches = np.mean(total_matches)
                    fitness = average_matches / len(responses)
                else:
                    fitness = total_matches[0] / len(responses)
                fitness_values[idx] = fitness

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

    def callback_generation(self, ga_instance):
        population = ga_instance.population
        fitness = ga_instance.last_generation_fitness
        sorted_indices = np.argsort(fitness)[::-1]

        unique_solutions = set()
        new_best_models = []

        for idx in sorted_indices:
            solution_tuple = tuple(population[idx])
            if solution_tuple not in unique_solutions:
                unique_solutions.add(solution_tuple)
                new_best_models.append((population[idx], fitness[idx]))
            if len(unique_solutions) == self._num_best_solutions:
                break

        new_best_models.sort(key=lambda x: x[1], reverse=True)
        self._best_models[:] = new_best_models[:self._num_best_solutions]

    def run_single_ga(self, evolution_number):
        ga_instance = pygad.GA(
            num_generations=self._ga_args.get('num_generations'),
            num_parents_mating=self._ga_args.get('num_parents_mating'),
            fitness_func=self.calculate_fitness,
            fitness_batch_size=len(self._initial_population),
            sol_per_pop=self._ga_args.get('sol_per_pop'),
            gene_space=[0, 1],
            num_genes=len(self._initial_population),
            initial_population=self._initial_population,
            parent_selection_type=self._ga_args.get('parent_selection_type', "sss"),
            crossover_type=self._ga_args.get('crossover_type', "single_point"),
            mutation_type='random',
            mutation_percent_genes=self._ga_args.get('mutation_percent_genes'),
            parallel_processing=self._ga_args.get('parallel_processing'),
            on_generation=self.callback_generation
        )

        ga_instance.run()

        best_models = self._best_models[:self._num_best_solutions]
        # for idx, (solution, fitness) in enumerate(best_models):
        #     print(f"Best {idx + 1} result of evolution {evolution_number}: solution: {solution}, fitness: {fitness}")

        return best_models

    def run(self):
        all_best_models = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self._num_cores) as executor:
            futures = [executor.submit(self.run_single_ga, i + 1) for i in range(self._num_runs)]
            for future in concurrent.futures.as_completed(futures):
                all_best_models.extend(future.result())

        all_best_models.sort(key=lambda x: x[1], reverse=True)
        self._best_models = all_best_models[:self._num_best_solutions]

        print('\nBest solutions across all evolutions: ')
        for idx, (solution, fitness) in enumerate(self._best_models):
            print(f"\nBest Solution {idx + 1}: {solution}, Fitness: {fitness}")
