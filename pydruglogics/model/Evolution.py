import os
import concurrent.futures
import multiprocessing
import datetime
import time
from typing import Optional, Dict, List
import pygad
from pydruglogics import BooleanModel
from pydruglogics.input.TrainingData import TrainingData
from pydruglogics.model.BooleanModelOptimizer import BooleanModelOptimizer


class Evolution(BooleanModelOptimizer):
    def __init__(self,
                 boolean_model=None,
                 training_data=None,
                 model_outputs=None,
                 ga_args: Optional[Dict] = None,
                 num_best_solutions: int = 3,
                 num_runs: int = 10,
                 num_cores: Optional[int] = None):
        """
        Initializes the Evolution class with a BooleanModel and genetic algorithm parameters.

        :param boolean_model: The boolean model to be evolved.
        :param training_data: Training data for the model.
        :param model_outputs: Model outputs for evaluation.
        :param ga_args: Dictionary containing all necessary arguments for pygad.
        :param num_best_solutions: Number of the best solutions to track.
        :param num_runs: Number of times to run the genetic algorithm.
        :param num_cores: Number of cores to use for parallel execution.
        """
        self._boolean_model = boolean_model
        self._mutation_type = boolean_model.mutation_type
        self._training_data = training_data or self._create_default_training_data()
        self._model_outputs = model_outputs
        self._ga_args = ga_args or {}
        self._best_boolean_models = []
        self._num_best_solutions = num_best_solutions
        self._num_runs = num_runs
        self._num_cores = num_cores if num_cores else multiprocessing.cpu_count()

        if not self._model_outputs:
            raise ValueError('Please provide the model outputs.')

    def _callback_generation(self, ga_instance):
        print(f"Generation {ga_instance.generations_completed}: Fitness values: {ga_instance.last_generation_fitness}")

    def _create_default_training_data(self):
        return TrainingData(observations=[(["-"], ["globaloutput:1"], 1.0)])

    def _calculate_matches(self, attractor, response):
        match_score = 0
        found_observations = 0

        for node in response:
            node_name, observed_node_state = node.split(":")
            observed_node_state = float(observed_node_state.strip())
            attractor_state = attractor.get(node_name, '*')
            predicted_node_state = 0.5 if attractor_state == '*' else float(attractor_state)
            match = 1.0 - abs(predicted_node_state - observed_node_state)
            match_score += match
            found_observations += 1
        if found_observations > 0:
            match_score /= found_observations

        return match_score

    def _run_single_ga(self, evolution_number):
        """
        Runs a single GA  and returns the best models for this run.
        """
        ga_instance = pygad.GA(
            num_generations=self._ga_args.get('num_generations'),
            num_parents_mating=self._ga_args.get('num_parents_mating'),
            fitness_func=self.calculate_fitness,
            sol_per_pop=self._ga_args.get('sol_per_pop'),
            gene_space=[0, 1],
            num_genes=len(self._boolean_model.binary_boolean_equations),
            parallel_processing=self._ga_args.get('parallel_processing'),
            random_seed=evolution_number,
            on_generation=self._callback_generation
        )

        ga_instance.run()

        sorted_population = sorted(
            [(ga_instance.population[idx], ga_instance.last_generation_fitness[idx]) for idx in
             range(len(ga_instance.population))], key=lambda x: x[1], reverse=True)[:self._num_best_solutions]

        print(f"Simulation {evolution_number}: Fitness = {sorted_population[0][1]}")
        return sorted_population

    def calculate_fitness(self, ga_instance, solution, solution_idx):
        mutated_boolean_model = self._boolean_model.clone()
        mutated_boolean_model.from_binary(solution, self._mutation_type)
        fitness = 0.0

        for observation in self._training_data.observations:
            response = observation['response']
            weight = observation['weight']
            mutated_boolean_model.calculate_attractors(mutated_boolean_model.attractor_tool)
            condition_fitness = 0.0
            if mutated_boolean_model.has_attractors():
                if 'globaloutput' in response[0]:
                    observed_global_output = float(response[0].split(":")[1])
                    predicted_global_output = mutated_boolean_model.calculate_global_output(self._model_outputs)
                    condition_fitness = 1.0 - abs(predicted_global_output - observed_global_output)

                else:
                    total_matches = [self._calculate_matches(attractor, response) for attractor in
                                     mutated_boolean_model.attractors]
                    if total_matches:
                        avg_matches = sum(total_matches) / len(total_matches)
                        condition_fitness += avg_matches

            fitness += condition_fitness * (weight / self._training_data.weight_sum)

            print('\nCalculating fitness..')
            print(f"Scaled fitness [0..1] for solution {solution_idx}:  {fitness}")

        return fitness

    def run(self) -> List[BooleanModel]:
        """
        Runs the genetic algorithm for the specified number of runs, accumulating the best
        models from each run and returning all of them.
        """
        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=self._num_cores) as executor:
            futures = [executor.submit(self._run_single_ga, i) for i in range(self._num_runs)]
            evolution_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        self._best_boolean_models = []

        for evolution_index, models in enumerate(evolution_results, start=1):
            for solution_index, (solution, fitness) in enumerate(models, start=1):
                best_boolean_model = self._boolean_model.clone()
                best_boolean_model.updated_boolean_equations = best_boolean_model.from_binary(solution,
                                                                                              self._mutation_type)
                best_boolean_model.binary_boolean_equations = solution
                best_boolean_model.fitness = fitness
                best_boolean_model.model_name = f"e{evolution_index}_s{solution_index}"
                self._best_boolean_models.append(best_boolean_model)

        print(f"Total runtime: {time.time() - start_time:.3f} seconds")
        return self._best_boolean_models

    def save_to_file_models(self, base_folder):
        now = datetime.datetime.now()
        current_date = now.strftime('%Y_%m_%d')
        current_time = now.strftime('%H%M')

        subfolder = os.path.join(base_folder, f"models_{current_date}_{current_time}")
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        for model in self._best_boolean_models:
            evolution_number = model.model_name.split("_")[0][1:]
            solution_index = model.model_name.split("_")[1][1:]
            filename = f"e{evolution_number}_s{solution_index}.bnet"
            filepath = os.path.join(subfolder, filename)

            boolean_model_bnet = f"# {current_date}, {current_time}\n"
            boolean_model_bnet += f"# Evolution: {evolution_number} Solution: {solution_index}\n"
            boolean_model_bnet += f"# Fitness Score: {model.fitness:.3f}\n"

            boolean_equation = model.to_bnet_format(model.updated_boolean_equations)
            boolean_model_bnet += boolean_equation

            with open(filepath, "w") as file:
                file.write(boolean_model_bnet)
            print(f"Model saved to {filepath}")

    @property
    def best_boolean_models(self) -> List[BooleanModel]:
        return self._best_boolean_models
