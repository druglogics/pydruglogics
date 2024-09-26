import os
import random
import concurrent.futures
import multiprocessing
import datetime
import time
from typing import List
import pygad
import numpy as np
from pydruglogics import BooleanModel
from pydruglogics.input.TrainingData import TrainingData
from pydruglogics.model.BooleanModelOptimizer import BooleanModelOptimizer
from pydruglogics.utils.Logger import Logger

class Evolution(BooleanModelOptimizer):
    def __init__(self, boolean_model=None, training_data=None, model_outputs=None,
                 ga_args=None, ev_args=None, verbosity=2):
        self._boolean_model = boolean_model
        self._mutation_type = boolean_model.mutation_type
        self._training_data = training_data or self._create_default_training_data()
        self._model_outputs = model_outputs
        self._ga_args = ga_args or {}
        self._ev_args = ev_args or {}
        self._best_boolean_models = []
        self._logger = Logger(verbosity)
        self.total_runtime = 0.0

        if not self._model_outputs:
            raise ValueError('Please provide the model outputs.')

    def _callback_generation(self, ga_instance):
        self._logger.log(f"Generation {ga_instance.generations_completed}: Fitness values: "
                        f"{ga_instance.last_generation_fitness}", 2)

    def _create_default_training_data(self):
        return TrainingData(observations=[(["-"], ["globaloutput:1"], 1.0)])

    def _run_single_ga(self, evolution_number, initial_population):
        """
        Runs a single GA and returns the best models for this run.
        :param evolution_number: The index of the current GA run.
        :param initial_population: The initial population for the GA.
        """
        self._logger.log(f"Running GA simulation {evolution_number}...", 1)

        ga_instance = pygad.GA(
            num_generations=self._ga_args.get('num_generations'),
            num_parents_mating=self._ga_args.get('num_parents_mating'),
            fitness_func=self.calculate_fitness,
            sol_per_pop=self._ga_args.get('sol_per_pop'),
            keep_elitism=self._ga_args.get('keep_elitism'),
            gene_space=[0, 1],
            initial_population=initial_population,
            random_seed=evolution_number,
            on_generation=self._callback_generation,
            fitness_batch_size=self._ga_args.get('fitness_batch_size')
        )

        ga_instance.run()

        sorted_population = sorted(
            [(ga_instance.population[idx], ga_instance.last_generation_fitness[idx])
             for idx in range(len(ga_instance.population))], key=lambda x: x[1],
            reverse=True)[:self._ev_args.get('num_best_solutions')]

        self._logger.log(f"Best fitness in Simulation {evolution_number}: Fitness = {sorted_population[0][1]}", 2)

        return sorted_population

    def calculate_fitness(self, ga_instance, solutions, solution_idx):
        """
        Calculates fitness for a batch of solutions. Each solution is evaluated in a batch, and a fitness score is returned for each.
        :param solutions: A batch of solutions (list of binary vectors).
        :param solution_idx: The index of the current solution batch.
        :return: A list of fitness values, one for each solution in the batch.
        """
        fitness_values = []

        for solution in solutions:
            mutated_boolean_model = self._boolean_model.clone()
            mutated_boolean_model.from_binary(solution, self._mutation_type)
            fitness = 0.0
            self._logger.log(f"Model {mutated_boolean_model.model_name}", 2)

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
                        self._logger.log(f"Observed Global Output: {observed_global_output:.2f}, ", 3)
                        self._logger.log(f"Predicted Global Output: {predicted_global_output:.2f}", 3)
                    else:
                        if mutated_boolean_model.has_stable_states():
                            condition_fitness += 1.0

                        total_matches = []
                        for index, attractor in enumerate(mutated_boolean_model.attractors):
                            self._logger.log(f"Checking stable state no. {index + 1}", 2)
                            match_score = 0
                            found_observations = 0
                            for node in response:
                                node_name, observed_node_state = node.split(":")
                                observed_node_state = float(observed_node_state.strip())
                                attractor_state = attractor.get(node_name, '*')
                                predicted_node_state = 0.5 if attractor_state == '*' else float(attractor_state)
                                match = 1.0 - abs(predicted_node_state - observed_node_state)
                                self._logger.log(f"Match for observation on node {node_name}: {match} (1 - "
                                                 f"|{predicted_node_state} - {observed_node_state}|)", 3)
                                match_score += match
                                found_observations += 1
                            self._logger.log(f"From {found_observations} observations, found {match_score} matches",2)
                            if found_observations > 0:
                                if mutated_boolean_model.has_stable_states():
                                    condition_fitness /= (found_observations+1)
                                else:
                                    condition_fitness /= found_observations
                                match_score /= found_observations
                            total_matches.append(match_score)
                        if total_matches:
                            avg_matches = sum(total_matches) / len(total_matches)
                            self._logger.log(f"Average match value through all stable states: {avg_matches}", 2)
                            condition_fitness += avg_matches

                fitness += condition_fitness * (weight / self._training_data.weight_sum)

            fitness_values.append(fitness)

        return fitness_values

    def create_initial_population(self, population_size, num_mutations, seed=None):
        """
        Creates an initial population for the GA.
        :param population_size: The number of individuals in the population.
        :param num_mutations: The number of mutations to perform on each individual.
        :param seed: Seed for reproducibility.
        :return: List of binary vectors representing the initial population.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        initial_vector = self._boolean_model.to_binary(self._mutation_type)
        population = []

        for _ in range(population_size):
            individual = initial_vector.copy()
            mutation_indices = np.random.choice(len(individual), num_mutations, replace=False)
            for idx in mutation_indices:
                individual[idx] = 1 - individual[idx]

            population.append(individual)

        return population

    def run(self) -> List[BooleanModel]:
        """
        Runs the genetic algorithm for the specified number of runs, accumulating the best
        models from each run and returning all of them.
        """
        start_time = time.time()
        seeds = self._ev_args.get('num_of_seeds')
        cores = self._ev_args.get('num_of_cores') if self._ev_args.get('num_of_cores')\
            else multiprocessing.cpu_count()

        self._logger.log("Starting the evolutionary process...", 0)

        if seeds is not None:
            np.random.seed(seeds)

        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            futures = []
            for i in range(self._ev_args.get('num_of_runs')):
                initial_population = self.create_initial_population(
                    population_size=self._ga_args.get('sol_per_pop'),
                    num_mutations=3,
                    seed=seeds + i if seeds is not None else None
                )
                futures.append(executor.submit(self._run_single_ga, i, initial_population))

            evolution_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        self._best_boolean_models = []

        for evolution_index, models in enumerate(evolution_results, start=1):
            for solution_index, (solution, fitness) in enumerate(models, start=1):
                best_boolean_model = self._boolean_model.clone()
                best_boolean_model.updated_boolean_equations = best_boolean_model.from_binary(solution, self._mutation_type)
                best_boolean_model.binary_boolean_equations = solution
                best_boolean_model.fitness = fitness
                best_boolean_model.model_name = f"e{evolution_index}_s{solution_index}"
                self._best_boolean_models.append(best_boolean_model)

        total_runtime = time.time() - start_time
        self.total_runtime = total_runtime
        self._logger.log(f"Total evolutionary process runtime: {total_runtime:.3f} seconds", 1)

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

            self._logger.log(f"Model saved to {filepath}", 2)

    @property
    def best_boolean_models(self) -> List[BooleanModel]:
        return self._best_boolean_models
