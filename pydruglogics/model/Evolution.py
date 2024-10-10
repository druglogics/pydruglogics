import os
import multiprocessing
import datetime
from typing import List
import pygad
import numpy as np
from numpy.random import default_rng
from joblib import Parallel, delayed
from pydruglogics import BooleanModel
from pydruglogics.input.TrainingData import TrainingData
from pydruglogics.model.BooleanModelOptimizer import BooleanModelOptimizer
from pydruglogics.utils.Logger import Logger
from pydruglogics.utils.Util import Util

class Evolution(BooleanModelOptimizer):
    def __init__(self, boolean_model=None, training_data=None, model_outputs=None,
                 ga_args=None, ev_args=None, verbosity=2):
        """
        Initializes the Evolution class with a BooleanModel and genetic algorithm parameters.
        :param boolean_model: The boolean model to be evolved.
        :param training_data: Training data for the model.
        :param model_outputs: Model outputs for evaluation.
        :param ga_args: Dictionary containing all necessary arguments for pygad.
        :param ev_args: Dictionary containing all necessary arguments for runnning the evolution.
        :param verbosity: Verbosity level of logging.
        """
        self._boolean_model = boolean_model
        self._mutation_type = boolean_model.mutation_type
        self._training_data = training_data or self._create_default_training_data()
        self._model_outputs = model_outputs
        self._ga_args = ga_args or {}
        self._ev_args = ev_args or {}
        self._best_boolean_models = []
        self._logger = Logger(verbosity)

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
            num_generations=self._ga_args.get('num_generations', 20),
            num_parents_mating=self._ga_args.get('num_parents_mating', 3),
            fitness_func=self.calculate_fitness,
            mutation_num_genes=self._ga_args.get('mutation_num_genes', 3),
            gene_space=[0, 1],
            gene_type= int,
            initial_population=initial_population,
            random_seed=self._ga_args.get('num_generations'),
            on_generation=self._callback_generation,
            fitness_batch_size=self._ga_args.get('fitness_batch_size', 20),
            crossover_type=self._ga_args.get('crossover_type', 'single_point'),
            mutation_type=self._ga_args.get('mutation_type', 'random'),
            parent_selection_type=self._ga_args.get('parent_selection_type', 'sss')
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
        fitness_values = Parallel(n_jobs=-1)(
            delayed(self._calculate_fitness_for_solution)(solution) for solution in solutions
        )
        return fitness_values

    def _calculate_fitness_for_solution(self, solution):
        fitness = 0.0
        mutated_boolean_model = self._boolean_model.clone()
        mutated_boolean_model.from_binary(solution, self._mutation_type)

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
                        self._logger.log(f"From {found_observations} observations, found {match_score} matches", 2)
                        if found_observations > 0:
                            if mutated_boolean_model.has_stable_states():
                                condition_fitness /= (found_observations + 1)
                            else:
                                condition_fitness /= found_observations
                            match_score /= found_observations
                        total_matches.append(match_score)
                    if total_matches:
                        avg_matches = sum(total_matches) / len(total_matches)
                        self._logger.log(f"Average match value through all stable states: {avg_matches}", 2)
                        condition_fitness += avg_matches
            fitness += condition_fitness * (weight / self._training_data.weight_sum)

        return fitness

    def create_initial_population(self, population_size, num_mutations, seed=None):
        """
        Creates an initial population for the GA.
        :param population_size: The number of individuals in the population.
        :param num_mutations: The number of mutations to perform on each individual.
        :param seed: Seed for reproducibility.
        :return: List of binary vectors representing the initial population.
        """
        rng = default_rng(seed)
        initial_vector = self._boolean_model.binary_boolean_equations
        population = []

        for _ in range(population_size):
            individual = initial_vector.copy()
            individual_mutation = rng.choice(len(individual), num_mutations, replace=False)
            for index in individual_mutation:
                individual[index] = 1-individual[index]

            population.append(individual)

        return population

    def run(self):
        seeds = self._ev_args.get('num_of_seeds', 20)
        cores = self._ev_args.get('num_of_cores') if self._ev_args.get('num_of_cores') else multiprocessing.cpu_count()
        num_of_runs = self._ev_args.get('num_of_runs', 20)

        self._logger.log("Starting the evolutionary process...", 0)
        initial_populations = []

        if seeds is not None:
            np.random.seed(seeds)

        for i in range(num_of_runs):
            seed = seeds + i if seeds is not None else None
            initial_population = self.create_initial_population(
                population_size=self._ga_args.get('fitness_batch_size'),
                num_mutations=self._ev_args.get('num_of_init_mutation', 10),
                seed=seed
            )
            initial_populations.append((i, initial_population))

        evolution_results = Parallel(n_jobs=cores)(
            delayed(self._run_single_ga)(i, initial_population) for i, initial_population in initial_populations
        )

        self._best_boolean_models = []

        for evolution_index, models in enumerate(evolution_results, start=1):
            for solution_index, (solution, fitness) in enumerate(models, start=1):
                best_boolean_model = self._boolean_model.clone()
                best_boolean_model.updated_boolean_equations = best_boolean_model.from_binary(solution, self._mutation_type)
                best_boolean_model.binary_boolean_equations = solution
                best_boolean_model.fitness = fitness
                best_boolean_model.model_name = f"e{evolution_index}_s{solution_index}"
                self._best_boolean_models.append(best_boolean_model)

        return self._best_boolean_models


    def save_to_file_models(self, base_folder= './models'):
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

            boolean_equation = Util.to_bnet_format(model.updated_boolean_equations)
            boolean_model_bnet += boolean_equation

            with open(filepath, "w") as file:
                file.write(boolean_model_bnet)

            self._logger.log(f"Model saved to {filepath}", 2)

    @property
    def best_boolean_models(self) -> List[BooleanModel]:
        return self._best_boolean_models
