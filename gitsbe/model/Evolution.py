import pygad
from gitsbe import BooleanModel, TrainingData


class Evolution:
    def __init__(self, boolean_model=None, ga_args=None):
        """
        Initializes the Evolution class with a BooleanModel and genetic algorithm parameters.
        :param boolean_model: The boolean model to be evolved.
        :param ga_args: Dictionary containing all necessary arguments for pygad.
        """
        self._boolean_model = boolean_model
        self._ga_args = ga_args or {}
        self._best_models = []
        self._mutation_number = self._ga_args.get('number_of_mutations')
        self._initial_population = self.create_initial_population(mutation_type=self._ga_args.get('mutation_type'),
                                                                  population_size=self._ga_args.get('population_size'))
        self._mutation_type = self.select_mutation(self._ga_args.get('mutation_type'))

    def select_mutation(self, mutation_type):
        if mutation_type == 'balanced':
            return self.balanced_mutation
        elif mutation_type == 'topology':
            return self.topology_mutation
        elif mutation_type == 'mixed':
            return self.mixed_mutation
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}. Possible values are 'balanced', 'topology', "
                             f"and 'mixed'")

    def create_initial_population(self, mutation_type, population_size):
        initial_population = []
        mut_number = self._mutation_number

        for _ in range(population_size):
            if mutation_type == 'balanced':
                self._boolean_model.balance_mutation(mut_number)
                initial_population.append(self._boolean_model.to_binary('balanced'))
            elif mutation_type == 'topology':
                self._boolean_model.topology_mutations(mut_number)
                initial_population.append(self._boolean_model.to_binary('topology'))
            elif mutation_type == 'mixed':
                self._boolean_model.balance_mutation(mut_number)
                self._boolean_model.topology_mutations(mut_number)
                initial_population.append(self._boolean_model.to_binary('mixed'))

        return initial_population

    def calculate_fitness(self, ga_instance, solutions, solutions_idx):
        """
        Calculate fitness of models by going through all the observations defined in the
        training data and computing individual fitness values for each one of them.
        """
        attractor_tool = self._boolean_model.attractor_tool
        training_data = TrainingData.get_instance()
        fitness_values = []

        for solution, solution_idx in zip(solutions, solutions_idx):
            fitness = 0
            if self._ga_args.get('mutation_type') == 'mixed':
                self._boolean_model.update_boolean_model_both(solution)
            elif self._ga_args.get('mutation_type') == 'topology':
                self._boolean_model.update_boolean_model_topology(solution)
            elif self._ga_args.get('mutation_type') == 'balanced':
                self._boolean_model.update_boolean_model_balance(solution)

            self._boolean_model.calculate_attractors(attractor_tool)

            for condition_number, observation in enumerate(training_data.observations):
                condition_fitness = 0
                response = observation['response']
                weight = observation['weight']
                condition = observation['condition']

                mutated_boolean_model = self._boolean_model
                mutated_boolean_model.model_name = f"{self._boolean_model.model_name}_condition_{condition_number}"
                mutated_boolean_model.update_boolean_model_both(solution)
                mutated_boolean_model.calculate_attractors(attractor_tool)

                if response[0].split(":")[0] == "globaloutput":
                    observed_global_output = float(response[0].split(":")[1])
                    predicted_global_output = mutated_boolean_model.calculate_global_output()
                    condition_fitness = 1 - abs(predicted_global_output - observed_global_output)
                else:
                    if mutated_boolean_model.has_stable_states():
                        condition_fitness += 1

                    average_match = 0
                    found_observations = 0
                    matches = []

                    for attractor in mutated_boolean_model.attractors:
                        match_sum = 0
                        for response_str in response:
                            node, observation = response_str.split(":")
                            index_of_node = mutated_boolean_model.get_index_of_equation(node)
                            if index_of_node >= 0:
                                found_observations += 1
                                node_state = attractor.get(node)
                                state_value = 0.5 if node_state == "-" else float(node_state)
                                match = 1 - abs(state_value - float(observation))
                                match_sum += match

                        matches.append(match_sum)

                    if matches:
                        average_match = sum(matches) / len(matches)
                    condition_fitness += average_match

                    if found_observations > 0:
                        if mutated_boolean_model.has_stable_states():
                            condition_fitness /= (found_observations + 1)
                        else:
                            condition_fitness /= found_observations

                fitness += condition_fitness * weight / training_data.get_weight_sum

            print('\nCalculating fitness..')
            print(f"Scaled fitness [0..1] for solution {solution_idx}:  {fitness}")

            fitness_values.append(fitness)

        return fitness_values

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

    def run_pygad(self):
        initial_mutated_population = self._initial_population
        ga_instance = pygad.GA(
            num_generations=self._ga_args.get('num_generations', 50),
            num_parents_mating=self._ga_args.get('num_parents_mating', 2),
            fitness_func=self.calculate_fitness,
            fitness_batch_size=self._ga_args.get('fitness_batch_size', len(initial_mutated_population)),
            sol_per_pop=self._ga_args.get('sol_per_pop', 100),
            num_genes=len(initial_mutated_population),
            initial_population=initial_mutated_population,
            parent_selection_type=self._ga_args.get('parent_selection_type', "sss"),
            crossover_type=self._ga_args.get('crossover_type', "single_point"),
            mutation_type=self._mutation_type,
            mutation_percent_genes=self._ga_args.get('mutation_percent_genes', 10)
        )

        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(f"Best solution: {solution}")
        print(f"Best solution fitness: {solution_fitness}")

    @property
    def boolean_model(self):
        return self._boolean_model
