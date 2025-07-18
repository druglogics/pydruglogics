pydruglogics.model.Evolution
============================

Optimizer that evolves Boolean models using genetic algorithms, fitness evaluation, and parallelization.

.. automethod:: pydruglogics.model.Evolution.Evolution.calculate_fitness

   Fitness function for GA, evaluates each candidate model using the provided training data.

   **Parameters**

   - ``ga_instance``: Instance of the GA.
   - ``solutions (list)``: Batch of solutions (binary vectors).
   - ``solution_idx (int)``: Current solution batch index.

   **Returns**

   - list: Fitness values for each solution.

.. automethod:: pydruglogics.model.Evolution.Evolution.create_initial_population

   Create an initial GA population by randomly mutating the reference Boolean model.

   **Parameters**

   - ``population_size (int)``: Population size.
   - ``num_mutations (int)``: Number of mutations per individual.
   - ``seed (int, optional)``: Seed for reproducibility.

   **Returns**

   - list: Initial population (binary vectors).

.. automethod:: pydruglogics.model.Evolution.Evolution.run

   Launch the evolutionary search, returning the best Boolean models found.

   **Returns**

   - list: Best Boolean models.

.. automethod:: pydruglogics.model.Evolution.Evolution.save_to_file_models

   Save the best models of the evolution run as `.bnet` files to disk.

   **Parameters**

   - ``base_folder (str, optional)``: Output directory.
