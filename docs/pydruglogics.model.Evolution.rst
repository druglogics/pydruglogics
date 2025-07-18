pydruglogics.model.Evolution
============================

Optimizer that evolves Boolean models using genetic algorithms, fitness evaluation, and parallelization.

.. automethod:: pydruglogics.model.Evolution.Evolution.calculate_fitness

Fitness function for GA, evaluates each candidate model using the provided training data.


.. automethod:: pydruglogics.model.Evolution.Evolution.create_initial_population

Create an initial GA population by randomly mutating the reference Boolean model.


.. automethod:: pydruglogics.model.Evolution.Evolution.run

Launch the evolutionary search, returning the best Boolean models found.


.. automethod:: pydruglogics.model.Evolution.Evolution.save_to_file_models

Save the best models of the evolution run as `.bnet` files to disk.
