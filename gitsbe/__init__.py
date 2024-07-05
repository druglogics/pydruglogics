from gitsbe.input.TrainingData import TrainingData
from gitsbe.model.BooleanEquation import BooleanEquation
from gitsbe.model.BooleanModel import BooleanModel
from gitsbe.model.Evolution import Evolution
from gitsbe.model.InteractionModel import InteractionModel
from gitsbe.input.ModelOutputs import ModelOutputs

import pygad
import numpy as np

if __name__ == '__main__':
    # Interaction
    interaction = InteractionModel()
    interaction.load_sif_file('../example_model_args/toy_ags_network.sif')
    interaction.remove_interactions(True, True)
    interaction.remove_self_regulated_interactions()
    interaction.build_multiple_interactions()
    print('Interactions')
    print(interaction)

    for i in range(interaction.size()):
        # BooleanEquation
        boolean_equation = BooleanEquation(interaction, i)
        print('Activating regulators')
        print(boolean_equation.activating_regulators)
        print('\nInhibitory regulators: ')
        print(boolean_equation.inhibitory_regulators)

        print('\nBefore mutate regulator: ')
        print(boolean_equation.get_boolean_equation())
        boolean_equation.mutate_regulator()
        print('After mutate regulator: ')
        print(boolean_equation.get_boolean_equation())

        print('\nBefore mutate link operator: ')
        print(boolean_equation.get_boolean_equation())
        boolean_equation.mutate_link_operator()
        print('After mutate link operator: ')
        print(boolean_equation.get_boolean_equation())

        print('Covert to sif lines: ')
        print(boolean_equation.convert_to_sif_lines('\t'))

    # ModelOutputs
    modeloutputs = ModelOutputs('../example_model_args/toy_ags_modeloutputs.tab')
    ModelOutputs.initialize('../example_model_args/toy_ags_modeloutputs.tab')

    training_data = TrainingData('../example_model_args/toy_ags_training_data.tab')
    TrainingData.initialize('../example_model_args/toy_ags_training_data.tab')

    # Boolean Model init, PyGAD run
    boolean_model_bnet = BooleanModel(file='../example_model_args/ap-1_else-0_wt.bnet', model_name='test')

    ga_args = {
        'num_generations': 50,
        'num_parents_mating': 2,
        'fitness_batch_size': 10,
        'sol_per_pop': 100,
        'num_genes': 10,
        'parent_selection_type': "sss",
        'crossover_type': "single_point",
        'mutation_percent_genes': 20,
        'mutation_type': 'mixed',  # can be 'balanced', 'topology' or 'mixed'
        'population_size': 20,
        'number_of_mutations': 2
    }

    evolution = Evolution(boolean_model_bnet, ga_args)
    evolution.run_pygad()
