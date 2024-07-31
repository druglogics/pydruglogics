from gitsbe.input.TrainingData import TrainingData
from gitsbe.model.BooleanModel import BooleanModel
from gitsbe.model.Evolution import Evolution
from gitsbe.model.InteractionModel import InteractionModel
from gitsbe.input.ModelOutputs import ModelOutputs
import time

if __name__ == '__main__':
    # Interaction
    interaction = InteractionModel()
    interaction.load_sif_file('../example_model_args/toy_ags2_network.sif')
    interaction.build_multiple_interactions()
    print('Interactions')
    print(interaction)

    # ModelOutputs
    model_outputs = ModelOutputs('../example_model_args/toy_ags2_modeloutputs.tab')

    # TrainingData
    training_data = TrainingData('../example_model_args/toy_ags2_training_data.tab')

    # BooleanModel init from .bnet file
    boolean_model_bnet = BooleanModel(file='../example_model_args/toy_ags2_equations.bnet', model_name='test')

    # BooleanModel init from .sif file
    boolean_model_sif = BooleanModel(model=interaction, model_name='test2')

    # init pygad.GA
    ga_args = {
        'num_generations': 20,
        'num_parents_mating': 2,
        'sol_per_pop': 10,
        'num_genes': 10,
        'parent_selection_type': "sss",
        'crossover_type': "single_point",
        'mutation_percent_genes': 40,
        'population_size': 20,
        'number_of_mutations': 3,
        'parallel_processing': 4
    }
    start_time = time.time()
    evolution = Evolution(boolean_model=boolean_model_bnet, training_data=training_data,
                          model_outputs=model_outputs, num_best_solutions=3, num_mutations=5,
                          num_mutations_per_eq=3, num_runs=20, num_cores=4, ga_args=ga_args)
    evolution.run()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time} seconds")
