from gitsbe.input.TrainingData import TrainingData
from gitsbe.model.BooleanEquation import BooleanEquation
from gitsbe.model.BooleanModel import BooleanModel
from gitsbe.model.InteractionModel import InteractionModel
from gitsbe.input.ModelOutputs import ModelOutputs

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

        print('\nBefore mutate random operator: ')
        print(boolean_equation.get_boolean_equation())
        boolean_equation.mutate_random_operator()
        print('After random operator: ')
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

    # Attractor calculation
    boolean_model_bnet = BooleanModel(file='../example_model_args/ap-1_else-0_wt.bnet', model_name='test')
    print('\nAttractors')
    print(boolean_model_bnet.attractors)
    print('\nGlobal output')
    print(boolean_model_bnet.calculate_global_output())

    print('\n Training data')
    training_data = TrainingData('../example_model_args/toy_ags_training_data.tab')
    TrainingData.initialize('../example_model_args/toy_ags_training_data.tab')
    print(training_data)

    boolean_model_bnet.calculate_fitness('biolqm')
    print(boolean_model_bnet.topology_mutations(2))
