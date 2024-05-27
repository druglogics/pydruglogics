from gitsbe.model.BooleanEquation import BooleanEquation
from gitsbe.model.Interaction import Interaction

if __name__ == '__main__':
    # Interaction
    interaction = Interaction()
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
        print(boolean_equation.get_activating_regulators())
        print('\nInhibitory regulators: ')
        print(boolean_equation.get_inhibitory_regulators())

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

        print('\nBefore shuffle regulatory priority: ')
        print(boolean_equation.get_boolean_equation())
        boolean_equation.shuffle_random_regulatory_priority()
        print('After shuffle regulatory priority: ')
        print(boolean_equation.get_boolean_equation())

        print('Covert to sif lines: ')
        print(boolean_equation.convert_to_sif_lines('\t'))
