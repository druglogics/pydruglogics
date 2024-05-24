from gitsbe.model.BooleanEquation import BooleanEquation
from gitsbe.model.GeneralModel import GeneralModel
from gitsbe.model.Interaction import Interaction

if __name__ == '__main__':
    # GeneralModel
    gm = GeneralModel()
    gm.load_sif_file('../example_model_args/toy_ags_network.sif')
    gm.remove_interactions(True, True)
    gm.remove_self_regulated_interactions()
    gm.build_multiple_interactions()
    print('Interactions')
    print(gm)

    # MultipleInteractions
    multiple_interactions = Interaction('->', 'CASP2', 'CASP5')
    multiple_interactions.add_activating_regulator('CASP3')
    multiple_interactions.add_activating_regulator('FOXO5')
    multiple_interactions.add_activating_regulator('FOXO7')
    multiple_interactions.add_inhibitory_regulator('RAC1')
    multiple_interactions.add_inhibitory_regulator('RAC9')
    multiple_interactions.add_inhibitory_regulator('RAC3')
    print("Multiple Interactions:")
    print(multiple_interactions)
    boolean_eq_string = BooleanEquation(multiple_interactions)

    boolean_equation = BooleanEquation(boolean_eq_string)
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
