import pytest
from gitsbe.model.BooleanEquation import BooleanEquation


@pytest.fixture
def equation():
    return BooleanEquation()


def test_initialization_with_null():
    null_equation = BooleanEquation(None)
    assert null_equation.target == '', 'Null passed, equation should be initialized with an empty target'


def test_initialization_with_valid_expression_without_link():
    expression = 'A *= not ( B )'
    valid_equation = BooleanEquation(expression)
    assert valid_equation.target == 'A', 'Equation should be initialized with the correct target'
    assert valid_equation.get_values_activating_regulators() == [], 'Activating regulators should match the expression'
    assert valid_equation.get_values_inhibitory_regulators() == [1], 'Inhibitory regulators should match the expression'


def test_mutate_link_operator(equation):
    equation.link = 'and'
    original_link = equation.link
    equation.mutate_link_operator()
    assert equation.link != original_link, 'Link operator should be mutated'


def test_convert_to_sif_lines(equation):
    equation.target = 'A'
    equation.activating_regulators = {'B': 1, 'C': 1}
    equation.inhibitory_regulators = {'D': 1}
    sif_lines = equation.convert_to_sif_lines(' ')
    expected_lines = ['B -> A', 'C -> A', 'D -| A']
    assert sif_lines == expected_lines, 'SIF lines should match the expected output'


def test_modify_activating_values_from_list(equation):
    equation.activating_regulators = {'B': 1, 'C': 0}
    new_values = [0, 1]
    equation.modify_activating_values_from_list(new_values)
    assert equation.get_values_activating_regulators() == new_values, ('Activating regulators '
                                                                       'should be updated with new values')


def test_modify_inhibitory_values_from_list(equation):
    equation.inhibitory_regulators = {'D': 1, 'E': 0}
    new_values = [0, 1]
    equation.modify_inhibitory_values_from_list(new_values)
    assert equation.get_values_inhibitory_regulators() == new_values, ('Inhibitory regulators '
                                                                       'should be updated with new values')


def test_modify_link_from_list(equation):
    new_values = [1]
    equation.modify_link_from_list(new_values)
    assert equation.link == 'and', 'Link operator should be updated with new value'
