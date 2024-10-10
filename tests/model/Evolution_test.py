import pytest
from unittest.mock import Mock, patch
from gitsbe.model.BooleanModel import BooleanModel
from gitsbe.input.TrainingData import TrainingData
from gitsbe.model.Evolution import Evolution


class TestEvolution:

    @pytest.fixture
    def boolean_model(self):
        return Mock(spec=BooleanModel)

    @pytest.fixture
    def ga_args(self):
        return {
            'number_of_mutations': 2,
            'mutation_type': 'balanced',
            'population_size': 10,
            'num_parents_mating': 2,
            'fitness_batch_size': 10,
            'sol_per_pop': 10,
            'parent_selection_type': "sss",
            'crossover_type': "single_point",
            'mutation_percent_genes': 10,
            'num_generations': 10
        }

    @pytest.fixture
    def evolution(self, boolean_model, ga_args):
        return Evolution(boolean_model, ga_args)

    def test_select_mutation(self, evolution):
        assert evolution.select_mutation('balanced') == evolution.balanced_mutation
        assert evolution.select_mutation('topology') == evolution.topology_mutation
        assert evolution.select_mutation('mixed') == evolution.mixed_mutation
        with pytest.raises(ValueError):
            evolution.select_mutation('unknown')

    def test_create_initial_population(self, evolution, boolean_model):
        boolean_model.to_binary.return_value = [0, 1, 0, 1, 0, 1, 0]
        initial_population = evolution.create_initial_population('balanced', 3)
        assert len(initial_population) == 3
        boolean_model.balance_mutation.assert_called_with(evolution._mutation_number)
        boolean_model.to_binary.assert_called_with('balanced')
