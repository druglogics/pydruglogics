import pytest
import numpy as np
from unittest.mock import Mock, patch, PropertyMock, call, mock_open
from pydruglogics import BooleanModel
from pydruglogics.model.Evolution import Evolution
from pydruglogics.input.TrainingData import TrainingData
from pydruglogics.utils.BNetworkUtil import BNetworkUtil
from pydruglogics.model.BooleanModelOptimizer import BooleanModelOptimizer



class TestEvolution:

    @pytest.fixture
    def mock_boolean_model(self):
        model = Mock(spec=BooleanModel)
        model.mutation_type = 'topology'
        model.clone.return_value = model
        model.binary_boolean_equations = [1, 0, 1, 1]
        return model

    @pytest.fixture
    def mock_training_data(self):
        training_data = Mock(spec=TrainingData)
        training_data.observations = [{'response': ["globaloutput:1"], 'weight': 1.0}]
        training_data.weight_sum = 1.0
        return training_data

    @pytest.fixture
    def evolution(self, mock_boolean_model, mock_training_data):
        model_outputs = Mock()
        ga_args = {'num_generations': 10, 'fitness_batch_size': 2}
        ev_args = {'num_of_runs': 2, 'num_best_solutions': 1}
        return Evolution(boolean_model=mock_boolean_model, training_data=mock_training_data,
                         model_outputs=model_outputs, ga_args=ga_args, ev_args=ev_args)

    def test_init_no_model_outputs_raises_error(self):
        with pytest.raises(ValueError, match='Model outputs are not provided.'):
            Evolution(boolean_model=Mock(), training_data=Mock(), model_outputs=None)

    def test_run_single_ga(self, evolution, mock_boolean_model):
        initial_population = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        num_best_solutions = evolution._ev_args['num_best_solutions']

        with patch('pygad.GA', autospec=True) as MockGA:
            mock_ga_instance = MockGA.return_value
            mock_ga_instance.run.return_value = None
            mock_ga_instance.last_generation_fitness = [0.9, 0.8]
            mock_ga_instance.population = initial_population
            results = evolution._run_single_ga(1, initial_population)

            assert len(results) == num_best_solutions
            assert all(len(result) == 2 for result in results)
            assert isinstance(results[0][0], np.ndarray)
            assert isinstance(results[0][1], float)

    def test_calculate_fitness(self, evolution, mock_boolean_model):
        solutions = [[1, 0, 1]]
        mock_fitness_values = [0.9]
        expected_fitness_values_scaled = [fitness * 100 for fitness in mock_fitness_values]

        with (patch('joblib.Parallel') as MockParallel,
              patch.object(evolution, '_calculate_fitness_for_solution',
                           side_effect=mock_fitness_values) as mock_fitness_func):
            mock_parallel = MockParallel.return_value
            mock_parallel.__enter__.return_value = mock_parallel
            mock_parallel.__exit__.return_value = None
            mock_parallel.return_value = mock_fitness_values

            fitness_values = evolution.calculate_fitness(None, solutions, 0)
            assert fitness_values == expected_fitness_values_scaled

    def test_create_initial_population(self, evolution, mock_boolean_model):
        population = evolution.create_initial_population(5, 2, seed=42)
        assert population.shape == (5, len(mock_boolean_model.binary_boolean_equations))
        assert not np.array_equal(population[0], population[1])

    def test_run(self, evolution, mock_boolean_model):
        mock_results = [([1, 0, 1, 1], 0.95),([0, 1, 0, 1], 0.90)]

        with patch.object(evolution, '_run_single_ga', return_value=mock_results) as mock_run_ga, \
                patch.object(evolution, 'create_initial_population',
                             return_value=np.array([[1, 0, 1, 1], [0, 1, 0, 1]])), \
                patch.object(mock_boolean_model, 'clone') as mock_clone:
            mock_cloned_model = mock_clone.return_value
            mock_cloned_model.model_name = "test_model"
            best_models = evolution.run()
            assert len(best_models) == evolution._ev_args['num_of_runs'] * len(mock_results)

            for model in best_models:
                assert model.model_name.startswith("e")
                assert model.fitness in [0.95, 0.90]

    def test_save_to_file_models_success(self, evolution):
        with patch('os.makedirs'), patch('builtins.open', mock_open()) as mocked_open:
            mock_model = Mock(model_name='e1_s1', fitness=0.95)
            mock_model.updated_boolean_equations = [
                ("A", {"B": 1}, {}, ""),
                ("B", {"D": 1}, {"E": 1}, "&"),
                ("K", {"D": 1, "H": 1}, {"E": 1}, "|"),
                ("C", {}, {"G": 1}, ""),
                ("D", {"A": 1}, {"B": 1, "C": 1}, "&"),
                ("E", {"A": 1, "F": 1}, {"B": 1}, "&"),
                ("F", {}, {}, ""),
                ("G", {}, {"B": 1, "F": 1}, ""),
                ("H", {"A": 1, "F": 1}, {}, ""),
                ("I", {"A": 1, "B": 1, "C": 1}, {}, ""),
                ("J", {}, {"D": 1, "E": 1, "F": 1}, ""),
                ("L", {}, {}, ""),
                ("M", {"X": 1}, {}, "")
            ]
            evolution._best_boolean_models = [mock_model]
            evolution.save_to_file_models(base_folder='/test/models')
            file_path, mode = mocked_open.call_args[0]

            assert file_path.startswith('/test/models/models_'), "Unexpected folder structure in file path."
            assert file_path.endswith('e1_s1.bnet'), "File name does not match expected output."
            assert mode == 'w', "File was not opened in write mode."

    def test_save_to_file_models_io_error(self, evolution):
        mock_model = Mock()
        mock_model.model_name = 'e1_s1'
        mock_model.fitness = 0.95
        mock_model.boolean_equations = [('A', {'B': 1}, {'C': 0}, '&')]
        evolution._best_boolean_models = [mock_model]

        with patch('os.makedirs'), patch('builtins.open', side_effect=IOError("Error writing file")), \
                patch('pydruglogics.utils.BNetworkUtil.BNetworkUtil.to_bnet_format', return_value="A, B & C\n"):
            with pytest.raises(IOError, match="Error writing file"):
                evolution.save_to_file_models(base_folder='/test/models')

    def test_best_boolean_models_property(self, evolution):
        mock_model = Mock()
        evolution._best_boolean_models = [mock_model]
        assert evolution.best_boolean_models == [mock_model]

    def test_callback_generation(self, evolution):
        mock_ga_instance = Mock()
        mock_ga_instance.generations_completed = 5
        mock_ga_instance.last_generation_fitness = [0.9, 0.85, 0.87]

        with patch('logging.debug') as mock_logging_debug:
            evolution._callback_generation(mock_ga_instance)
            mock_logging_debug.assert_called_once_with("Generation 5: Fitness values: [0.9, 0.85, 0.87]")

    def test_create_default_training_data(self, evolution):
        training_data = evolution._create_default_training_data()
        assert len(training_data.observations) == 1
        assert training_data.observations[0] == {'condition': ['-'], 'response': ['globaloutput:1'], 'weight': 1.0}

    def test_calculate_fitness_for_solution(self, evolution, mock_boolean_model):
        mock_solution = [1, 0, 1]
        mock_model_clone = Mock()
        mock_model_clone.from_binary.return_value = None
        mock_model_clone.has_attractors.return_value = True
        mock_model_clone.calculate_attractors.return_value = None
        mock_model_clone.calculate_global_output.return_value = 0.9
        mock_model_clone.has_stable_states.return_value = True
        mock_model_clone.attractors = [{'Node1': 1, 'Node2': 0}]

        evolution._boolean_model.clone.return_value = mock_model_clone
        evolution._training_data = Mock()
        evolution._training_data.observations = [{'response': ['globaloutput:1'], 'weight': 1.0}]
        evolution._training_data.weight_sum = 1.0

        with patch('logging.debug') as mock_logging_debug:
            fitness = evolution._calculate_fitness_for_solution(mock_solution)
            assert fitness == 1.0 - abs(0.9 - 1.0)
            mock_logging_debug.assert_any_call("Observed Global Output: 1.00, ")
            mock_logging_debug.assert_any_call("Predicted Global Output: 0.90")

    def test_calculate_fitness_for_solution_with_stable_states(self, evolution, mock_boolean_model):
        mock_solution = [1, 0, 1]
        mock_model_clone = Mock()
        mock_model_clone.from_binary.return_value = None
        mock_model_clone.has_attractors.return_value = True
        mock_model_clone.has_stable_states.return_value = True
        mock_model_clone.attractors = [{'Node1': 1, 'Node2': 0},{'Node1': 0, 'Node2': 1}]
        evolution._boolean_model.clone.return_value = mock_model_clone
        evolution._training_data = Mock()
        evolution._training_data.observations = [{'response': ['Node1:1', 'Node2:0'], 'weight': 1.0}]
        evolution._training_data.weight_sum = 1.0

        fitness = evolution._calculate_fitness_for_solution(mock_solution)

        assert fitness > 0
        assert mock_model_clone.calculate_attractors.called
        assert mock_model_clone.has_stable_states.called

    def test_calculate_fitness_for_solution_division_by_found_observations(self, evolution, mock_boolean_model):
        mock_solution = [1, 0, 1]
        mock_model_clone = Mock()
        mock_model_clone.from_binary.return_value = None
        mock_model_clone.has_attractors.return_value = True
        mock_model_clone.has_stable_states.return_value = False
        mock_model_clone.attractors = [{'Node1': 1, 'Node2': 0}]
        mock_model_clone.calculate_attractors.return_value = None
        mock_model_clone.get = lambda node_name, default: mock_model_clone.attractors[0].get(node_name, default)

        evolution._boolean_model.clone.return_value = mock_model_clone
        evolution._training_data = Mock()
        evolution._training_data.observations = [{'response': ['Node1:1', 'Node2:0'], 'weight': 1.0}]
        evolution._training_data.weight_sum = 1.0
        fitness = evolution._calculate_fitness_for_solution(mock_solution)
        assert fitness > 0, "Expected positive fitness, got zero or negative value"
        assert mock_model_clone.calculate_attractors.called, "Attractors calculation was not called"
