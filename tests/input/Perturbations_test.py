import pytest
from unittest.mock import MagicMock
from pydruglogics.input.Perturbations import Perturbation

class TestPerturbation:
    @pytest.fixture
    def mock_logger(self):
        logger = MagicMock()
        return logger

    @pytest.fixture
    def drug_data(self):
        # Mock drug data: Name, Targets, Effect (optional)
        return [
            ['Drug1', 'TargetA,TargetB', 'inhibits'],
            ['Drug2', 'TargetC', 'activates'],
            ['Drug3', 'TargetD']
        ]

    @pytest.fixture
    def perturbation_data(self):
        return [
            ['Drug1', 'Drug2'],
            ['Drug2', 'Drug3']
        ]

    def test_init_with_drug_data(self, drug_data, perturbation_data, mock_logger):
        perturbation = Perturbation(drug_data=drug_data, perturbation_data=perturbation_data, verbosity=1)
        assert len(perturbation.drugs) == 3
        assert len(perturbation.perturbations) == 2
        assert perturbation._logger == mock_logger

    def test_init_without_drug_data_raises_exception(self):
        with pytest.raises(ValueError, match='Please provide drug data.'):
            Perturbation(drug_data=None)

    def test_init_without_perturbation_data_initiates_from_drug_panel(self, drug_data):
        perturbation = Perturbation(drug_data=drug_data)
        # Check if perturbations are initialized from drug panel
        assert len(perturbation.perturbations) > 0

    def test_load_drug_panel_from_data(self, drug_data, mock_logger):
        perturbation = Perturbation(drug_data=drug_data)
        assert perturbation.drugs[0]['name'] == 'Drug1'
        assert perturbation.drugs[0]['targets'] == ['TargetA', 'TargetB']
        assert perturbation.drugs[0]['effect'] == 'inhibits'

    def test_load_perturbations_from_data(self, drug_data, perturbation_data, mock_logger):
        perturbation = Perturbation(drug_data=drug_data, perturbation_data=perturbation_data)
        assert len(perturbation.perturbations) == 2

    def test_perturbation_representation(self, drug_data, perturbation_data):
        perturbation = Perturbation(drug_data=drug_data, perturbation_data=perturbation_data)
        expected_output = '[Drug1 (targets: TargetA, TargetB), Drug2 (targets: TargetC)]\n[Drug2 (targets: TargetC), Drug3 (targets: TargetD)]'
        assert str(perturbation) == expected_output

    def test_drug_names_property(self, drug_data):
        perturbation = Perturbation(drug_data=drug_data)
        assert perturbation.drug_names == ['Drug1', 'Drug2', 'Drug3']

    def test_drug_effects_property(self, drug_data):
        perturbation = Perturbation(drug_data=drug_data)
        assert perturbation.drug_effects == ['inhibits', 'activates', 'inhibits']  # Default effect is 'inhibits'

    def test_drug_targets_property(self, drug_data):
        perturbation = Perturbation(drug_data=drug_data)
        assert perturbation.drug_targets == [['TargetA', 'TargetB'], ['TargetC'], ['TargetD']]
