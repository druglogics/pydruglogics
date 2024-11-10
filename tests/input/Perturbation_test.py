import pytest
from unittest.mock import MagicMock, patch
from pydruglogics.input.Perturbations import Perturbation
import logging


class TestPerturbation:
    @pytest.fixture
    def mock_logger(self):
        logger = MagicMock()
        return logger

    @pytest.fixture
    def drug_data(self):
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

    @pytest.fixture
    def perturbation_data_incorrect(self):
        return [
            ['DummyDrug']
        ]

    @pytest.fixture
    def drug_data_incorrect(self):
        return [
            ['DrugWithNoTargets'],  # No targets
            ['', 'TargetWithoutName']  # No name
        ]

    def test_init_without_drug_data_raises_exception(self):
        with pytest.raises(ValueError, match='Please provide drug data.'):
            Perturbation(drug_data=None)

    def test_init_without_perturbation_data_initiates_from_drug_panel(self, drug_data):
        perturbation = Perturbation(drug_data=drug_data)
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
        expected_output = ('[Drug1 (targets: TargetA, TargetB), Drug2 (targets: TargetC)]\n'
                           '[Drug2 (targets: TargetC), Drug3 (targets: TargetD)]')
        assert str(perturbation) == expected_output

    def test_drug_names_property(self, drug_data):
        perturbation = Perturbation(drug_data=drug_data)
        assert perturbation.drug_names == ['Drug1', 'Drug2', 'Drug3']

    def test_drug_effects_property(self, drug_data):
        perturbation = Perturbation(drug_data=drug_data)
        assert perturbation.drug_effects == ['inhibits', 'activates', 'inhibits']

    def test_drug_targets_property(self, drug_data):
        perturbation = Perturbation(drug_data=drug_data)
        assert perturbation.drug_targets == [['TargetA', 'TargetB'], ['TargetC'], ['TargetD']]

    def test_no_perturbations_available(self, drug_data):
        perturbation = Perturbation(drug_data=drug_data, perturbation_data=[])
        assert str(perturbation) == 'No perturbations available.'

    def test_load_drug_panel_from_data_missing_targets(self, drug_data_incorrect):
        perturbation = Perturbation(drug_data=[])

        with pytest.raises(ValueError, match="Each drug entry must contain at least 'name' and 'targets'."):
            perturbation._load_drug_panel_from_data(drug_data_incorrect)

    def test_load_drug_panel_from_data_empty_list(self):
        perturbation = Perturbation(drug_data=[])
        empty_data = []
        perturbation._load_drug_panel_from_data(empty_data)

        assert perturbation._drug_panel == []

    def test_load_perturbations_with_empty_entries(self, drug_data):
        perturbation_data_with_empty_entries = [
            ['Drug1', 'Drug2'],
            [],  # Empty entry
            ['Drug3']
        ]
        with patch('logging.warning') as mock_warning:
            perturbation = Perturbation(drug_data=drug_data, perturbation_data=perturbation_data_with_empty_entries)
            assert len(perturbation.perturbations) == 2
            mock_warning.assert_called_once_with("Some perturbation entries were empty and have been ignored.")

    def test_load_perturbations_with_missing_drug_in_panel(self, drug_data):
        perturbation_data_with_missing_drug = [
            ['Drug1', 'UnknownDrug'],
            ['Drug2', 'Drug3']
        ]
        with patch('logging.warning') as mock_warning:
            perturbation = Perturbation(drug_data=drug_data, perturbation_data=perturbation_data_with_missing_drug)
            assert len(perturbation.perturbations) == 1
            mock_warning.assert_called_once_with('Some drugs in the perturbation were not found in the drug panel.')


    def test_print_method_handles_exceptions(self, drug_data):
        class FaultyPerturbation(Perturbation):
            def __str__(self) -> str:
                raise RuntimeError("Intentional error for testing")

        perturbation = FaultyPerturbation(drug_data=drug_data)
        with patch('builtins.print') as mock_print:
            perturbation.print()

            mock_print.assert_any_call("An error occurred while printing Perturbation: Intentional error for testing")
