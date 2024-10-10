import pytest
from gitsbe import InteractionModel
from gitsbe.utils.Util import Util


@pytest.fixture
def model():
    return InteractionModel()


def test_initialization_with_null():
    null_model = InteractionModel(None)
    assert null_model.size() == 0, 'Null passed, the model should be initialized with an empty list.'


def test_initialization_with_empty_list():
    empty_model = InteractionModel([])
    assert empty_model.size() == 0, 'Empty list passed, the model should be initialized with an empty list'


def test_initialization_with_valid_list():
    interactions = [
        {'source': 'A', 'target': 'B', 'arc': 1},
        {'source': 'C', 'target': 'D', 'arc': -1},
    ]
    valid_model = InteractionModel(interactions)
    assert valid_model.size() == 2, 'The model should be initialized with the provided test interactions'
    assert valid_model.interactions == interactions, 'The model interactions should match the provided test interactions'


def test_load_non_sif_file(mocker):
    mocker.patch.object(Util, 'get_file_extension', return_value='txt')
    model = InteractionModel()
    with pytest.raises(IOError) as exception:
        model.load_sif_file('interactions.bnet')
    assert 'ERROR: The extension needs to be .sif (other formats not yet supported)' in str(exception.value)


def test_load_empty_sif_file(mocker):
    mocker.patch.object(Util, 'get_file_extension', return_value='sif')
    mocker.patch.object(Util, 'read_lines_from_file', return_value=[])
    model = InteractionModel()
    model.load_sif_file('empty_interactions.sif')
    assert model.size() == 0, 'The model should have no interactions after loading an empty SIF file'


def test_load_sif_file(mocker):
    interactions = ['A |-| B', 'C |-> B', 'E <-| F', 'G <-> A']
    mocker.patch.object(Util, 'get_file_extension', return_value='sif')
    mocker.patch.object(Util, 'read_lines_from_file', return_value=interactions)
    mocker.patch.object(Util, 'parse_interaction', side_effect=lambda x: {'interaction': x})

    model = InteractionModel()
    model.load_sif_file('valid.sif')
    assert model.size() == 8, 'The model should have twice as many lines as the sif file.'


def test_remove_self_regulated_interactions(mocker):
    interactions = ['A <- A', 'B -> C', 'C <-| C']
    mocker.patch.object(Util, 'get_file_extension', return_value='sif')
    mocker.patch.object(Util, 'read_lines_from_file', return_value=interactions)
    mocker.patch.object(Util, 'parse_interaction',
                        side_effect=lambda x: {'source': x.split(' ')[0], 'target': x.split(' ')[-1]})

    model = InteractionModel()
    model.load_sif_file('interactions.sif')
    model.remove_self_regulated_interactions()
    assert model.size() == 1, 'The model should have removed self-regulated interactions'


def test_build_multiple_interactions(mocker):
    interactions = ['A -> B', 'C -> B', 'D -| B']
    mocker.patch.object(Util, 'get_file_extension', return_value='sif')
    mocker.patch.object(Util, 'read_lines_from_file', return_value=interactions)
    mocker.patch.object(Util, 'parse_interaction',
                        side_effect=lambda x: {'source': x.split(' ')[0], 'target': x.split(' ')[-1],
                                               'arc': 1 if '->' in x else -1})
    mocker.patch.object(Util, 'create_interaction',
                        side_effect=lambda target: {'target': target, 'activating_regulators': [],
                                                    'inhibitory_regulators': []})

    model = InteractionModel()
    model.load_sif_file('interactions.sif')
    model.build_multiple_interactions()
    assert model.size() == 4, 'The model should have combined interactions correctly'
