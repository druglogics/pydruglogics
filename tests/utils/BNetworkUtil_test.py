import pytest
import os
from unittest.mock import patch, mock_open
from pydruglogics.utils.BNetworkUtil import BNetworkUtil

class TestBNetworkUtil:

    def test_get_file_extension_with_extension(self):
        assert BNetworkUtil.get_file_extension("example.txt") == "txt"

    def test_get_file_extension_no_extension(self):
        assert BNetworkUtil.get_file_extension("example") == ""
        assert BNetworkUtil.get_file_extension("") == ""

    def test_remove_extension(self):
        assert BNetworkUtil.remove_extension("example.txt") == "example"
        assert BNetworkUtil.remove_extension("/path/to/example.txt") == "example"
        assert BNetworkUtil.remove_extension("example") == "example"

    @patch("builtins.open", new_callable=mock_open, read_data="line1\n# comment\nline2\n\nline3\n")
    def test_read_lines_from_file_skip_empty_and_comments(self, mock_file):
        lines = BNetworkUtil.read_lines_from_file("test.txt")
        assert lines == ["line1", "line2", "line3"]
        mock_file.assert_called_once_with("test.txt", "r")

    @patch("builtins.open", new_callable=mock_open, read_data="line1\n# comment\nline2\n\nline3\n")
    def test_read_lines_from_file_include_empty_and_comments(self, mock_file):
        lines = BNetworkUtil.read_lines_from_file("test.txt", skip_empty_lines_and_comments=False)
        assert lines == ["line1", "# comment", "line2", "", "line3"]
        mock_file.assert_called_once_with("test.txt", "r")

    @pytest.mark.parametrize("value,expected", [
        ("123", True),
        ("12.3", True),
        ("abc", False),
        ("", False),
        (123, True),
        (12.3, True)
    ])
    def test_is_numeric_string(self, value, expected):
        assert BNetworkUtil.is_numeric_string(value) == expected

    def test_parse_interaction_valid(self):
        result = BNetworkUtil.parse_interaction("A activates B")
        assert result == {
            'source': 'A',
            'target': 'B',
            'arc': 1,
            'activating_regulators': [],
            'inhibitory_regulators': []
        }

    def test_parse_interaction_invalid_format(self):
        with pytest.raises(ValueError, match="ERROR: Wrongly formatted interaction"):
            BNetworkUtil.parse_interaction("A ->")

    def test_parse_interaction_invalid_type(self):
        with pytest.raises(ValueError, match="ERROR: Unrecognized interaction type"):
            BNetworkUtil.parse_interaction("A interacts_with B")

    def test_create_interaction(self):
        result = BNetworkUtil.create_interaction("C")
        assert result == {
            'target': 'C',
            'activating_regulators': [],
            'inhibitory_regulators': []
        }

    def test_bnet_string_to_dict(self):
        bnet_string = "A, A & B\nB, !C\n"
        result = BNetworkUtil.bnet_string_to_dict(bnet_string)
        assert result == {"A": "A & B", "B": "!C"}

    def test_to_bnet_format(self):
        boolean_equations = [
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
            ("M", {"X": 1}, {}, ""),
        ]

        result = BNetworkUtil.to_bnet_format(boolean_equations)
        expected = (
            "A, (B)\n"
            "B, (D) & !(E)\n"
            "K, ((D) | H) | !(E)\n"
            "C, !(G)\n"
            "D, (A) & !((B) | C)\n"
            "E, ((A) | F) & !(B)\n"
            "F, 0\n"
            "G, !((B) | F)\n"
            "H, ((A) | F)\n"
            "I, (((A) | B) | C)\n"
            "J, !(((D) | E) | F)\n"
            "L, 0\n"
            "M, (X)"
        )

        assert result == expected

    def test_create_equation_from_bnet_with_activation_only(self):
        equation_str = "X, A | B"
        result = BNetworkUtil.create_equation_from_bnet(equation_str)
        assert result == ("X", {"A": 1, "B": 1}, {}, "")

    def test_create_equation_from_bnet_with_inhibition(self):
        equation_str = "Y, A & !B"
        result = BNetworkUtil.create_equation_from_bnet(equation_str)
        assert result == ("Y", {"A": 1}, {"B": 1}, "&")

    @pytest.mark.parametrize("value,expected", [(None, False), ("not_a_number", False),])
    def test_is_numeric_string_edge_cases(self, value, expected):
        assert BNetworkUtil.is_numeric_string(value) == expected

    def test_create_equation_from_bnet_with_pipe_in_regulators(self):
        equation_str = "Z, A | !B"
        result = BNetworkUtil.create_equation_from_bnet(equation_str)
        assert result == ("Z", {"A": 1}, {"B": 1}, "|")

    @pytest.mark.parametrize("file_ext,expected", [(".hiddenfile", ".hiddenfile"),
                                                   ("path/to/.hiddenfile", ".hiddenfile"),
                                                   ("path/to/filename", "filename")])
    def test_remove_extension_edge_cases(self, file_ext, expected):
        assert BNetworkUtil.remove_extension(file_ext) == expected