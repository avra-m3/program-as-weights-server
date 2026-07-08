from pathlib import Path

import pytest

from paw_re_server.data import load_test_cases
from paw_re_server.models import TestCase

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_test_cases_parses_yaml() -> None:
    cases = load_test_cases(FIXTURES / "sample_cases.yaml")

    assert cases == [
        TestCase(input="I love this!", expected="positive", name="happy path"),
        TestCase(input="This is terrible.", expected="negative", name=None),
    ]


def test_load_test_cases_rejects_non_list(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("input: not-a-list\n")

    with pytest.raises(ValueError, match="must contain a YAML list"):
        load_test_cases(bad_file)


def test_load_test_cases_requires_input_and_expected(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("- input: only input, no expected\n")

    with pytest.raises(ValueError, match="missing required"):
        load_test_cases(bad_file)
