import pytest

from paw_re_server.models import TestCase
from paw_re_server.runner import run_case, run_suite


def fake_fn(text: str) -> str:
    """A stand-in for a compiled PAW function, no network required."""
    return "positive" if "love" in text.lower() else "negative"


def test_run_case_pass() -> None:
    case = TestCase(input="I love this!", expected="positive")
    result = run_case(fake_fn, case)

    assert result.passed
    assert result.actual == "positive"


def test_run_case_fail() -> None:
    case = TestCase(input="I love this!", expected="negative")
    result = run_case(fake_fn, case)

    assert not result.passed


def test_run_suite_reports_accuracy() -> None:
    cases = [
        TestCase(input="I love this!", expected="positive"),
        TestCase(input="This is terrible.", expected="negative"),
        TestCase(input="I love this too!", expected="negative"),  # wrong on purpose
    ]

    report = run_suite(fake_fn, cases)

    assert report.total == 3
    assert report.passed_count == 2
    assert report.accuracy == pytest.approx(2 / 3)
    assert len(report.failures) == 1
