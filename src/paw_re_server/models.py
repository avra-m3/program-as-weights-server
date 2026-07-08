"""Data models for describing and evaluating PAW test cases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TestCase:
    """A single input/expected-output pair to evaluate a PAW function against."""

    input: str
    expected: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class EvalResult:
    """The outcome of running one TestCase through a compiled PAW function."""

    case: TestCase
    actual: str
    passed: bool


@dataclass(frozen=True, slots=True)
class SuiteReport:
    """Aggregate results for an entire test suite run."""

    results: list[EvalResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def accuracy(self) -> float:
        return self.passed_count / self.total if self.total else 0.0

    @property
    def failures(self) -> list[EvalResult]:
        return [r for r in self.results if not r.passed]
