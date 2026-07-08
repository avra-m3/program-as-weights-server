"""Generic test harness for ProgramAsWeights (PAW) functions.

Workflow:
    1. Write a natural-language spec in a ``.txt`` file under ``specs/``.
    2. Write input/expected test cases in a YAML file under ``data/``.
    3. Run this module, pointing it at both files. It will compile (or reuse)
       the PAW program, run it over every test case, and print a pass/fail
       report.

This module is intentionally task-agnostic: it doesn't know anything about
what the spec is supposed to do, it just compiles + evaluates.

Usage:
    uv run python -m paw_re_server.runner \\
        --spec specs/my_task.txt \\
        --data data/my_task.yaml

    # Reuse an already-compiled program instead of recompiling:
    uv run python -m paw_re_server.runner \\
        --program-id da03/my-classifier \\
        --data data/my_task.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import programasweights as paw

from paw_re_server.data import load_spec, load_test_cases
from paw_re_server.models import EvalResult, SuiteReport, TestCase

PawFunction = Callable[..., str]


def get_function(
    *,
    spec_path: str | Path | None = None,
    program_id: str | None = None,
    slug: str | None = None,
    compiler: str | None = None,
) -> PawFunction:
    """Return a callable PAW function, compiling from a spec if needed.

    Precedence: ``program_id`` (reuse, no network compile) > ``spec_path``
    (compile fresh, then load).
    """
    if program_id:
        return paw.function(program_id)

    if not spec_path:
        raise ValueError("Provide either --program-id or --spec")

    spec = load_spec(spec_path)
    program = paw.compile(spec, slug=slug, compiler=compiler)
    print(
        f"Compiled program id={program.id!r} slug={program.slug!r} "
        f"(save --program-id {program.id} to reuse without recompiling)",
        file=sys.stderr,
    )
    return paw.function(program.id)


def run_case(fn: PawFunction, case: TestCase) -> EvalResult:
    """Run a single test case through the PAW function."""
    actual = fn(case.input)
    return EvalResult(case=case, actual=actual, passed=actual == case.expected)


def run_suite(fn: PawFunction, cases: list[TestCase]) -> SuiteReport:
    """Run every test case and collect a report."""
    return SuiteReport(results=[run_case(fn, case) for case in cases])


def print_report(report: SuiteReport) -> None:
    """Pretty-print a suite report to stdout."""
    for i, result in enumerate(report.results):
        label = result.case.name or f"case #{i}"
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {label}")
        if not result.passed:
            print(f"    input:    {result.case.input!r}")
            print(f"    expected: {result.case.expected!r}")
            print(f"    actual:   {result.actual!r}")

    print()
    print(f"{report.passed_count}/{report.total} passed ({report.accuracy:.0%})")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    program_source = parser.add_mutually_exclusive_group(required=True)
    program_source.add_argument(
        "--spec", type=Path, help="Path to a spec .txt file to compile."
    )
    program_source.add_argument(
        "--program-id",
        help="Reuse an already-compiled program id/slug instead of compiling.",
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to a YAML test-data file."
    )
    parser.add_argument("--slug", help="Optional slug to name a fresh compile.")
    parser.add_argument(
        "--compiler", help="Optional compiler override (e.g. paw-ft-bs48)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    fn = get_function(
        spec_path=args.spec,
        program_id=args.program_id,
        slug=args.slug,
        compiler=args.compiler,
    )
    cases = load_test_cases(args.data)
    report = run_suite(fn, cases)
    print_report(report)

    return 0 if report.accuracy == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
