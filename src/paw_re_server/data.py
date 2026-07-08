"""Loading test data (and spec text) from disk."""

from __future__ import annotations

from pathlib import Path

import yaml

from paw_re_server.models import TestCase


def load_spec(spec_path: str | Path) -> str:
    """Read a natural-language PAW spec from a text file."""
    path = Path(spec_path)
    return path.read_text(encoding="utf-8").strip()


def load_test_cases(data_path: str | Path) -> list[TestCase]:
    """Load test cases from a YAML file.

    Expected schema (see data/README.md):

        - input: "some text"
          expected: "expected output"
          name: optional label
        - input: "..."
          expected: "..."
    """
    path = Path(data_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        raise ValueError(
            f"{path} must contain a YAML list of test cases, got {type(raw).__name__}"
        )

    cases: list[TestCase] = []
    for i, item in enumerate(raw):
        if "input" not in item or "expected" not in item:
            raise ValueError(
                f"{path}: test case #{i} is missing required 'input'/'expected' keys"
            )
        cases.append(
            TestCase(
                input=str(item["input"]),
                expected=str(item["expected"]),
                name=item.get("name"),
            )
        )
    return cases
