"""Unit tests for the safe calculator tool."""

import pytest

from tools.builtins.calculator import Calculator


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3", "5"),
        ("2 * (3 + 4)", "14"),
        ("10 / 4", "2.5"),
        ("10 // 4", "2"),
        ("10 % 3", "1"),
        ("2 ** 10", "1024"),
        ("-5 + 2", "-3"),
        ("8 / 2", "4"),  # float result that is whole renders without .0
    ],
)
def test_evaluates_arithmetic(expression, expected):
    assert Calculator().invoke({"expression": expression}).output == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",  # function call / name -> rejected
        "1 + foo",  # name -> rejected
        "2 & 3",  # bitwise op -> rejected
        "1 +",  # syntax error
        "2 ** 99999",  # exponent guard
    ],
)
def test_rejects_unsafe_or_invalid_expressions(expression):
    result = Calculator().invoke({"expression": expression})
    assert result.ok is False
    assert result.error


def test_division_by_zero_is_graceful():
    result = Calculator().invoke({"expression": "1 / 0"})
    assert result.ok is False
    assert "ZeroDivisionError" in result.error


def test_spec_shape():
    spec = Calculator.to_spec()
    assert spec.name == "calculator"
    assert "expression" in spec.parameters["properties"]
