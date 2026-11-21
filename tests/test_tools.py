
from tools import calculator


def test_calculator_addition():
    result = calculator.invoke({"expression": "10 + 20"})
    assert result == "30"


def test_calculator_multiplication():
    result = calculator.invoke({"expression": "5 * 4"})
    assert result == "20"


def test_calculator_invalid_expression():
    result = calculator.invoke({"expression": "5 + "})
    assert result == "Invalid expression"
