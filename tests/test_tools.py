
from tools import calculator, web_search
from unittest.mock import patch


@patch("tools.TavilyClient")
def test_web_search(mock_tavily):
    mock_client = mock_tavily.return_value

    mock_client.search.return_value = {
        "results": [
            {"title": "LangGraph", "url": "https://example.com"}
        ]
    }

    result = web_search.invoke({"query": "LangGraph"})

    assert "LangGraph" in result
    mock_client.search.assert_called_once_with(query="LangGraph")

def test_calculator_addition():
    result = calculator.invoke({"expression": "10 + 20"})
    assert result == "30"


def test_calculator_multiplication():
    result = calculator.invoke({"expression": "5 * 4"})
    assert result == "20"


def test_calculator_invalid_expression():
    result = calculator.invoke({"expression": "5 + "})
    assert result == "Invalid expression"
