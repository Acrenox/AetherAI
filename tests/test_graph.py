
from unittest.mock import MagicMock, patch
from graph import calculation_node, MathExpression

from graph import (
    router_node,
    research_node,
    calculation_node,
    general_node
)


def test_router_node():
    mock_decision = MagicMock()
    mock_decision.category = "research"

    with patch("graph.router_llm", autospec=False) as mock_router:
        mock_router.invoke.return_value = mock_decision

        result = router_node({"query": "Find recent AI papers"})

    assert result["category"] == "research"


def test_research_node():
    with patch("graph.web_search", autospec=False) as mock_search:
        mock_search.invoke.return_value = "Mock search results"

        result = research_node({"query": "Latest AI news"})

    assert result["result"] == "Mock search results"


def test_calculation_node():
    mock_parser = MagicMock()
    mock_parser.invoke.return_value = MathExpression(
        expression="450 * 28"
    )

    with patch("graph.math_parser", mock_parser):
        result = calculation_node({
            "query": "450 multiplied by 28"
        })

    assert result["result"] == "12600"


def test_general_node():
    mock_response = MagicMock()
    mock_response.content = "AI agents perform tasks autonomously"

    with patch("graph.llm", autospec=False) as mock_llm:
        mock_llm.invoke.return_value = mock_response

        result = general_node({"messages": []})

    assert result["result"] == "AI agents perform tasks autonomously"