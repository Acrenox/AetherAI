from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from graph import app, MathExpression


def test_calculation_workflow():

    mock_decision = MagicMock()
    mock_decision.category = "calculation"

    mock_parser = MagicMock()
    mock_parser.invoke.return_value = MathExpression(
        expression="450 * 28"
    )

    with patch("graph.router_llm") as mock_router:
        with patch("graph.math_parser", mock_parser):

            mock_router.invoke.return_value = mock_decision

            response = app.invoke(
                {
                    "messages": [
                        HumanMessage(content="450 multiplied by 28")
                    ],
                    "query": "450 multiplied by 28",
                    "category": "",
                    "result": ""
                },
                config={
                    "configurable": {
                        "thread_id": "test_workflow"
                    }
                }
            )

    assert response["result"] == "12600"
    assert response["category"] == "calculation"

def test_research_workflow():

    mock_decision = MagicMock()
    mock_decision.category = "research"

    mock_response = MagicMock()
    mock_response.content = "Mock synthesized research answer"

    with patch("graph.router_llm", autospec=False) as mock_router, \
         patch("graph.web_search", autospec=False) as mock_search, \
         patch("graph.llm", autospec=False) as mock_llm:

        mock_router.invoke.return_value = mock_decision
        mock_search.invoke.return_value = "Mock research results"
        mock_llm.invoke.return_value = mock_response

        response = app.invoke(
            {
                "messages": [HumanMessage(content="Find recent RAG papers")],
                "query": "Find recent RAG papers",
                "category": "",
                "result": ""
            },
            config={"configurable": {"thread_id": "test_research"}}
        )

    assert response["category"] == "research"
    assert response["result"] == "Mock synthesized research answer"

def test_general_workflow():

    mock_decision = MagicMock()
    mock_decision.category = "general"

    mock_response = AIMessage(
        content="An AI agent performs tasks autonomously."
    )

    with patch("graph.router_llm", autospec=False) as mock_router, \
         patch("graph.llm", autospec=False) as mock_llm:

        mock_router.invoke.return_value = mock_decision
        mock_llm.invoke.return_value = mock_response

        response = app.invoke(
            {
                "messages": [
                    HumanMessage(content="What is an AI agent?")
                ],
                "query": "What is an AI agent?",
                "category": "",
                "result": ""
            },
            config={
                "configurable": {
                    "thread_id": "test_general"
                }
            }
        )

    assert response["category"] == "general"
    assert response["result"] == "An AI agent performs tasks autonomously."