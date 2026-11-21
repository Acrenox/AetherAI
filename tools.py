from langchain.tools import tool
from tavily import TavilyClient
import os


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        return str(eval(expression))
    except Exception:
        return "Invalid expression"


@tool
def web_search(query: str) -> str:
    """Search the web for current information."""

    client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    response = client.search(query=query)

    results = response.get("results", [])

    return str(results[:5])