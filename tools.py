from langchain.tools import tool
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import ast
import operator



@tool
def calculator(expression: str) -> str:
    """Safely calculate basic mathematical expressions."""

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    def calculate(node):
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return node.value
            raise ValueError("Invalid value")

        if isinstance(node, ast.BinOp):
            op = operators.get(type(node.op))

            if op is None:
                raise ValueError("Unsupported operator")

            if isinstance(node.op, ast.Pow):
                exponent = calculate(node.right)
                if abs(exponent) > 100:
                    raise ValueError("Exponent too large")
                left = calculate(node.left)
                return op(left, exponent)

            return op(calculate(node.left), calculate(node.right))

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -calculate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return calculate(node.operand)

        raise ValueError("Invalid expression")

    try:
        if len(expression) > 100:
            raise ValueError("Expression too long")

        tree = ast.parse(expression, mode="eval")
        return str(calculate(tree.body))

    except Exception:
        return "Invalid expression"

@tool
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    reraise=True
)
def web_search(query: str) -> str:
    """Search the web for current information."""

    client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    response = client.search(query=query)

    results = response.get("results", [])

    return str(results[:5])