
from typing import Literal

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

from langfuse_client import langfuse, langfuse_handler, flush_langfuse

from tools import calculator, web_search
from config import settings

# --------------------------------------------------
# Langfuse initialization
# --------------------------------------------------


# --------------------------------------------------
# Structured output models
# --------------------------------------------------

class RouteDecision(BaseModel):
    category: Literal["research", "calculation", "general"]


class MathExpression(BaseModel):
    expression: str

# --------------------------------------------------
# Agent Response
# --------------------------------------------------

class AgentResponse(BaseModel):
    answer: str
    category: Literal["research", "calculation", "general"]
    sources: list[dict[str, str]] = Field(default_factory=list)
    confidence: float


# --------------------------------------------------
# LLM initialization
# --------------------------------------------------

llm = None
router_llm = None
math_parser = None
response_llm = None

if settings.openai_api_key:

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=settings.openai_api_key,
        temperature=0,
        streaming=True
    )

    router_llm = llm.with_structured_output(RouteDecision)
    math_parser = llm.with_structured_output(MathExpression)
    response_llm = llm.with_structured_output(AgentResponse)

# --------------------------------------------------
# Agent State
# --------------------------------------------------

class AgentState(MessagesState):
    query: str
    category: Literal["research", "calculation", "general"]
    result: str
    answer: str
    sources: list[dict[str, str]] = Field(default_factory=list)
    confidence: float


# --------------------------------------------------
# Router node
# --------------------------------------------------

def router_node(state: AgentState):

    if router_llm is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    query = state["query"]

    decision = router_llm.invoke([
        SystemMessage(content="""
        Classify the user's query into one category:

        research: Finding latest information, news,
        research papers or external information.

        calculation: Mathematical calculations.

        general: Explanations, concepts and general questions.
        """),
        HumanMessage(content=query)
    ])

    return {
        "category": decision.category
    }


# --------------------------------------------------
# Research node
# --------------------------------------------------


def research_node(state: AgentState):

    if llm is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    search_results = web_search.invoke({
        "query": state["query"]
    })

    response = llm.invoke([
        SystemMessage(content="""
        You are AetherAI, an AI research assistant.

        Answer the user's question using the supplied
        search results.

        Instructions:
        1. Summarize information in simple language.
        2. Handle multiple questions separately.
        3. Include important numbers and units.
        4. Mention the source when available.
        5. Never invent missing information.
        6. If information is unavailable, say so.
        7. Use clear headings and bullet points.
        """),
        HumanMessage(content=f"""
        User question:
        {state['query']}

        Search results:
        {search_results}
        """)
    ])

    return {
        "messages": [AIMessage(content=response.content)],
        "result": response.content
    }

# --------------------------------------------------
# Calculation node
# --------------------------------------------------

def calculation_node(state: AgentState):

    if math_parser is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    query = state["query"]

    parsed = math_parser.invoke([
        SystemMessage(content="""
        Convert the user's natural language query into
        a mathematical expression.

        Rules:
        1. Return only a valid arithmetic expression.
        2. Support addition, subtraction, multiplication,
           division, modulo and exponentiation.
        3. Understand natural language such as:
           - Multiply 30 and 30.
           - Divide 100 by 5.
           - Subtract 15 from 50.
           - What is 25 percent of 200?
        4. Do not calculate the answer.
        5. Do not generate Python code.
        6. If the request is not a valid mathematical
           calculation, return an empty expression.
        """),
        HumanMessage(content=query)
    ])

    expression = parsed.expression.strip()

    if not expression:
        raise ValueError("Could not interpret mathematical expression")

    result = calculator.invoke({
        "expression": expression
    })

    return {
        "messages": [AIMessage(content=str(result))],
        "result": str(result)
    }


# --------------------------------------------------
# General question node
# --------------------------------------------------

def general_node(state: AgentState):

    if llm is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    response = llm.invoke(state["messages"])

    return {
        "messages": [response],
        "result": response.content
    }


# --------------------------------------------------
# Conditional routing
# --------------------------------------------------

def route_query(state: AgentState):
    return state["category"]


# --------------------------------------------------
# Build LangGraph workflow
# --------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("research", research_node)
workflow.add_node("calculation", calculation_node)
workflow.add_node("general", general_node)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    route_query,
    {
        "research": "research",
        "calculation": "calculation",
        "general": "general"
    }
)

workflow.add_edge("research", END)
workflow.add_edge("calculation", END)
workflow.add_edge("general", END)


# --------------------------------------------------
# Compile graph
# --------------------------------------------------

checkpointer = InMemorySaver()

app = workflow.compile(
    checkpointer=checkpointer
)


# --------------------------------------------------
# Run agent independently
# --------------------------------------------------

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "user_1"
        },
        "callbacks": [langfuse_handler]
    }

    queries = [
        "What's 450 multiplied by 28?",
        "Find recent research papers about RAG",
        "Explain how an AI agent works",
        "What is the latest development in LangGraph?"
    ]

    try:

        for query in queries:

            response = app.invoke(
                {
                    "messages": [HumanMessage(content=query)],
                    "query": query,
                    "category": "",
                    "result": ""
                },
                config=config
            )

            print("\nQuery:", query)
            print("Category:", response["category"])
            print("Result:", response["result"])

    finally:
        flush_langfuse()
