
from typing import Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
from tools import calculator, web_search
import os
import re

load_dotenv()

# Langfuse initialization
langfuse = get_client()
langfuse_handler = CallbackHandler()

# API configuration
api_key = os.getenv("OPENAI_API_KEY")

# Router response structure
class RouteDecision(BaseModel):
    category: Literal["research", "calculation", "general"]


# Initialize LLM only when API key is available
llm = None
router_llm = None

if api_key:
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=api_key,
        temperature=0
    )

    router_llm = llm.with_structured_output(RouteDecision)


# Agent state
class AgentState(MessagesState):
    query: str
    category: str
    result: str


# Router node
def router_node(state: AgentState):

    if router_llm is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    query = state["query"]

    decision = router_llm.invoke([
        {
            "role": "system",
            "content": """
            Classify the user's query into one category:

            research: Finding latest information, news,
            research papers or external information.

            calculation: Mathematical calculations.

            general: Explanations, concepts and general questions.
            """
        },
        {
            "role": "user",
            "content": query
        }
    ])

    return {"category": decision.category}


# Research node
def research_node(state: AgentState):

    result = web_search.invoke({
        "query": state["query"]
    })

    return {
        "messages": [AIMessage(content=str(result))],
        "result": str(result)
    }


# Calculation node
def calculation_node(state: AgentState):

    expression = state["query"].lower()

    replacements = {
        "multiplied by": "*",
        "times": "*",
        "divided by": "/",
        "plus": "+",
        "minus": "-"
    }

    for word, operator in replacements.items():
        expression = expression.replace(word, operator)

    expression = re.sub(r"[^0-9+\-*/(). ]", "", expression)

    result = calculator.invoke({
        "expression": expression
    })

    return {
        "messages": [AIMessage(content=str(result))],
        "result": str(result)
    }


# General question node
def general_node(state: AgentState):

    if llm is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    response = llm.invoke(state["messages"])

    return {
        "messages": [response],
        "result": response.content
    }


# Conditional routing
def route_query(state: AgentState):
    return state["category"]


# Build LangGraph workflow
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

# Compile graph
checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# Run agent
if __name__ == "__main__":

    if not api_key:
        raise RuntimeError("Please configure OPENAI_API_KEY in your .env file")

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

    langfuse.flush()
