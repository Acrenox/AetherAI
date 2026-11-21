from typing import TypedDict
from pydantic import BaseModel
from typing import Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from tools import calculator, web_search
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import re
import os

load_dotenv()

langfuse = get_client()
langfuse_handler = CallbackHandler()

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key,
    temperature=0
)

class RouteDecision(BaseModel):
    category: Literal["research", "calculation", "general"]

router_llm = llm.with_structured_output(RouteDecision)

class AgentState(MessagesState):
    query: str
    category: str
    result: str

def router_node(state: AgentState):

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

def research_node(state: AgentState):
    result = web_search.invoke({
        "query": state["query"]
    })

    return {
        "messages": [AIMessage(content=str(result))],
        "result": str(result)
    }

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


def general_node(state: AgentState):
    response = llm.invoke(state["messages"])

    return {
        "messages": [response],
        "result": response.content
    }

def route_query(state: AgentState):
    return state["category"]

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

checkpointer = InMemorySaver()

app = workflow.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": "user_1"
    },
    "callbacks": [langfuse_handler]
}

if __name__ == "__main__":

    queries = [
        "What's 450 multiplied by 28?",
        "Find recent research papers about RAG",
        "Explain how an AI agent works",
        "What is the latest development in LangGraph?"
    ]

    for query in queries:

        response = app.invoke({
            "messages": [HumanMessage(content=query)],
            "query": query,
            "category": "",
            "result": ""
        }, config=config)

        print("\nQuery:", query)
        print("Category:", response["category"])
        print("Result:", response["result"])
    langfuse.flush()