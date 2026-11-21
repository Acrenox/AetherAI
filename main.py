from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from rich import print
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langfuse import get_client
from langfuse.langchain import CallbackHandler
import os

from tools import calculator, web_search

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

langfuse = get_client()

langfuse_handler = CallbackHandler()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0,
    streaming=True
)

checkpointer = InMemorySaver()

agent = create_agent(
    model = llm,
    tools=[calculator, web_search],
    system_prompt="You are a helpful AI assistant. Use the calculator when mathematical calculations are needed.",
    checkpointer=checkpointer
)

config = {
    "configurable":{
        "thread_id": "user_1"
    }
}

while True:
    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:
        break

    response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": query}
        ]
    },
    config={
        **config,
        "callbacks": [langfuse_handler]
    }
    )

    print("\nAgent:", response["messages"][-1].content)

    langfuse.flush()