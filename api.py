
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from graph import app as agent_app
from config import settings


app = FastAPI(
    title="AetherAI",
    description="AI Research Agent API",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    query: str


# API key authentication
def check_key(x_api_key: str = Header()):
    if not settings.api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return {"message": "Welcome to AetherAI"}


# Protected agent endpoint
@app.post("/ask", dependencies=[Depends(check_key)])
def ask_agent(request: QueryRequest):
    try:
        response = agent_app.invoke(
            {
                "query": request.query,
                "messages": [
                    HumanMessage(content=request.query)
                ]
            },
            config={
                "configurable": {
                    "thread_id": str(uuid4())
                }
            }
        )

        return {
            "query": request.query,
            "category": response.get("category"),
            "result": response.get("result")
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
