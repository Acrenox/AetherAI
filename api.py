
from uuid import uuid4

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Header,
    Request
)
from pathlib import Path
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from langfuse_client import langfuse, langfuse_handler

from graph import app as agent_app
from config import settings

import json
from fastapi.responses import StreamingResponse


# Rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="AetherAI",
    description="AI Research Agent API",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


class QueryRequest(BaseModel):
    query: str


# API authentication
def check_key(x_api_key: str = Header()):
    if not settings.api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )


@app.get("/")
def home():
    return FileResponse(
        Path(__file__).parent / "static" / "index.html"
    )


@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/ask", dependencies=[Depends(check_key)])
@limiter.limit("10/minute")
def ask_agent(request: Request, payload: QueryRequest):

    with langfuse.start_as_current_observation(
        name="aetherai-api-request",
        as_type="span",
        input={"query": payload.query},
        metadata={"endpoint": "/ask"}
    ) as trace:

        try:
            response = agent_app.invoke(
                {
                    "query": payload.query,
                    "messages": [HumanMessage(content=payload.query)]
                },
                config={
                    "configurable": {
                        "thread_id": str(uuid4())
                    },
                    "callbacks": [langfuse_handler]
                }
            )

            result = {
                "query": payload.query,
                "category": response.get("category"),
                "result": response.get("result")
            }

            trace.update(output=result)

            return result

        except Exception as exc:
            trace.update(
                level="ERROR",
                status_message=str(exc)
            )
            raise HTTPException(
                status_code=500,
                detail="Agent execution failed"
            )


@app.post("/ask/stream", dependencies=[Depends(check_key)])
@limiter.limit("10/minute")
async def stream_agent(request: Request, payload: QueryRequest):

    async def generate():

        thread_id = str(uuid4())
        final_state = {}

        with langfuse.start_as_current_observation(
            name="aetherai-stream-request",
            as_type="span",
            input={"query": payload.query},
            metadata={"endpoint": "/ask/stream"}
        ) as trace:

            try:
                config = {
                    "configurable": {
                        "thread_id": thread_id
                    },
                    "callbacks": [langfuse_handler]
                }

                async for mode, chunk in agent_app.astream(
                    {
                        "query": payload.query,
                        "messages": [
                            HumanMessage(content=payload.query)
                        ]
                    },
                    config=config,
                    stream_mode=["messages", "values"]
                ):

                    # Capture the latest workflow state
                    if mode == "values":
                        final_state = chunk

                    # Stream only actual answer tokens
                    elif mode == "messages":

                        message, metadata = chunk
                        node = metadata.get("langgraph_node")

                        if node in ["research", "general"]:
                            content = message.content

                            if isinstance(content, str) and content:
                                yield (
                                    f"data: {json.dumps({
                                        'type': 'token',
                                        'content': content
                                    })}\n\n"
                                )

                result = {
                    "query": payload.query,
                    "category": final_state.get("category"),
                    "result": final_state.get("result")
                }

                trace.update(output=result)

                # Send final workflow result
                yield (
                    f"data: {json.dumps({
                        'type': 'done',
                        'data': result
                    })}\n\n"
                )

            except Exception as exc:

                trace.update(
                    level="ERROR",
                    status_message=str(exc)
                )

                yield (
                    f"data: {json.dumps({
                        'type': 'error',
                        'message': 'Agent execution failed'
                    })}\n\n"
                )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
