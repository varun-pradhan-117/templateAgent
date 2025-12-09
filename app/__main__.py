import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from .agent import RoomAgent

app = FastAPI()
agent = RoomAgent(model_provider="ollama")

class ChatRequest(BaseModel):
    query: str
    context_id: str | None = None

@app.post("/chat")
def chat_endpoint(payload: ChatRequest):
    result = agent.invoke(
        query=payload.query, 
        context_id=payload.context_id
    )
    return result


if __name__ == "__main__":
    uvicorn.run("app.__main__:app", host="0.0.0.0", port=8000, reload=True)