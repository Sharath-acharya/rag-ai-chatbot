from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import FileResponse

from app.config import DOCS_DIR
from app.models import ChatRequest
from app.rag import RAGChatbot

app = FastAPI(title="Free RAG AI Chatbot")
app.mount("/static", StaticFiles(directory="static"), name="static")

chatbot = RAGChatbot(DOCS_DIR)

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        reply = chatbot.ask(request.message)
        return {"reply": reply}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
