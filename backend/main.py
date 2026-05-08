"""FastAPI エントリポイント"""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from chat import router as chat_router
from escalate import router as escalate_router
from feedback import router as feedback_router

app = FastAPI(title="FAQ Chatbot API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(escalate_router)
app.include_router(feedback_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
