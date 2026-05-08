"""FastAPI エントリポイント"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from chat import router as chat_router
from escalate import router as escalate_router
from feedback import router as feedback_router

app = FastAPI(title="FAQ Chatbot API", version="0.2.0")

# ── CORS ──
# 環境変数 CORS_ORIGINS にカンマ区切りでオリジンを指定可能。
# 未設定時はローカル開発用のデフォルト値を使用する。
# NOTE: プロトタイプ段階では "*" でも可。
#       本番運用時は Railway / Vercel のドメインのみに絞ること。
_DEFAULT_ORIGINS = (
    "http://localhost:3000,"
    "http://localhost:3001,"
    "http://127.0.0.1:3000,"
    "http://127.0.0.1:3001"
)
_origins = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
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


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
