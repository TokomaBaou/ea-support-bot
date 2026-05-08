"""フィードバック API

回答に対する👍/👎評価を記録する。プロトタイプではログ出力のみ。
👎 のみ Slack にも通知し、改善対象として担当者の目に触れるようにする。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from slack import notify_slack

router = APIRouter()
logger = logging.getLogger(__name__)


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: Literal["up", "down"]
    comment: str | None = None


class FeedbackResponse(BaseModel):
    ok: bool
    received_at: str


@router.post("/api/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    now = datetime.now(timezone.utc).isoformat()
    logger.info(
        "feedback received: session=%s message=%s rating=%s comment=%r",
        req.session_id,
        req.message_id,
        req.rating,
        req.comment,
    )

    if req.rating == "down":
        comment_line = f"\nコメント: {req.comment}" if req.comment else ""
        notify_slack(
            {
                "text": (
                    "👎 ネガティブ評価を受信\n"
                    f"session: `{req.session_id}` / message: `{req.message_id}`"
                    f"{comment_line}"
                )
            }
        )

    return FeedbackResponse(ok=True, received_at=now)
