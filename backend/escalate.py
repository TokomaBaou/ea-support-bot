"""エスカレーション API

「人に相談する」ボタン押下時に、受講生の質問・直前のBot回答案・受講生IDを
Slack に通知する。
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel, Field

from slack import notify_slack

router = APIRouter()


class EscalateRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=1, max_length=2000)
    bot_answer: str | None = None
    student_id: str | None = None


class EscalateResponse(BaseModel):
    ok: bool
    escalated_at: str
    notified: bool


@router.post("/api/escalate", response_model=EscalateResponse)
def escalate(req: EscalateRequest) -> EscalateResponse:
    now = datetime.now(timezone.utc).isoformat()
    student_label = req.student_id or "（未取得）"
    bot_block = req.bot_answer or "（直前のBot回答なし）"

    payload = {
        "text": "📞 有人サポート希望（受講生からの相談）",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "📞 有人サポート希望"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*セッション*\n{req.session_id}"},
                    {"type": "mrkdwn", "text": f"*受講生ID*\n{student_label}"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*ご質問*\n>{req.question}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*直前のBot回答*\n>{bot_block}"},
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"⏰ 受付: {now}　👋 担当者からの折り返しをお願いします。",
                    }
                ],
            },
        ],
    }
    notified = notify_slack(payload)
    return EscalateResponse(ok=True, escalated_at=now, notified=notified)
