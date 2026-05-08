"""Slack 通知ヘルパー

`SLACK_WEBHOOK_URL` が設定されていれば Incoming Webhook に POST する。
未設定時は警告ログを出して何もしない (ローカルデモを止めない)。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _get_webhook_url() -> str | None:
    return os.environ.get("SLACK_WEBHOOK_URL")


def notify_slack(payload: dict[str, Any]) -> bool:
    url = _get_webhook_url()
    if not url:
        logger.warning(
            "SLACK_WEBHOOK_URL is not set; skipping Slack notification. payload=%s",
            payload.get("text") or list(payload.keys()),
        )
        return False
    try:
        response = httpx.post(url, json=payload, timeout=5.0)
        response.raise_for_status()
        return True
    except httpx.HTTPError as e:
        logger.error("Slack notification failed: %s", e)
        return False
