"""回答パターン DB

backend/data/response_patterns.json を読み込み、
意図に応じたデフォルトパターンを提供する。

パターン:
  - brief: 短文簡易回答（FAQ質問・あいさつ）
  - detailed_feedback: 詳細フィードバック（やり方質問）
  - encouragement: 応援・承認（体験報告）
  - empathy: 共感相談（悩み相談）
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict


class ResponsePattern(TypedDict):
    id: str
    name: str
    description: str
    system_instruction: str
    default_for: list[str]


_DATA_PATH = Path(__file__).parent / "data" / "response_patterns.json"


@lru_cache(maxsize=1)
def load_patterns() -> list[ResponsePattern]:
    """全パターンを読み込む。"""
    if not _DATA_PATH.exists():
        return []
    with _DATA_PATH.open("r", encoding="utf-8") as f:
        data: list[ResponsePattern] = json.load(f)
    return data


def get_pattern(pattern_id: str) -> ResponsePattern | None:
    """ID でパターンを取得する。"""
    for p in load_patterns():
        if p["id"] == pattern_id:
            return p
    return None


@lru_cache(maxsize=1)
def _intent_to_pattern_map() -> dict[str, str]:
    """パターンデータから 意図→パターンID のマッピングを構築する。"""
    mapping: dict[str, str] = {}
    for p in load_patterns():
        for intent in p.get("default_for", []):
            mapping[intent] = p["id"]
    return mapping


def get_default_pattern_for_intent(intent: str) -> ResponsePattern | None:
    """意図に対応するデフォルトパターンを返す。"""
    pattern_id = _intent_to_pattern_map().get(intent)
    if pattern_id is None:
        return None
    return get_pattern(pattern_id)
