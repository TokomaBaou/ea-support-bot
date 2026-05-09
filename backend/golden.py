"""ゴールデンアンサー管理

正解回答ペアの保存・検索・few-shot フォーマットを行う。
backend/data/golden_answers.json で永続化する。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict


class GoldenEntry(TypedDict):
    id: str
    question: str
    answer: str
    intent: str
    source: str
    created_at: str


_DATA_PATH = Path(__file__).parent / "data" / "golden_answers.json"


def _load_all() -> list[GoldenEntry]:
    """全ゴールデンアンサーを読み込む。"""
    if not _DATA_PATH.exists():
        return []
    with _DATA_PATH.open("r", encoding="utf-8") as f:
        data: list[GoldenEntry] = json.load(f)
    return data


def _save_all(entries: list[GoldenEntry]) -> None:
    """全ゴールデンアンサーを保存する。"""
    with _DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _bigrams(text: str) -> set[str]:
    """テキストから文字 bigram を生成する。"""
    text = text.replace(" ", "").replace("　", "")
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def search_golden(query: str, top_k: int = 2) -> list[GoldenEntry]:
    """クエリに類似するゴールデンアンサーを bigram 一致で検索する。"""
    entries = _load_all()
    if not entries:
        return []

    query_bg = _bigrams(query)
    if not query_bg:
        return []

    scored: list[tuple[GoldenEntry, float]] = []
    for entry in entries:
        entry_bg = _bigrams(entry["question"])
        overlap = len(query_bg & entry_bg)
        score = overlap / len(query_bg)
        scored.append((entry, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [e for e, s in scored[:top_k] if s > 0.1]


def add_golden(
    question: str,
    answer: str,
    intent: str,
    source: str = "faq",
) -> GoldenEntry:
    """新しいゴールデンアンサーを追加する。"""
    entries = _load_all()
    entry: GoldenEntry = {
        "id": f"ga_{uuid.uuid4().hex[:8]}",
        "question": question,
        "answer": answer,
        "intent": intent,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    entries.append(entry)
    _save_all(entries)
    return entry


def format_few_shot(goldens: list[GoldenEntry]) -> str:
    """ゴールデンアンサーを few-shot 参考テキストにフォーマットする。"""
    if not goldens:
        return ""
    parts: list[str] = []
    for g in goldens:
        parts.append(f"Q: {g['question']}\nA: {g['answer']}")
    return "\n\n".join(parts)
