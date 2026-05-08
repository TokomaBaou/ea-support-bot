"""応答フォーマッタ

チャネルごとに表示形式を切り替える。Web は構造化された JSON を返し、
LINE 用フォーマッタは将来追加する想定。
"""

from __future__ import annotations

from typing import Literal, TypedDict

from faq_search import FaqHit


ChannelType = Literal["web", "line"]


class SourceFaq(TypedDict):
    question: str
    answer: str
    category: str


class FormattedResponse(TypedDict):
    answer: str
    sources: list[SourceFaq]


def format_for_web(answer: str, hits: list[FaqHit]) -> FormattedResponse:
    sources: list[SourceFaq] = [
        {
            "question": hit["entry"]["question"],
            "answer": hit["entry"]["answer"],
            "category": hit["entry"]["category"],
        }
        for hit in hits
    ]
    return {"answer": answer, "sources": sources}


def format_response(
    answer: str, hits: list[FaqHit], channel: ChannelType
) -> FormattedResponse:
    if channel == "web":
        return format_for_web(answer, hits)
    if channel == "line":
        # LINE 用は将来実装。当面は web と同じ形式を返す。
        return format_for_web(answer, hits)
    raise ValueError(f"Unsupported channel type: {channel}")
