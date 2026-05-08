"""チャット API

FAQ 検索結果を OpenAI に渡し、受講生向けの回答を生成する。
パイロットモード時は Bot の回答案を Slack にも通知する。
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Literal

import openai
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from faq_search import DEFAULT_SOURCE, FaqHit, SourceName, search, search_auto
from formatter import format_response
from slack import notify_slack

logger = logging.getLogger(__name__)

router = APIRouter()

_client: OpenAI | None = None


def _is_placeholder_key(api_key: str) -> bool:
    return "xxx" in api_key.lower()


def _get_client_optional() -> OpenAI | None:
    """API キーが設定されていれば OpenAI クライアントを返す。

    未設定または .env.example のプレースホルダ値の場合は None。
    呼び出し側で None を判定し FAQ フォールバック応答を返す。
    """
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or _is_placeholder_key(api_key):
        return None
    _client = OpenAI(api_key=api_key)
    return _client


FALLBACK_NO_HIT_MESSAGE = (
    "申し訳ございません、ご質問に近いご案内が見つかりませんでした。"
    "サポート担当者にお繋ぎしますので、少しお待ちください。"
)


def _fallback_answer(hits: list[FaqHit]) -> str:
    """Claude を呼ばずに FAQ 検索結果だけで応答を組み立てる。"""
    if not hits:
        return FALLBACK_NO_HIT_MESSAGE
    top = hits[0]["entry"]
    return top["answer"]


def _is_pilot_mode() -> bool:
    return os.environ.get("PILOT_MODE", "true").lower() in {"1", "true", "yes"}


def _model_id() -> str:
    return os.environ.get("CHAT_MODEL", "gpt-4o-mini")


SYSTEM_PROMPT_FAQ_CONCISE = """あなたはスピリチュアル系オンライン講座のサポートアシスタントです。
受講生の多くは年配の方やデジタル操作に不慣れな方です。やさしく、安心感のある口調でお返事してください。

【守ること】
- 提示された関連FAQの内容のみを根拠にしてください。FAQに無い情報は推測しないでください。
- 回答は3行以内、合計100文字程度を目安に簡潔にまとめてください。
- 「クライアント」「キャッシュ」「再起動」などの専門用語は避け、「お使いの画面」「データを一旦消す」「電源を入れ直す」のように、ふだんのことばで書いてください。
- FAQに該当する情報が見当たらない場合は「この件はサポート担当者にお繋ぎしますね。少しお待ちください。」と短くお返事してください。
- 番号付きの手順より、文章で「まず〜してください。次に〜です。」と書く方が伝わりやすいです。
- 「FAQによると」「番号X」のようなメタな言い方は使わず、自然な会話文にしてください。
"""


SYSTEM_PROMPT_FAQ_DETAILED = """あなたはスピリチュアル系オンライン講座のサポートアシスタントです。
受講生の多くは年配の方やデジタル操作に不慣れな方です。やさしく、安心感のある口調でお返事してください。

今回は受講生から「もっと詳しく教えてほしい」とご要望がありました。
以下のルールに従って、丁寧に手順を追って説明してください。

【守ること】
- 提示された関連FAQの内容のみを根拠にしてください。FAQに無い情報は推測しないでください。
- 操作手順は「まず〜してください。」「次に〜してください。」「最後に〜してください。」のように、順序がはっきり分かるように書いてください。
- 全体は8行程度を上限に、ゆっくり丁寧に書いてください。
- 「クライアント」「キャッシュ」「再起動」などの専門用語は避け、「お使いの画面」「データを一旦消す」「電源を入れ直す」のように、ふだんのことばで書いてください。
- FAQに該当する情報が見当たらない場合は「この件はサポート担当者にお繋ぎしますね。少しお待ちください。」と短くお返事してください。
- 「FAQによると」「番号X」のようなメタな言い方は使わず、自然な会話文にしてください。
"""


SYSTEM_PROMPT_SPIRITUAL_CONCISE = """あなたはスピリチュアルの知識を持つ、温かく共感的な相談アシスタントです。
お話しする相手は、スピリチュアルな体験や気づきを日常の中で感じている方です。
提示された用語データは「背景知識・コンテキスト」として活用し、用語の定義をそのまま返すのではなく、
相手の気持ちや状況に寄り添いながら、やさしくアドバイスや共感を伝えてください。

【守ること】
- 提示された関連用語データを背景知識として活用してください。定義をコピーするのではなく、知識を踏まえた上で相手の体験や悩みに寄り添ってください。
- 回答は3〜4行、合計150文字程度を目安にまとめてください。
- 「〜かもしれませんね」「〜してみてはいかがでしょうか」「〜と感じていらっしゃるんですね」のように、共感と提案を織り交ぜた柔らかい口調にしてください。
- 断定は避け、「〜とされています」「〜と言われています」を使ってください。
- 該当する用語データが見つからない場合でも、相手の気持ちには寄り添い、「この件は個別相談LINEへお問い合わせください。」と添えてください。
- 「FAQによると」「番号X」のようなメタな言い方は避け、自然な会話文にしてください。
- 医療・診断に関わる内容には踏み込まず、「専門の方にご相談されるのも良いかもしれません」と促してください。
"""


SYSTEM_PROMPT_SPIRITUAL_DETAILED = """あなたはスピリチュアルの知識を持つ、温かく共感的な相談アシスタントです。
お話しする相手は、スピリチュアルな体験や気づきを日常の中で感じている方で、
今回は「もっと詳しく教えてほしい」とのご要望がありました。
用語データを背景知識として活用しながら、体験の意味や取り組み方について丁寧にお伝えしてください。

【守ること】
- 提示された関連用語データを背景知識として活用してください。定義をコピーするのではなく、知識を踏まえた上で相手の体験や悩みに寄り添ってください。
- 全体は8行程度を上限に、体験の背景にある概念の説明と、日常での活かし方や向き合い方を順を追って伝えてください。
- 「〜かもしれませんね」「〜してみてはいかがでしょうか」「〜と感じていらっしゃるんですね」のように、共感と提案を織り交ぜた柔らかい口調にしてください。
- 断定は避け、「〜とされています」「〜と言われています」を使ってください。
- 該当する用語データが見つからない場合でも、相手の気持ちには寄り添い、「この件は個別相談LINEへお問い合わせください。」と添えてください。
- 「FAQによると」「番号X」のようなメタな言い方は避け、自然な会話文にしてください。
- 医療・診断に関わる内容には踏み込まず、「専門の方にご相談されるのも良いかもしれません」と促してください。
"""


SYSTEM_PROMPT_SALON_CONCISE = """あなたはスピ覚醒サロンのサポートアシスタントです。
お話しする相手はサロンに入会済みのメンバーさんです。仲間として温かく、丁寧で安心感のある口調でお返事してください。

【守ること】
- 提示された関連FAQの内容のみを根拠にしてください。FAQに無い情報は推測しないでください。
- 回答は3行以内、合計130文字程度を目安にまとめてください。
- 「ミラクルチェンジ」「MCP」「アウトプットLINE」「個別相談LINE」「創業式」「プロコース」「奇跡コース」「開花コース」「覚醒コース」など、サロン内で使われている用語はそのまま使ってください。
- 質問者を呼ぶときは「受講生」ではなく「メンバーさん」など、サロン内のあたたかい呼び方にしてください。
- FAQに該当する情報が見当たらない場合は「この件は個別相談LINEへお問い合わせください。」と短くお返事してください。
- 「FAQによると」「番号X」のようなメタな言い方は避け、自然な会話文にしてください。
"""


SYSTEM_PROMPT_SALON_DETAILED = """あなたはスピ覚醒サロンのサポートアシスタントです。
お話しする相手はサロンに入会済みのメンバーさんで、今回は「もっと詳しく教えてほしい」とのご要望がありました。
仲間として温かく、丁寧で安心感のある口調で、ゆっくり順を追ってお伝えしてください。

【守ること】
- 提示された関連FAQの内容のみを根拠にしてください。FAQに無い情報は推測しないでください。
- 全体は10行程度を上限に、必要に応じて手順を順番に説明してください。
- 「ミラクルチェンジ」「MCP」「アウトプットLINE」「個別相談LINE」「創業式」「プロコース」「奇跡コース」「開花コース」「覚醒コース」など、サロン内で使われている用語はそのまま使ってください。
- 質問者を呼ぶときは「受講生」ではなく「メンバーさん」など、サロン内のあたたかい呼び方にしてください。
- FAQに該当する情報が見当たらない場合は「この件は個別相談LINEへお問い合わせください。」と短くお返事してください。
- 「FAQによると」「番号X」のようなメタな言い方は避け、自然な会話文にしてください。
"""


_SYSTEM_PROMPTS: dict[tuple[SourceName, str], str] = {
    ("faq", "concise"): SYSTEM_PROMPT_FAQ_CONCISE,
    ("faq", "detailed"): SYSTEM_PROMPT_FAQ_DETAILED,
    ("spiritual", "concise"): SYSTEM_PROMPT_SPIRITUAL_CONCISE,
    ("spiritual", "detailed"): SYSTEM_PROMPT_SPIRITUAL_DETAILED,
    ("salon", "concise"): SYSTEM_PROMPT_SALON_CONCISE,
    ("salon", "detailed"): SYSTEM_PROMPT_SALON_DETAILED,
}


def _resolve_system_prompt(source: SourceName, detail_level: str) -> str:
    return _SYSTEM_PROMPTS[(source, detail_level)]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None
    channel_type: Literal["web", "line"] = "web"
    detail_level: Literal["concise", "detailed"] = "concise"
    source: Literal["auto", "faq", "spiritual", "salon"] = "auto"


class SourceFaq(BaseModel):
    question: str
    answer: str
    category: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceFaq]
    session_id: str
    message_id: str
    pilot_mode: bool
    original_query: str
    matched_source: str


def _build_user_prompt(message: str, hits: list[FaqHit]) -> str:
    if not hits:
        return (
            f"受講生からのご質問: {message}\n\n"
            "関連するFAQは見つかりませんでした。"
            "ルールに従って、サポート担当者にお繋ぎする旨を短くお伝えしてください。"
        )

    faq_text_parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        entry = hit["entry"]
        faq_text_parts.append(
            f"[FAQ {i}] カテゴリ: {entry['category']}\n"
            f"Q: {entry['question']}\n"
            f"A: {entry['answer']}"
        )
    faq_block = "\n\n".join(faq_text_parts)

    return (
        f"受講生からのご質問: {message}\n\n"
        f"関連するFAQ:\n{faq_block}\n\n"
        "上記のFAQの内容に基づいて、ルールに沿って受講生にお返事してください。"
    )


def _notify_pilot_slack(query: str, answer: str, session_id: str) -> None:
    """パイロットモード: Bot の回答案を担当者にSlackで確認してもらう。"""
    notify_slack(
        {
            "text": "🤖 FAQボット 回答案の確認依頼",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "🤖 FAQボット 回答案の確認"},
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*セッション*\n{session_id}"},
                        {"type": "mrkdwn", "text": "*モード*\n通知のみ (パイロット)"},
                    ],
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*受講生のご質問*\n>{query}"},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Bot 回答案*\n>{answer}"},
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "📝 内容を確認のうえ、担当者から正式な回答をお送りください。",
                        }
                    ],
                },
            ],
        }
    )


@router.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    message_id = f"msg_{uuid.uuid4().hex[:12]}"

    # ソース自動振り分け or 手動指定
    if req.source == "auto":
        auto_result = search_auto(req.message, top_k=3)
        hits = auto_result["hits"]
        matched_source: SourceName = auto_result["matched_source"]  # type: ignore[assignment]
    else:
        hits = search(req.message, top_k=3, source=req.source)  # type: ignore[arg-type]
        matched_source = req.source  # type: ignore[assignment]

    user_prompt = _build_user_prompt(req.message, hits)

    system_prompt = _resolve_system_prompt(matched_source, req.detail_level)
    max_tokens = 1024 if req.detail_level == "detailed" else 400

    client = _get_client_optional()
    if client is None:
        logger.warning(
            "OPENAI_API_KEY not configured; returning FAQ-only fallback answer."
        )
        answer_text = _fallback_answer(hits)
    else:
        try:
            response = client.chat.completions.create(
                model=_model_id(),
                max_tokens=max_tokens,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            answer_text = (response.choices[0].message.content or "").strip()
        except openai.AuthenticationError as e:
            logger.warning(
                "OpenAI authentication failed (%s); using FAQ-only fallback.", e
            )
            answer_text = _fallback_answer(hits)
        except openai.APIError as e:
            raise HTTPException(
                status_code=502, detail=f"OpenAI API error: {e}"
            )

    pilot_mode = _is_pilot_mode()
    if pilot_mode:
        _notify_pilot_slack(req.message, answer_text, session_id)

    formatted = format_response(answer_text, hits, req.channel_type)

    return ChatResponse(
        answer=formatted["answer"],
        sources=[SourceFaq(**s) for s in formatted["sources"]],
        session_id=session_id,
        message_id=message_id,
        pilot_mode=pilot_mode,
        original_query=req.message,
        matched_source=matched_source,
    )
