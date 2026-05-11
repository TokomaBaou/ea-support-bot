"""チャット API

Phase 2: Agentic RAG による意図分類 + ツール選択で回答を生成する。
OpenAI API が利用できない場合は Phase 1 の FAQ 検索フォールバックを使用。
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

from agent import AgentResult, run_agent
from faq_search import FaqHit, SourceName, search, search_auto
from golden import GoldenEntry, add_golden
from patterns import ResponsePattern, load_patterns
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
    """GPT を呼ばずに FAQ 検索結果だけで応答を組み立てる。"""
    if not hits:
        return FALLBACK_NO_HIT_MESSAGE
    top = hits[0]["entry"]
    return top["answer"]


def _is_pilot_mode() -> bool:
    return os.environ.get("PILOT_MODE", "true").lower() in {"1", "true", "yes"}


# ── Phase 1 システムプロンプト（フォールバック時の参考用に保持） ──

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


# ── リクエスト / レスポンスモデル ──


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
    intent: str = "unknown"
    tools_used: list[str] = []
    response_pattern: str = "brief"


# ── ゴールデンアンサーエンドポイント ──


class GoldenAnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    intent: str = "faq_question"
    source: str = "faq"


class GoldenAnswerResponse(BaseModel):
    ok: bool
    entry: dict[str, str]


# ── ヘルパー ──


def _run_fallback(
    message: str, source: str
) -> tuple[str, list[dict[str, str]], str]:
    """Phase 1 フォールバック: FAQ 検索のみで簡易回答。"""
    matched: str
    if source == "auto":
        auto_result = search_auto(message, top_k=3)
        hits = auto_result["hits"]
        matched = auto_result["matched_source"]
    else:
        hits = search(message, top_k=3, source=source)  # type: ignore[arg-type]
        matched = source
    answer = _fallback_answer(hits)
    sources: list[dict[str, str]] = [
        {
            "question": h["entry"]["question"],
            "answer": h["entry"]["answer"],
            "category": h["entry"]["category"],
        }
        for h in hits
    ]
    return answer, sources, matched


def _notify_negative_slack(query: str, intent: str, bot_answer: str) -> None:
    """ネガティブ意図検知時にSlackへ通知する。"""
    notify_slack(
        {
            "text": "🚨 ネガティブ検知",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🚨 ネガティブ検知",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*受講生のメッセージ*\n>{query}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*分類*\n{intent}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": "*推奨対応*\n担当者による個別フォロー",
                        },
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Bot回答*\n>{bot_answer}",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "⚠️ 早急に担当者からフォローをお願いします。",
                        }
                    ],
                },
            ],
        }
    )


def _notify_pilot_slack(query: str, answer: str, session_id: str) -> None:
    """パイロットモード: Bot の回答案を担当者にSlackで確認してもらう。"""
    notify_slack(
        {
            "text": "🤖 FAQボット 回答案の確認依頼",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🤖 FAQボット 回答案の確認",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*セッション*\n{session_id}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": "*モード*\n通知のみ (パイロット)",
                        },
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*受講生のご質問*\n>{query}",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Bot 回答案*\n>{answer}",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                "📝 内容を確認のうえ、"
                                "担当者から正式な回答をお送りください。"
                            ),
                        }
                    ],
                },
            ],
        }
    )


# ── メインチャットエンドポイント ──


@router.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    message_id = f"msg_{uuid.uuid4().hex[:12]}"

    client = _get_client_optional()

    # デフォルトメタデータ
    intent = "unknown"
    tools_used: list[str] = []
    response_pattern_id = "brief"
    answer_text: str
    hit_sources: list[dict[str, str]]
    matched_source: str

    if client is not None:
        try:
            # Phase 2: Agentic RAG
            agent_result: AgentResult = run_agent(
                client=client,
                message=req.message,
                detail_level=req.detail_level,
                source=req.source,
            )
            answer_text = agent_result["answer"]
            hit_sources = agent_result["sources"]
            matched_source = agent_result["matched_source"]
            intent = agent_result["intent"]
            tools_used = agent_result["tools_used"]
            response_pattern_id = agent_result["response_pattern"]
        except openai.AuthenticationError as e:
            logger.warning(
                "OpenAI authentication failed (%s); using fallback.", e
            )
            answer_text, hit_sources, matched_source = _run_fallback(
                req.message, req.source
            )
        except openai.APIError as e:
            raise HTTPException(
                status_code=502, detail=f"OpenAI API error: {e}"
            )
        except Exception as e:
            logger.error(
                "Agent error: %s; using fallback.", e, exc_info=True
            )
            answer_text, hit_sources, matched_source = _run_fallback(
                req.message, req.source
            )
    else:
        logger.warning(
            "OPENAI_API_KEY not configured; using FAQ-only fallback."
        )
        answer_text, hit_sources, matched_source = _run_fallback(
            req.message, req.source
        )

    if intent == "negative":
        _notify_negative_slack(req.message, intent, answer_text)

    pilot_mode = _is_pilot_mode()
    if pilot_mode:
        _notify_pilot_slack(req.message, answer_text, session_id)

    return ChatResponse(
        answer=answer_text,
        sources=[SourceFaq(**s) for s in hit_sources],
        session_id=session_id,
        message_id=message_id,
        pilot_mode=pilot_mode,
        original_query=req.message,
        matched_source=matched_source,
        intent=intent,
        tools_used=tools_used,
        response_pattern=response_pattern_id,
    )


# ── ゴールデンアンサーエンドポイント ──


@router.post("/api/golden-answer", response_model=GoldenAnswerResponse)
def add_golden_answer_endpoint(req: GoldenAnswerRequest) -> GoldenAnswerResponse:
    """正解回答ペアを追加する。"""
    entry: GoldenEntry = add_golden(
        question=req.question,
        answer=req.answer,
        intent=req.intent,
        source=req.source,
    )
    return GoldenAnswerResponse(ok=True, entry=dict(entry))


# ── パターン一覧エンドポイント ──


@router.get("/api/patterns")
def list_patterns_endpoint() -> list[ResponsePattern]:
    """全回答パターンを返す。"""
    return load_patterns()
