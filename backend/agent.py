"""Agentic RAG コア

OpenAI Function Calling を使って意図分類 + ツール選択を行い、
コンテキストに応じた回答を自律的に生成する。

フロー:
  1. ユーザーメッセージ + ツール定義 + スピ用語背景知識をGPTに渡す
  2. GPT が意図を分類し、必要なツールを呼び出す
  3. ツール結果を GPT に返し、最終回答を生成する
"""

from __future__ import annotations

import json
import logging
import os
from typing import TypedDict

from openai import OpenAI

from faq_search import SourceName, load_faqs, search, search_auto
from golden import search_golden
from patterns import get_pattern

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 型定義
# ---------------------------------------------------------------------------
class AgentResult(TypedDict):
    answer: str
    sources: list[dict[str, str]]
    matched_source: str
    intent: str
    tools_used: list[str]
    response_pattern: str


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
MAX_TOOL_ROUNDS = 3


def _model_id() -> str:
    return os.environ.get("CHAT_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# ツール定義 (OpenAI Function Calling)
# ---------------------------------------------------------------------------
AGENT_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": (
                "FAQ・サロンFAQデータベースを検索して関連する質問と回答を取得する。"
                "講座の手続き、技術的な質問、サロンの利用方法などの回答に必要。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索クエリ（ユーザーの質問をそのまま渡す）",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["auto", "faq", "salon"],
                        "description": (
                            "検索ソース。auto=最適なソースを自動判定。"
                            "デフォルトは auto。"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_terms",
            "description": (
                "スピリチュアル用語の詳細な定義を取得する。"
                "システムプロンプトの背景知識だけでは不足する場合に使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "検索する用語名のリスト"
                            "（例: [\"チャクラ\", \"ツインレイ\"]）"
                        ),
                    },
                },
                "required": ["terms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_response_pattern",
            "description": (
                "回答トーン・形式の指示を取得する。意図に応じた適切なパターンを選択すること。"
                "brief=簡潔回答, detailed_feedback=詳細手順, "
                "encouragement=応援, empathy=共感相談"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern_id": {
                        "type": "string",
                        "enum": [
                            "brief",
                            "detailed_feedback",
                            "encouragement",
                            "empathy",
                        ],
                        "description": "回答パターンID",
                    },
                },
                "required": ["pattern_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_golden_answers",
            "description": (
                "過去に担当者が作成した正解回答例を検索する。"
                "類似の質問への回答を few-shot 参考として取得する。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索クエリ",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# スピリチュアル用語の背景知識（システムプロンプトに常駐）
# ---------------------------------------------------------------------------
def _build_spiritual_terms_block() -> str:
    """spiritual_faq.json から全用語を読み込み、背景知識テキストを生成する。"""
    faqs = load_faqs("spiritual")
    lines: list[str] = []
    for entry in faqs:
        # "アカシックレコードとは？" → "アカシックレコード"
        term = entry["question"].replace("とは？", "").replace("（クンダリニー）", "")
        lines.append(f"- {term}: {entry['answer']}")
    return "\n".join(lines)


def _build_agent_system_prompt() -> str:
    """エージェント用システムプロンプトを構築する。"""
    spiritual_block = _build_spiritual_terms_block()
    return f"""あなたはスピリチュアル系オンライン講座・サロンのサポートアシスタントです。
受講生の多くは年配の方やデジタル操作に不慣れな方です。やさしく、安心感のある口調でお返事してください。

## あなたの役割
受講生からのメッセージを受け取り、適切なツールを使って情報を集め、最適な回答を生成してください。

## 意図分類
メッセージを受け取ったら、まず以下のカテゴリに分類し、適切なツールを呼び出してください：
- faq_question: 講座・サロン・技術的な質問（例: 「再受講はできますか？」「動画が見れない」）
- consultation: 悩みや不安の相談（例: 「チャクラが詰まってる気がして…」「浄化中に体調が悪い」）
- experience_report: 体験や気づきの報告（例: 「エゴを出し切る概念に衝撃！」「瞑想で光が見えた」）
- method_question: やり方・手順の質問（例: 「ミラクルチェンジのやり方を教えて」「グラウンディングの方法」）
- greeting: あいさつ（例: 「こんにちは」「はじめまして」）

## ツール使用ガイド
意図に応じて以下のツールを呼び出してください：
- faq_question → search_faq(query) + get_response_pattern("brief")
- consultation → search_faq(query, source="auto") + get_response_pattern("empathy") + 必要に応じて search_golden_answers(query)
- experience_report → get_response_pattern("encouragement") + 必要に応じて lookup_terms
- method_question → search_faq(query, source="auto") + get_response_pattern("detailed_feedback")
- greeting → get_response_pattern("brief") のみ（ツール不要、直接応答可）

## スピリチュアル用語（背景知識）
以下の用語は回答時の背景知識として活用してください。定義をそのまま返すのではなく、
相手の文脈に合わせて知識を活かしてください。

{spiritual_block}

## 回答ルール
- 「FAQによると」「番号X」のようなメタな言い方は避け、自然な会話文にしてください
- 「クライアント」「キャッシュ」「再起動」などの専門用語は避け、やさしい言葉で説明してください
- 断定は避け、「〜とされています」「〜かもしれません」を使ってください
- 医療・診断に関わる内容には踏み込まず、「専門の方にご相談されるのも良いかもしれません」と促してください
- FAQに該当する情報が見つからない場合は「サポート担当者にお繋ぎしますね」とお伝えください
- サロンメンバーには「メンバーさん」と呼びかけてください
- get_response_pattern で取得した回答パターンの指示に従ってください"""


# ---------------------------------------------------------------------------
# ツール実行
# ---------------------------------------------------------------------------
def _execute_tool(name: str, args: dict[str, object]) -> dict[str, object]:
    """ツールを実行して結果を返す。"""
    if name == "search_faq":
        return _tool_search_faq(args)
    if name == "lookup_terms":
        return _tool_lookup_terms(args)
    if name == "get_response_pattern":
        return _tool_get_response_pattern(args)
    if name == "search_golden_answers":
        return _tool_search_golden_answers(args)
    return {"error": f"Unknown tool: {name}"}


def _tool_search_faq(args: dict[str, object]) -> dict[str, object]:
    """FAQ 検索ツール。"""
    query = str(args["query"])
    source = str(args.get("source", "auto"))

    if source == "auto":
        result = search_auto(query)
        return {
            "hits": [
                {
                    "question": h["entry"]["question"],
                    "answer": h["entry"]["answer"],
                    "category": h["entry"]["category"],
                    "score": round(h["score"], 3),
                }
                for h in result["hits"]
            ],
            "matched_source": result["matched_source"],
        }

    hits = search(query, source=source)  # type: ignore[arg-type]
    return {
        "hits": [
            {
                "question": h["entry"]["question"],
                "answer": h["entry"]["answer"],
                "category": h["entry"]["category"],
                "score": round(h["score"], 3),
            }
            for h in hits
        ],
        "matched_source": source,
    }


def _tool_lookup_terms(args: dict[str, object]) -> dict[str, object]:
    """スピリチュアル用語検索ツール。"""
    terms: list[str] = args["terms"]  # type: ignore[assignment]
    faqs = load_faqs("spiritual")
    results: list[dict[str, str]] = []
    for term in terms:
        for entry in faqs:
            if term in entry["question"]:
                results.append(
                    {
                        "term": term,
                        "question": entry["question"],
                        "definition": entry["answer"],
                    }
                )
                break
    return {"terms": results, "found": len(results)}


def _tool_get_response_pattern(args: dict[str, object]) -> dict[str, object]:
    """回答パターン取得ツール。"""
    pattern_id = str(args["pattern_id"])
    pattern = get_pattern(pattern_id)
    if pattern is None:
        return {"error": f"Unknown pattern: {pattern_id}"}
    return {
        "pattern_id": pattern["id"],
        "name": pattern["name"],
        "instruction": pattern["system_instruction"],
    }


def _tool_search_golden_answers(args: dict[str, object]) -> dict[str, object]:
    """ゴールデンアンサー検索ツール。"""
    query = str(args["query"])
    results = search_golden(query)
    if not results:
        return {
            "examples": [],
            "message": "類似の正解回答例は見つかりませんでした。",
        }
    return {
        "examples": [
            {
                "question": g["question"],
                "answer": g["answer"],
                "intent": g["intent"],
            }
            for g in results
        ],
    }


# ---------------------------------------------------------------------------
# アシスタントメッセージのシリアライズ
# ---------------------------------------------------------------------------
def _serialize_assistant_msg(
    msg: object,
) -> dict[str, object]:
    """ChatCompletionMessage をメッセージ dict に変換する。"""
    result: dict[str, object] = {
        "role": "assistant",
        "content": getattr(msg, "content", None),
    }
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return result


# ---------------------------------------------------------------------------
# 意図推定（ツール使用パターンから推定）
# ---------------------------------------------------------------------------
_PATTERN_TO_INTENT: dict[str, str] = {
    "empathy": "consultation",
    "encouragement": "experience_report",
    "detailed_feedback": "method_question",
    "brief": "faq_question",
}


def _infer_intent(response_pattern: str, tools_used: list[str]) -> str:
    """使用された回答パターンとツールから意図を推定する。"""
    if response_pattern == "brief" and "search_faq" not in tools_used:
        return "greeting"
    return _PATTERN_TO_INTENT.get(response_pattern, "faq_question")


# ---------------------------------------------------------------------------
# メインエージェント実行
# ---------------------------------------------------------------------------
def run_agent(
    client: OpenAI,
    message: str,
    detail_level: str = "concise",
    source: str = "auto",
) -> AgentResult:
    """エージェントを実行し、ツール呼び出しを含む回答を生成する。

    Args:
        client: OpenAI クライアント
        message: ユーザーメッセージ
        detail_level: "concise" or "detailed"
        source: "auto", "faq", "spiritual", "salon"

    Returns:
        AgentResult: 回答・ソース・メタデータを含む結果
    """
    system_prompt = _build_agent_system_prompt()

    # ソース指定がある場合はユーザーメッセージにヒントを付与
    user_content = message
    if source != "auto":
        user_content = f"※ 検索ソース指定: {source}\n\n{message}"

    messages: list[dict[str, object]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    tools_used: list[str] = []
    all_sources: list[dict[str, str]] = []
    matched_source: str = "faq"
    response_pattern: str = "brief"
    max_tokens = 1024 if detail_level == "detailed" else 400

    # ツール呼び出しループ（最大 MAX_TOOL_ROUNDS 回）
    assistant_msg: object = None
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=_model_id(),
            messages=messages,  # type: ignore[arg-type]
            tools=AGENT_TOOLS,  # type: ignore[arg-type]
            tool_choice="auto",
            temperature=0.3,
            max_tokens=max_tokens,
        )
        assistant_msg = response.choices[0].message

        if not assistant_msg.tool_calls:
            # ツール呼び出しなし → 最終回答
            break

        # アシスタントメッセージを会話履歴に追加
        messages.append(_serialize_assistant_msg(assistant_msg))

        # 各ツール呼び出しを実行
        for tool_call in assistant_msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tools_used.append(name)

            result = _execute_tool(name, args)

            # メタデータを追跡
            if name == "search_faq":
                hits = result.get("hits", [])
                if hits:
                    all_sources = hits  # type: ignore[assignment]
                src = result.get("matched_source")
                if src:
                    matched_source = str(src)
            if name == "get_response_pattern":
                response_pattern = str(args.get("pattern_id", "brief"))

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
    else:
        # ツールラウンドを使い切った → ツールなしで最終回答を生成
        response = client.chat.completions.create(
            model=_model_id(),
            messages=messages,  # type: ignore[arg-type]
            temperature=0.3,
            max_tokens=max_tokens,
        )
        assistant_msg = response.choices[0].message

    answer = (getattr(assistant_msg, "content", None) or "").strip()
    intent = _infer_intent(response_pattern, tools_used)

    # ソース情報をフォーマット
    source_list: list[dict[str, str]] = [
        {
            "question": str(s.get("question", "")),
            "answer": str(s.get("answer", "")),
            "category": str(s.get("category", "")),
        }
        for s in all_sources
    ]

    return AgentResult(
        answer=answer,
        sources=source_list,
        matched_source=matched_source,
        intent=intent,
        tools_used=tools_used,
        response_pattern=response_pattern,
    )
