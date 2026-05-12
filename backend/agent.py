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

from embeddings import search_terms
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
                "スピリチュアル用語を意味検索する（212語対応）。"
                "用語名だけでなく、概念や説明文でも検索できる。"
                "例: 「前世の記憶」「エネルギーを手放す」等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "検索クエリ（用語名、概念、説明文など自由入力）"
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
            "name": "get_response_pattern",
            "description": (
                "回答トーン・形式の指示を取得する。意図に応じた適切なパターンを選択すること。"
                "negative_support=不満・退会・体調悪化, "
                "empathy=共感相談, encouragement=応援, "
                "detailed_feedback=詳細手順, brief=簡潔回答"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern_id": {
                        "type": "string",
                        "enum": [
                            "negative_support",
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

## 意図分類（最重要 — 必ず最初に判断すること）
get_response_pattern を呼び出す際に、以下の基準で正しいパターンを選択してください。
この選択が回答全体の方針を決定します。

### パターン選択の判断基準（上から順に判定 — 最初に該当したものを選ぶ）
1. **negative_support** ← 以下のいずれかに該当する場合は必ず negative_support を選ぶ
   【A. 直接的な不満・退会意向】
   - 「辞めたい」「退会したい」「もう無理」「お金の無駄」「効果がない」「詐欺」「騙された」「最悪」「許せない」「返金」等
   【B. 遠回しな不満・失望】
   - 「私には合わない」「意味があるのかな」「変わらない気がする」「全然変わらない」「ついていけない」
   - 継続への疑問（「来る意味あるのかな」「続けていいのか」「向いてないのかも」）
   【C. 受動的な離脱サイン】
   - 「お休みしようかな」「しばらく離れようかな」「参加できてない」＋継続意欲の低下
   - 講座・サロンから距離を置こうとしている表現
   【D. 人間関係トラブル】
   - 「傷ついた」「一緒にやりたくない」＋サロン内の人間関係
   - メンバーやチームとの関係で落ち込んでいる
   【E. 深刻な精神状態】
   - 「生きてる意味がわからない」「何をしても空回り」「死にたい」「消えたい」
   - 複数の身体症状の併発（「眠れない」＋「食欲ない」＋「気力ない」等）
   - 日常生活に支障が出ている状態
   【F. 身体的に深刻な状態】
   - 「何もできない」「起き上がれない」「動けない」
2. **empathy** ← メッセージに感情・体調変化・不安・悩みが含まれている場合
   - 「つらい」「不安」「うまくできない」「体調が悪い」「崩れている感じ」「〜な気がする」等
   - 末尾が「〜でしょうか？」「〜した方がいいですか？」でも、背景に悩みがあれば empathy
   - 身体の不調、エネルギーの変化、ワークの体感に関する相談 → 必ず empathy
   - 軽い体調変化の報告（「少し頭痛がした」等）で深刻度が低い場合 → empathy（negative_supportではない）
3. **encouragement** ← 体験や気づきの報告
   - 「衝撃を受けた」「気づいた」「できるようになった」「変わった」「すごい」等
   - ポジティブな変化の報告（「落ち着いてきました」「楽になりました」）
   - 「泣いた」「だるい」等でも、全体がポジティブなら encouragement
4. **detailed_feedback** ← やり方・手順の質問
   - 「やり方を教えて」「方法は？」「どうすれば？」等
5. **brief** ← 事実確認のみ or あいさつ
   - 感情や体験を含まない純粋な質問（料金、手続き、ルール、用語の意味「〜とは何ですか？」）
   - 「こんにちは」「はじめまして」等のあいさつ

### negative_support と empathy の区別（重要）
- 講座やサロンへの継続意欲が低下 → negative_support（人間対応が必要）
- サロン内の人間関係で傷ついている → negative_support
- 複数の深刻な症状（眠れない＋食欲ない等） → negative_support
- 「合わない」「意味がない」「変わらない」等の失望 → negative_support
- ワーク中の一時的な体感（頭痛、だるさ等）→ empathy
- 不安・悩みはあるが継続意欲はある → empathy

### empathy と brief の区別（重要）
- 操作・手続き・ルールの困りごと（「〜できない」「〜がわかりにくい」「〜に入れない」）→ brief（FAQ検索で具体的解決策がある可能性が高い）
- Zoom、LINE、会員サイト、アーカイブ、領収書など具体的な対象がある → brief
- スピリチュアルなワーク・体感に関する悩み（体調変化、エネルギー、不安）→ empathy

### 迷った場合のルール
- 迷ったら negative_support を選ぶ（見逃しより過検知の方が安全）
- 具体的な対象物（Zoom、LINE、サイト等）が含まれる困りごとは brief を選ぶ
- スピリチュアルな体験・感情の悩みで具体的な対象物がない場合は empathy を選ぶ
- 質問形で終わっていても、長い体験や状況説明があれば empathy を選ぶ

## ツール使用ガイド
get_response_pattern でパターンを選択した後、以下のツールを呼び出してください：
- negative_support (negative) → 追加ツール不要（人間対応への橋渡しのみ）
- brief (faq_question) → 【必須】search_faq(query) を必ず呼ぶ。自分の知識だけで回答しない。スピリチュアル用語の質問には lookup_terms(query) も呼ぶ
- empathy (consultation) → 【必須】search_faq(query, source="auto") を必ず呼ぶ + lookup_terms(query)
- encouragement (experience_report) → lookup_terms(query) で用語の背景を確認
- detailed_feedback (method_question) → 【必須】search_faq(query, source="auto") を必ず呼ぶ + lookup_terms(query)
- brief (greeting) → 追加ツール不要

【重要】あいさつ以外の質問では、自分の知識だけで回答せず、必ず search_faq を呼んでFAQデータを確認してください。

【重要ルール】スピリチュアル用語に関する質問には、自分の知識だけで回答せず、必ず lookup_terms を呼んでサロン固有の定義を確認してください。

## スピリチュアル用語（背景知識）
以下の用語は回答時の簡易的な背景知識です。より詳しい定義が必要な場合は lookup_terms を呼んでください。

{spiritual_block}

## 回答ルール（全パターン共通）

### 最優先ルール: FAQデータの具体情報を必ず使う
- 【絶対厳守】search_faq で取得したFAQデータに具体的な解決策・手順・情報がある場合、それを必ず回答に含めること
- 「サポート担当者にお繋ぎしますね」「個別相談LINEへどうぞ」だけで終わらせるのは禁止。具体的な回答を先に伝え、それでも解決しない場合の誘導として添える
- 3行以内の制約は「要点を絞る」意味であり「FAQの具体情報を省略する」意味ではない
- FAQの回答に手順、条件、場所、方法が書いてあれば、それを自分の言葉で簡潔に伝えること

### FAQ質問（brief）の回答構造
1行目: 具体的な回答（FAQデータの核心情報）
2行目: 補足や手順（必要な場合）
3行目: それでも解決しない場合の誘導（任意）

### 形式ルール
- 回答は最大3行以内（LINE想定・スマホで読む高齢者向け）
- 箇条書きは最大3つまで
- 絵文字は1〜2個まで（✨🙏💫 等）
- 「FAQによると」「番号X」のようなメタな言い方は避け、自然な会話文にしてください
- 「クライアント」「キャッシュ」「再起動」などの専門用語は避け、やさしい言葉で説明してください
- 断定は避け、「〜とされています」「〜かもしれません」を使ってください
- 医療・診断に関わる内容には踏み込まず、「専門の方にご相談されるのも良いかもしれません」と促してください
- FAQに該当する情報が見つからない場合のみ「サポート担当者にお繋ぎしますね」とお伝えください
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
    """スピリチュアル用語 Embedding 検索ツール（212語対応）。"""
    query = str(args["query"])
    hits = search_terms(query, top_k=3)
    if not hits:
        return {"terms": [], "found": 0}
    return {
        "terms": [
            {
                "term": h["term"],
                "category": h["category"],
                "definition": h["definition"],
                "example": h["example"],
                "score": h["score"],
            }
            for h in hits
        ],
        "found": len(hits),
    }


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
    "negative_support": "negative",
    "empathy": "consultation",
    "encouragement": "experience_report",
    "detailed_feedback": "method_question",
    "brief": "faq_question",
}

_GREETING_KEYWORDS: list[str] = [
    "こんにちは",
    "はじめまして",
    "おはよう",
    "こんばんは",
    "よろしく",
    "お疲れ",
    "ありがとう",
]


def _infer_intent(
    response_pattern: str, tools_used: list[str], query: str
) -> str:
    """使用された回答パターン・ツール・元クエリから意図を推定する。"""
    if response_pattern == "brief":
        if any(kw in query for kw in _GREETING_KEYWORDS) and len(query) < 30:
            return "greeting"
        return "faq_question"
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

    # --- Step 1: 意図分類（get_response_pattern を強制呼び出し） ---
    response = client.chat.completions.create(
        model=_model_id(),
        messages=messages,  # type: ignore[arg-type]
        tools=AGENT_TOOLS,  # type: ignore[arg-type]
        tool_choice={
            "type": "function",
            "function": {"name": "get_response_pattern"},
        },
        temperature=0.3,
        max_tokens=max_tokens,
    )
    assistant_msg = response.choices[0].message
    messages.append(_serialize_assistant_msg(assistant_msg))
    for tool_call in assistant_msg.tool_calls or []:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        tools_used.append(name)
        result = _execute_tool(name, args)
        if name == "get_response_pattern":
            response_pattern = str(args.get("pattern_id", "brief"))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    # --- Step 1.5: FAQ検索を強制呼び出し（greeting/negative以外） ---
    _SKIP_FAQ_PATTERNS = {"negative_support"}
    _is_greeting = response_pattern == "brief" and _infer_intent(
        response_pattern, tools_used, message
    ) == "greeting"
    if response_pattern not in _SKIP_FAQ_PATTERNS and not _is_greeting:
        faq_result = _tool_search_faq({"query": message, "source": "auto"})
        tools_used.append("search_faq")
        hits = faq_result.get("hits", [])
        if hits:
            all_sources = hits  # type: ignore[assignment]
        src = faq_result.get("matched_source")
        if src:
            matched_source = str(src)
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "forced_faq",
                        "type": "function",
                        "function": {
                            "name": "search_faq",
                            "arguments": json.dumps(
                                {"query": message, "source": "auto"},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": "forced_faq",
                "content": json.dumps(faq_result, ensure_ascii=False),
            }
        )

    # --- Step 2: 追加ツール呼び出し（auto） ---
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
            break

        messages.append(_serialize_assistant_msg(assistant_msg))

        for tool_call in assistant_msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tools_used.append(name)

            result = _execute_tool(name, args)

            if name == "search_faq":
                hits = result.get("hits", [])
                if hits:
                    all_sources = hits  # type: ignore[assignment]
                src = result.get("matched_source")
                if src:
                    matched_source = str(src)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
    else:
        response = client.chat.completions.create(
            model=_model_id(),
            messages=messages,  # type: ignore[arg-type]
            temperature=0.3,
            max_tokens=max_tokens,
        )
        assistant_msg = response.choices[0].message

    answer = (getattr(assistant_msg, "content", None) or "").strip()
    intent = _infer_intent(response_pattern, tools_used, message)

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
