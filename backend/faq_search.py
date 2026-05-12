"""FAQ 検索ロジック

文字 bigram + キーワード一致による軽量な検索。
日本語形態素解析を使わずに済ませるため、bigram の IDF 重み付けで
「ください」「ですか」など頻出語尾の影響を抑える。

`source` パラメータでソースを切り替える:
  - "faq": 一般 FAQ (faq.json)
  - "spiritual": スピリチュアル用語辞典 (spiritual_faq.json)
  - "salon": スピ覚醒サロン入会後 FAQ (salon_faq.json)
IDF はソースごとに独立に算出する (語彙特性が異なるため)。

ひらがな口語表現の正規化:
  ユーザーがひらがなで入力した場合 (例: 「りょうしゅうしょ」) を
  正式表記 (例: 「領収書」) に置換してから検索する。
  長いキーから順にマッチし、部分一致にも対応する。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from math import log
from pathlib import Path
from typing import Literal, TypedDict


class FaqEntry(TypedDict):
    id: str
    category: str
    question: str
    answer: str


class FaqHit(TypedDict):
    entry: FaqEntry
    score: float


class AutoSearchResult(TypedDict):
    hits: list[FaqHit]
    matched_source: str  # "faq" | "spiritual" | "salon"


SourceName = Literal["faq", "spiritual", "salon"]
ALL_SOURCES: list[SourceName] = ["faq", "spiritual", "salon"]
DEFAULT_SOURCE: SourceName = "faq"

_DATA_DIR = Path(__file__).parent / "data"
_SOURCE_PATHS: dict[SourceName, Path] = {
    "faq": _DATA_DIR / "faq.json",
    "spiritual": _DATA_DIR / "spiritual_faq.json",
    "salon": _DATA_DIR / "salon_faq.json",
}
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[぀-ゟ゠-ヿ一-鿿]+")

# ---------------------------------------------------------------------------
# ひらがな → 正式表記 正規化辞書
# FAQ / spiritual / salon 全データから頻出キーワードを抽出し、
# ユーザーがひらがなで崩して入力しても正しくヒットできるようにする。
# キーはすべて **ひらがな**、値は FAQ データ内の表記に合わせる。
# ---------------------------------------------------------------------------
_READING_MAP: dict[str, str] = {
    # --- 一般 FAQ ---
    "りょうしゅうしょ": "領収書",
    "かいいんとうろく": "会員登録",
    "ろぐいん": "ログイン",
    "ぱすわーど": "パスワード",
    "めーるあどれす": "メールアドレス",
    "たいかい": "退会",
    "こうざ": "講座",
    "どうが": "動画",
    "しんちょく": "進捗",
    "しゅうりょうしょう": "修了証",
    "じゅこうきげん": "受講期限",
    "げっしゃせい": "月謝制",
    "しはらい": "支払い",
    "へんきん": "返金",
    "ぶんかつばらい": "分割払い",
    "かいやく": "解約",
    "あぷり": "アプリ",
    "かだい": "課題",
    "かりきゅらむ": "カリキュラム",
    "おんらいん": "オンライン",
    "くれじっとかーど": "クレジットカード",
    "ぎんこうふりこみ": "銀行振込",
    "さいじゅこう": "再受講",
    "あーかいぶ": "アーカイブ",
    # --- スピリチュアル用語 ---
    "あかしっくれこーど": "アカシックレコード",
    "あすとらるかい": "アストラル界",
    "あすとらるたい": "アストラル体",
    "あせんでっどますたー": "アセンデッド・マスター",
    "あちゅーめんと": "アチューメント",
    "あふぁめーしょん": "アファメーション",
    "いんてんしょん": "インテンション",
    "えーてるたい": "エーテル体",
    "おーぶ": "オーブ",
    "おーら": "オーラ",
    "おらくるかーど": "オラクルカード",
    "がーでぃあん": "ガーディアン",
    "かるま": "カルマ",
    "くんだりーに": "クンダリーニ",
    "くんだりにー": "クンダリーニ",
    "こうてんはんのう": "好転反応",
    "こーざるたい": "コーザル体",
    "こーるいんめそっど": "コールインメソッド",
    "さーどあい": "サードアイ",
    "さいきっくあたっく": "サイキックアタック",
    "しっくすせんす": "シックスセンス",
    "すぴりっとがいど": "スピリットガイド",
    "すぴりちゅある": "スピリチュアル",
    "すぴりちゅあるはどう": "スピリチュアル波動",
    "すぴりちゅあるひーりんぐ": "スピリチュアルヒーリング",
    "すぴりちゅあるわーるど": "スピリチュアルワールド",
    "せるふあちゅーんめんと": "セルフアチューンメント",
    "せんたりんぐ": "センタリング",
    "そうるめいと": "ソウルメイト",
    "ちゃくら": "チャクラ",
    "ついんそうる": "ツインソウル",
    "ついんれい": "ツインレイ",
    "でじゃゔゅ": "デジャヴュ",
    "でじゃぶ": "デジャヴュ",
    "てれぱしー": "テレパシー",
    "とりぷるれい": "トリプルレイ",
    "はいやーせるふ": "ハイヤーセルフ",
    "ひーらー": "ヒーラー",
    "ひーりんぐ": "ヒーリング",
    "ひきよせのほうそく": "引き寄せの法則",
    "ひきよせ": "引き寄せ",
    "ぷらーな": "プラーナ",
    "ぷらくてぃしょなー": "プラクティショナー",
    "まんとら": "マントラ",
    "めんたるたい": "メンタル体",
    "らいとうぉーりあ": "ライトウォーリア",
    "らいとわーかー": "ライトワーカー",
    "りーでぃんぐ": "リーディング",
    "ぷれあですせいじん": "プレアデス星人",
    "ぷれあです": "プレアデス",
    "しりうすせいじん": "シリウス星人",
    "しりうす": "シリウス",
    "おりおんせいじん": "オリオン星人",
    "おりおん": "オリオン",
    # --- サロン FAQ ---
    "みらくるちぇんじ": "ミラクルチェンジ",
    "じょうかわーく": "浄化ワーク",
    "じょうか": "浄化",
    "めいそう": "瞑想",
    "ぐらうんでぃんぐ": "グラウンディング",
    "じげんじょうしょう": "次元上昇",
    "ふぉありんぐ": "フォアリング",
    "れいてき": "霊的",
    "ずーむ": "Zoom",
    "らいん": "LINE",
    "ちーむ": "チーム",
    "くらぶ": "クラブ",
    "わーく": "ワーク",
    "そうぎょうしき": "創業式",
    "りあるこうざ": "リアル講座",
    "ぷろこーす": "プロコース",
    "きせきこーす": "奇跡コース",
    "かいかこーす": "開花コース",
    "かくせいこーす": "覚醒コース",
    "かいいんさいと": "会員サイト",
    "こべつそうだん": "個別相談",
    "あうとぷっと": "アウトプット",
    "せんそう": "前世",
    "かこせ": "過去世",
}

# 長いキーから優先してマッチさせるためソート済みリストを事前生成
_READING_KEYS_SORTED: list[tuple[str, str]] = sorted(
    _READING_MAP.items(), key=lambda kv: len(kv[0]), reverse=True
)


def normalize_query(query: str) -> str:
    """ひらがな口語表現を正式表記に置換する。

    長いキーから順に部分一致で置換することで、
    「りょうしゅうしょほしい」→「領収書ほしい」のような変換を行う。
    """
    result = query
    for reading, canonical in _READING_KEYS_SORTED:
        if reading in result:
            result = result.replace(reading, canonical)
    return result


def load_faqs(source: SourceName = DEFAULT_SOURCE) -> list[FaqEntry]:
    path = _SOURCE_PATHS[source]
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _bigrams(text: str) -> set[str]:
    text = text.replace(" ", "").replace("　", "")
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _keyword_tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_PATTERN.finditer(text)}


@lru_cache(maxsize=4)
def _idf_weights(source: SourceName) -> dict[str, float]:
    """指定ソースの bigram IDF 重みを事前計算する。

    レアな bigram (例: 「領収」「ツイ」) は重く、
    頻出する語尾 bigram (例: 「くだ」「です」「とさ」) は軽くなる。
    """
    faqs = load_faqs(source)
    df: dict[str, int] = {}
    for entry in faqs:
        text = f"{entry['question']} {entry['answer']}"
        for bigram in _bigrams(text):
            df[bigram] = df.get(bigram, 0) + 1
    n = len(faqs)
    return {bg: log((n + 1) / (count + 1)) + 1.0 for bg, count in df.items()}


@lru_cache(maxsize=1)
def _global_idf_weights() -> dict[str, float]:
    """全ソースを横断した bigram IDF 重み。

    ソース間のスコア比較時に使用する。
    ソースごとの IDF では「教えて」「ください」等の共通パターンが
    ソースによって異なる重みを持ち比較が歪むため、
    全データを母集団とした統一 IDF を用いる。
    """
    all_faqs: list[FaqEntry] = []
    for src in ALL_SOURCES:
        all_faqs.extend(load_faqs(src))
    df: dict[str, int] = {}
    for entry in all_faqs:
        text = f"{entry['question']} {entry['answer']}"
        for bigram in _bigrams(text):
            df[bigram] = df.get(bigram, 0) + 1
    n = len(all_faqs)
    return {bg: log((n + 1) / (count + 1)) + 1.0 for bg, count in df.items()}


def _idf_overlap(
    query_bigrams: set[str],
    target_bigrams: set[str],
    weights: dict[str, float],
) -> float:
    if not query_bigrams:
        return 0.0
    overlap = query_bigrams & target_bigrams
    if not overlap:
        return 0.0
    overlap_score = sum(weights.get(bg, 1.0) for bg in overlap)
    query_score = sum(weights.get(bg, 1.0) for bg in query_bigrams)
    if query_score == 0:
        return 0.0
    return overlap_score / query_score


def _score(query: str, entry: FaqEntry, weights: dict[str, float]) -> float:
    query_bigrams = _bigrams(query)
    if not query_bigrams:
        return 0.0

    target_text = f"{entry['question']} {entry['answer']} {entry['category']}"
    target_bigrams = _bigrams(target_text)
    target_score = _idf_overlap(query_bigrams, target_bigrams, weights)

    question_bigrams = _bigrams(entry["question"])
    question_score = _idf_overlap(query_bigrams, question_bigrams, weights)

    query_tokens = _keyword_tokens(query)
    target_tokens = _keyword_tokens(target_text)
    if query_tokens:
        token_overlap = len(query_tokens & target_tokens)
        token_score = token_overlap / len(query_tokens)
    else:
        token_score = 0.0

    return target_score * 0.4 + question_score * 0.4 + token_score * 0.2


def search(
    query: str,
    top_k: int = 3,
    min_score: float = 0.05,
    source: SourceName = DEFAULT_SOURCE,
) -> list[FaqHit]:
    if not query.strip():
        return []

    # ひらがな口語表現を正式表記に正規化してから検索
    normalized = normalize_query(query)

    faqs = load_faqs(source)
    weights = _idf_weights(source)
    scored: list[FaqHit] = [
        {"entry": entry, "score": _score(normalized, entry, weights)}
        for entry in faqs
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [hit for hit in scored[:top_k] if hit["score"] >= min_score]


# ---------------------------------------------------------------------------
# ルーティング用: 共通パターン除去 → コアキーワード抽出
# 「ツインレイについて教えて」→「ツインレイ」のように
# 質問の骨格 (function words) を除去してトピック語だけ残す。
# 長いパターンから優先的にマッチさせる。
# ---------------------------------------------------------------------------
_ROUTING_STRIP_PATTERNS: list[str] = sorted(
    [
        "のやり方を教えてください",
        "について教えてください",
        "について教えて",
        "はどこですればいいですか",
        "はどうすればいいですか",
        "を教えてください",
        "ってなんですか",
        "って何ですか",
        "はもらえますか",
        "したいのですが",
        "したいです",
        "はいつですか",
        "とは何ですか",
        "とはなんですか",
        "を教えて",
        "について",
        "てください",
        "ですか？",
        "ですか",
        "ですが",
        "って",
        "とは",
    ],
    key=len,
    reverse=True,
)


def _extract_core_keywords(query: str) -> str:
    """クエリから共通パターンを除去してコアキーワードを抽出する。"""
    result = query.strip().rstrip("？?。、！!")
    for pattern in _ROUTING_STRIP_PATTERNS:
        result = result.replace(pattern, "")
    return result.strip()


# ---------------------------------------------------------------------------
# Step 0: 優先ルーティングテーブル
# キーワードの組み合わせで特定のソースに強制ルーティングする。
# bigram フォールバックでは区別しにくい「動画→faq vs salon」等の
# 曖昧なケースを解消するためのルール。
# ---------------------------------------------------------------------------
_PRIORITY_ROUTES: list[dict[str, object]] = [
    # 「動画」単体（音関連を除く）→ faq（動画再生トラブル系）
    {
        "require_all": ["動画"],
        "exclude": ["音", "音声", "音量"],
        "source": "faq",
    },
    # 「音」+「出」→ salon（音が出ない系）
    {
        "require_all": ["音", "出"],
        "source": "salon",
    },
    # 「Zoom」→ salon（Zoom 接続トラブル系）
    {
        "require_any": ["Zoom", "zoom", "ZOOM"],
        "source": "salon",
    },
]


def _check_priority_routes(query: str) -> SourceName | None:
    """優先ルーティングテーブルを照合し、マッチするソースを返す。

    マッチしない場合は None を返す。
    ルールは定義順に評価され、最初にマッチしたものが採用される。
    """
    q_lower = query.lower()
    for rule in _PRIORITY_ROUTES:
        require_all: list[str] = rule.get("require_all", [])  # type: ignore[assignment]
        require_any: list[str] = rule.get("require_any", [])  # type: ignore[assignment]
        exclude: list[str] = rule.get("exclude", [])  # type: ignore[assignment]
        source: SourceName = rule["source"]  # type: ignore[assignment]

        # require_all: すべてのキーワードが含まれること
        if require_all and not all(kw in query for kw in require_all):
            continue

        # require_any: いずれかのキーワードが含まれること
        if require_any and not any(kw.lower() in q_lower for kw in require_any):
            continue

        # require_all も require_any も空 → ルール不正、スキップ
        if not require_all and not require_any:
            continue

        # exclude: いずれかのキーワードが含まれていたら不一致
        if exclude and any(kw in query for kw in exclude):
            continue

        return source

    return None


def search_auto(
    query: str,
    top_k: int = 3,
    min_score: float = 0.05,
) -> AutoSearchResult:
    """全ソースを横断検索し、最もスコアの高いソースの結果を返す。

    Step 0 (優先ルーティング):
      キーワードの組み合わせルールで即決できるケースを先に処理する。
      「動画」→faq、「音+出」→salon、「Zoom」→salon など。

    Step 1 (キーワードルーティング):
      クエリから共通パターンを除去してコアキーワードを抽出し、
      各ソースの FAQ 質問文に部分一致するか調べる。
      一致が見つかったソースで即決する。

    Step 2 (フォールバック: bigram ルーティング):
      キーワード一致が無い場合は、グローバル IDF + 質問テキスト
      重視の bigram スコアで最適ソースを判定する。

    Step 3: 選ばれたソースでソース固有 search() を実行して返す。
    """
    if not query.strip():
        return AutoSearchResult(hits=[], matched_source="faq")

    normalized = normalize_query(query)

    # --- Step 0: 優先ルーティング ---
    priority_source = _check_priority_routes(normalized)
    if priority_source is not None:
        hits = search(query, top_k=top_k, min_score=min_score, source=priority_source)
        return AutoSearchResult(hits=hits, matched_source=priority_source)

    # --- Step 1: コアキーワードの部分一致ルーティング ---
    core = _extract_core_keywords(normalized)
    if len(core) >= 2:
        best_source_kw: SourceName | None = None
        best_score_kw: float = 0.0
        for src in ALL_SOURCES:
            faqs = load_faqs(src)
            for entry in faqs:
                target = f"{entry['question']} {entry['answer']}"
                if core in target:
                    # 質問テキストに直接含まれる場合はボーナス
                    bonus = 1.5 if core in entry["question"] else 1.0
                    score = len(core) * bonus
                    if score > best_score_kw:
                        best_score_kw = score
                        best_source_kw = src
        if best_source_kw is not None:
            hits = search(query, top_k=top_k, min_score=min_score, source=best_source_kw)
            return AutoSearchResult(hits=hits, matched_source=best_source_kw)

    # --- Step 2: bigram フォールバックルーティング ---
    global_weights = _global_idf_weights()
    source_best: dict[SourceName, float] = {}

    for src in ALL_SOURCES:
        faqs = load_faqs(src)
        top_score = 0.0
        for entry in faqs:
            query_bigrams = _bigrams(normalized)
            question_bigrams = _bigrams(entry["question"])
            score = _idf_overlap(query_bigrams, question_bigrams, global_weights)
            if score > top_score:
                top_score = score
        source_best[src] = top_score

    # salonソースを優先: salonが他ソースの80%以上のスコアならsalonを選択
    best_score = max(source_best.values()) if source_best else 0.0
    salon_score = source_best.get("salon", 0.0)
    if salon_score > 0 and salon_score >= best_score * 0.8:
        best_source: SourceName = "salon"
    else:
        best_source = max(source_best, key=lambda s: source_best[s])

    best_hits = search(query, top_k=top_k, min_score=min_score, source=best_source)
    return AutoSearchResult(hits=best_hits, matched_source=best_source)
