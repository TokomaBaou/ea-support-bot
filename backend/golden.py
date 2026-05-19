"""ゴールデンアンサー管理

正解回答ペアの保存・検索・few-shot フォーマットを行う。
backend/data/golden_answers.json で永続化する。

検索は OpenAI Embedding（コサイン類似度）を使用。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class GoldenEntry(TypedDict, total=False):
    id: str
    question: str
    answer: str
    intent: str
    source: str
    created_at: str
    # 拡張フィールド
    context: str        # 事前の文脈・背景
    category: str       # カテゴリ（例: 身体の不調・震え）
    user_reaction: str  # ユーザーの感謝・反応


_DATA_PATH = Path(__file__).parent / "data" / "golden_answers.json"
_EMBEDDINGS_PATH = Path(__file__).parent / "data" / "golden_embeddings.npz"
_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Embedding ストア（遅延ロード・シングルトン）
# ---------------------------------------------------------------------------
class _GoldenStore:
    """ゴールデンアンサー用 Embedding ストア。"""

    def __init__(self) -> None:
        self.entries: list[GoldenEntry] = []
        self.matrix: np.ndarray | None = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.entries = _load_all()
        if not self.entries:
            self._loaded = True
            return

        if _EMBEDDINGS_PATH.exists():
            data = np.load(_EMBEDDINGS_PATH)
            self.matrix = data["embeddings"]
            if self.matrix.shape[0] != len(self.entries):
                logger.warning(
                    "Golden embedding count (%d) != entry count (%d); rebuilding.",
                    self.matrix.shape[0],
                    len(self.entries),
                )
                self.matrix = None
        if self.matrix is None:
            logger.info("Building golden embeddings for %d entries...", len(self.entries))
            self.matrix = _build_embeddings(self.entries)
            np.savez_compressed(_EMBEDDINGS_PATH, embeddings=self.matrix)
            logger.info("Golden embeddings saved to %s", _EMBEDDINGS_PATH)
        self._loaded = True

    def invalidate(self) -> None:
        """キャッシュを無効化して次回アクセス時に再ロードする。"""
        self._loaded = False
        self.entries = []
        self.matrix = None


_store = _GoldenStore()


# ---------------------------------------------------------------------------
# OpenAI Embedding ユーティリティ
# ---------------------------------------------------------------------------
def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def _embed_texts(texts: list[str]) -> np.ndarray:
    """テキストリストを Embedding ベクトルに変換する。"""
    client = _get_client()
    batch_size = 100
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=_MODEL, input=batch)
        all_vecs.extend([d.embedding for d in resp.data])
    return np.array(all_vecs, dtype=np.float32)


def _build_embeddings(entries: list[GoldenEntry]) -> np.ndarray:
    """ゴールデンアンサーの質問+文脈を Embedding 化する。"""
    texts = []
    for e in entries:
        # 質問 + カテゴリ + 文脈を結合して検索精度を上げる
        parts = [e.get("question", "")]
        if e.get("category"):
            parts.append(e["category"])
        if e.get("context"):
            # 文脈は長いので先頭200文字まで
            parts.append(e["context"][:200])
        texts.append(" ".join(parts))
    return _embed_texts(texts)


# ---------------------------------------------------------------------------
# データ永続化
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 検索（Embedding コサイン類似度）
# ---------------------------------------------------------------------------
def search_golden(query: str, top_k: int = 3) -> list[GoldenEntry]:
    """クエリに意味的に近いゴールデンアンサーを Embedding 検索で返す。"""
    _store._ensure_loaded()
    if _store.matrix is None or len(_store.entries) == 0:
        return []

    q_vec = _embed_texts([query])[0]

    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    m_norms = _store.matrix / (
        np.linalg.norm(_store.matrix, axis=1, keepdims=True) + 1e-9
    )
    scores = m_norms @ q_norm

    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[GoldenEntry] = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < 0.3:  # 類似度閾値
            continue
        entry = dict(_store.entries[idx])
        entry["_score"] = round(score, 4)  # type: ignore[typeddict-unknown-key]
        results.append(entry)  # type: ignore[arg-type]
    return results


# ---------------------------------------------------------------------------
# 追加
# ---------------------------------------------------------------------------
def add_golden(
    question: str,
    answer: str,
    intent: str,
    source: str = "method_qa",
    context: str = "",
    category: str = "",
    user_reaction: str = "",
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
        "context": context,
        "category": category,
        "user_reaction": user_reaction,
    }
    entries.append(entry)
    _save_all(entries)
    _store.invalidate()  # キャッシュを無効化
    return entry


# ---------------------------------------------------------------------------
# few-shot フォーマット
# ---------------------------------------------------------------------------
def format_few_shot(goldens: list[GoldenEntry]) -> str:
    """ゴールデンアンサーを few-shot 参考テキストにフォーマットする。"""
    if not goldens:
        return ""
    parts: list[str] = []
    for g in goldens:
        lines = []
        if g.get("category"):
            lines.append(f"カテゴリ: {g['category']}")
        if g.get("context"):
            lines.append(f"背景: {g['context'][:100]}...")
        lines.append(f"Q: {g['question']}")
        lines.append(f"A: {g['answer']}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Embedding 再構築
# ---------------------------------------------------------------------------
def rebuild_embeddings() -> int:
    """Embedding を再生成して保存する。"""
    entries = _load_all()
    if not entries:
        return 0
    matrix = _build_embeddings(entries)
    np.savez_compressed(_EMBEDDINGS_PATH, embeddings=matrix)
    _store.entries = entries
    _store.matrix = matrix
    _store._loaded = True
    return len(entries)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        from dotenv import load_dotenv

        load_dotenv()
        n = rebuild_embeddings()
        print(f"Built golden embeddings for {n} entries.")
    elif len(sys.argv) > 1 and sys.argv[1] == "search":
        from dotenv import load_dotenv

        load_dotenv()
        q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "震えが止まらない"
        hits = search_golden(q, top_k=3)
        for h in hits:
            print(f"  [{h.get('_score', '?')}] {h.get('category', '')} - {h['question'][:60]}...")
    else:
        print("Usage: python golden.py build | search <query>")
