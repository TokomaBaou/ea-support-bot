"""スピリチュアル用語 Embedding 検索

OpenAI text-embedding-3-small でベクトル化し、
コサイン類似度で意味的に近い用語を検索する。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TypedDict

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

_TERMS_PATH = Path(__file__).parent / "data" / "spiritual_terms_full.json"
_EMBEDDINGS_PATH = Path(__file__).parent / "data" / "spiritual_embeddings.npz"
_MODEL = "text-embedding-3-small"


class TermHit(TypedDict):
    term: str
    reading: str
    english: str
    category: str
    definition: str
    example: str
    score: float


class _Store:
    """Embedding ストア（遅延ロード・シングルトン）。"""

    def __init__(self) -> None:
        self.terms: list[dict[str, str]] = []
        self.matrix: np.ndarray | None = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.terms = _load_terms()
        if _EMBEDDINGS_PATH.exists():
            data = np.load(_EMBEDDINGS_PATH)
            self.matrix = data["embeddings"]
            if self.matrix.shape[0] != len(self.terms):
                logger.warning(
                    "Embedding count (%d) != term count (%d); rebuilding.",
                    self.matrix.shape[0],
                    len(self.terms),
                )
                self.matrix = None
        if self.matrix is None:
            logger.info("Building embeddings for %d terms...", len(self.terms))
            self.matrix = _build_embeddings(self.terms)
            np.savez_compressed(_EMBEDDINGS_PATH, embeddings=self.matrix)
            logger.info("Embeddings saved to %s", _EMBEDDINGS_PATH)
        self._loaded = True


_store = _Store()


def _load_terms() -> list[dict[str, str]]:
    with _TERMS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def _embed_texts(texts: list[str]) -> np.ndarray:
    client = _get_client()
    batch_size = 100
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=_MODEL, input=batch)
        all_vecs.extend([d.embedding for d in resp.data])
    return np.array(all_vecs, dtype=np.float32)


def _build_embeddings(terms: list[dict[str, str]]) -> np.ndarray:
    texts = [
        f"{t['term']} {t['definition']} {t['example']}" for t in terms
    ]
    return _embed_texts(texts)


def search_terms(query: str, top_k: int = 3) -> list[TermHit]:
    """クエリに意味的に近い用語を返す。"""
    _store._ensure_loaded()
    if _store.matrix is None or len(_store.terms) == 0:
        return []

    q_vec = _embed_texts([query])[0]

    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    m_norms = _store.matrix / (
        np.linalg.norm(_store.matrix, axis=1, keepdims=True) + 1e-9
    )
    scores = m_norms @ q_norm

    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[TermHit] = []
    for idx in top_indices:
        t = _store.terms[idx]
        results.append(
            TermHit(
                term=t["term"],
                reading=t.get("reading", ""),
                english=t.get("english", ""),
                category=t.get("category", ""),
                definition=t["definition"],
                example=t.get("example", ""),
                score=round(float(scores[idx]), 4),
            )
        )
    return results


def rebuild_embeddings() -> int:
    """Embedding を再生成して保存する。"""
    terms = _load_terms()
    matrix = _build_embeddings(terms)
    np.savez_compressed(_EMBEDDINGS_PATH, embeddings=matrix)
    _store.terms = terms
    _store.matrix = matrix
    _store._loaded = True
    return len(terms)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        from dotenv import load_dotenv

        load_dotenv()
        n = rebuild_embeddings()
        print(f"Built embeddings for {n} terms.")
    elif len(sys.argv) > 1 and sys.argv[1] == "search":
        from dotenv import load_dotenv

        load_dotenv()
        q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "チャクラ"
        hits = search_terms(q, top_k=5)
        for h in hits:
            print(f"  [{h['score']:.4f}] {h['term']} - {h['definition'][:60]}...")
    else:
        print("Usage: python embeddings.py build | search <query>")
