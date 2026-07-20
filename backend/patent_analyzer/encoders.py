"""Embedding encoders for the rerank path.

text-embedding-005 on Vertex is the production encoder (FiNE-Patents encoder
A/B: R@10 .880 vs MiniLM .780; batches 100 texts/request — see
backend/evals/README.md). MiniLM remains the offline/failure fallback, and
BM25 fusion only earns its keep under that weak fallback encoder.
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

GC_PROJECT = os.environ.get("GC_PROJECT", "aime-hello-world")
GC_LOCATION = os.environ.get("GC_LOCATION", "us-west1")
VERTEX_MODEL = os.environ.get("RERANK_ENCODER_MODEL", "text-embedding-005")
CACHE_DIR = Path(os.environ.get("EMBED_CACHE_DIR", "/tmp/embed_cache"))

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(vertexai=True, project=GC_PROJECT, location=GC_LOCATION)
    return _client


def embed_vertex(texts: list[str], task_type: str, model: str | None = None,
                 cache_dir: Path | None = None) -> np.ndarray:
    """Embed via Vertex AI with a per-text disk cache (keyed by model+task+sha1).

    text-embedding-005 batches up to 100 texts/request; gemini-embedding-001
    accepts 1, so cache misses fan out over a thread pool.
    """
    from google.genai import types as genai_types

    model = model or VERTEX_MODEL
    cache = (cache_dir or CACHE_DIR) / f"{model}__{task_type}"
    cache.mkdir(parents=True, exist_ok=True)

    def key(t: str) -> Path:
        return cache / (hashlib.sha1(t.encode()).hexdigest() + ".npy")

    out: dict[int, np.ndarray] = {}
    missing: list[int] = []
    for i, t in enumerate(texts):
        p = key(t)
        if p.exists():
            out[i] = np.load(p)
        else:
            missing.append(i)

    if missing:
        client = _get_client()
        config = genai_types.EmbedContentConfig(task_type=task_type, output_dimensionality=768)
        batch = 100 if model == "text-embedding-005" else 1

        def embed_batch(idxs: list[int]):
            contents = [texts[i][:8000] or " " for i in idxs]
            resp = client.models.embed_content(model=model, contents=contents, config=config)
            for i, emb in zip(idxs, resp.embeddings):
                v = np.array(emb.values, dtype=np.float32)
                v /= (np.linalg.norm(v) or 1.0)
                np.save(key(texts[i]), v)
                out[i] = v

        groups = _batch_by_budget(missing, texts, batch)
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(embed_batch, groups))

    return np.stack([out[i] for i in range(len(texts))])


# Vertex rejects requests above 20k tokens total; ~4 chars/token with headroom.
_REQUEST_CHAR_BUDGET = 60_000


def _batch_by_budget(idxs: list[int], texts: list[str], max_items: int) -> list[list[int]]:
    groups, cur, used = [], [], 0
    for i in idxs:
        n = min(len(texts[i]), 8000)
        if cur and (len(cur) >= max_items or used + n > _REQUEST_CHAR_BUDGET):
            groups.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += n
    if cur:
        groups.append(cur)
    return groups


def embed_docs(texts: list[str], model: str | None = None) -> np.ndarray:
    return embed_vertex(texts, "RETRIEVAL_DOCUMENT", model)


def embed_queries(texts: list[str], model: str | None = None) -> np.ndarray:
    return embed_vertex(texts, "RETRIEVAL_QUERY", model)
