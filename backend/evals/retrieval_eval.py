#!/usr/bin/env python3
"""Stage-level retrieval eval on FiNE-Patents.

Task: query = rejected application's claim 1; corpus = all cited (prior art)
patents in the sample set; target = the examiner-cited closest prior art.
Metrics: Recall@k, MRR. Compares doc representations and a BM25 baseline
so retrieval decisions are made on measurements, not vibes.

Usage:
    python3 evals/retrieval_eval.py --limit 500 --mode all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.hybrid import bm25_scores, rrf_fuse  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "eval_data" / "fine-patents" / "data" / "packaged"


def load_samples(limit: int | None, split: str | None):
    samples = []
    for app_dir in sorted(DATA_DIR.iterdir()):
        if not app_dir.is_dir():
            continue
        try:
            meta = json.loads((app_dir / "metadata.json").read_text())
            if split and meta.get("split") != split:
                continue
            rej = json.loads((app_dir / "rejected_patent.json").read_text())
            cit = json.loads((app_dir / "cited_patent.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        claims = rej.get("claims") or []
        if not claims or not cit.get("publication_number"):
            continue
        samples.append({
            "app": app_dir.name,
            "query_claim": claims[0] if isinstance(claims[0], str) else str(claims[0]),
            "target": cit["publication_number"],
            "cited": cit,
        })
        if limit and len(samples) >= limit:
            break
    return samples


def build_corpus(samples):
    corpus = {}
    for s in samples:
        c = s["cited"]
        pub = c["publication_number"]
        if pub not in corpus:
            corpus[pub] = c
    return corpus


def doc_chunks(doc: dict, mode: str) -> list[str]:
    title = doc.get("title") or ""
    abstract = doc.get("abstract") or ""
    if mode == "title_abstract":
        return [f"{title} {abstract}".strip()]
    if mode == "claims_chunks":
        chunks = [f"{title} {abstract}".strip()]
        for cl in (doc.get("claims") or [])[:30]:
            if isinstance(cl, str) and len(cl.strip()) > 30:
                chunks.append(cl.strip())
        return chunks
    raise ValueError(mode)


CACHE_DIR = Path(__file__).parent.parent / "eval_data" / ".emb_cache"


def embed_vertex(texts: list[str], model: str, task_type: str) -> np.ndarray:
    """Shared with the production rerank path (patent_analyzer.encoders) so
    the eval measures exactly the code that serves."""
    from patent_analyzer.encoders import embed_vertex as _shared
    return _shared(texts, task_type, model, cache_dir=CACHE_DIR)


def make_embed_fns(model: str):
    """Return (embed_docs, embed_queries) for a model name."""
    if model == "minilm":
        from patent_analyzer.semantic_search import embed_texts
        return embed_texts, embed_texts
    return (lambda t: embed_vertex(t, model, "RETRIEVAL_DOCUMENT"),
            lambda t: embed_vertex(t, model, "RETRIEVAL_QUERY"))


def eval_embedding(samples, corpus, mode: str, model: str = "minilm") -> dict:
    embed_docs, embed_queries = make_embed_fns(model)

    pubs = list(corpus.keys())
    all_chunks, owner = [], []
    for i, pub in enumerate(pubs):
        for ch in doc_chunks(corpus[pub], mode):
            all_chunks.append(ch)
            owner.append(i)
    owner = np.array(owner)
    print(f"[{mode}/{model}] embedding {len(all_chunks)} chunks for {len(pubs)} docs...")
    doc_vecs = embed_docs(all_chunks)
    query_vecs = embed_queries([s["query_claim"] for s in samples])

    ranks = []
    for qi, s in enumerate(samples):
        sims = doc_vecs @ query_vecs[qi]
        doc_scores = np.full(len(pubs), -1.0)
        np.maximum.at(doc_scores, owner, sims)  # max-pool over chunks
        order = np.argsort(doc_scores)[::-1]
        target_idx = pubs.index(s["target"])
        ranks.append(int(np.where(order == target_idx)[0][0]) + 1)
    return summarize(ranks)


def eval_bm25(samples, corpus) -> dict:
    pubs = list(corpus.keys())
    docs = [" ".join(doc_chunks(corpus[p], "claims_chunks")) for p in pubs]
    ranks = []
    for s in samples:
        scores = bm25_scores(s["query_claim"], docs)
        order = np.argsort(scores)[::-1]
        target_idx = pubs.index(s["target"])
        ranks.append(int(np.where(order == target_idx)[0][0]) + 1)
    return summarize(ranks)


def eval_hybrid(samples, corpus, k_rrf: int = 60, model: str = "minilm") -> dict:
    """RRF fusion of BM25 and claims_chunks dense ranks."""
    embed_docs, embed_queries = make_embed_fns(model)

    pubs = list(corpus.keys())
    docs_text = [" ".join(doc_chunks(corpus[p], "claims_chunks")) for p in pubs]
    all_chunks, owner = [], []
    for i, pub in enumerate(pubs):
        for ch in doc_chunks(corpus[pub], "claims_chunks"):
            all_chunks.append(ch)
            owner.append(i)
    owner = np.array(owner)
    doc_vecs = embed_docs(all_chunks)
    query_vecs = embed_queries([s["query_claim"] for s in samples])

    ranks = []
    for qi, s in enumerate(samples):
        sims = doc_vecs @ query_vecs[qi]
        dense = np.full(len(pubs), -1.0)
        np.maximum.at(dense, owner, sims)
        sparse = bm25_scores(s["query_claim"], docs_text)
        order = np.argsort(rrf_fuse([dense, sparse], k=k_rrf))[::-1]
        target_idx = pubs.index(s["target"])
        ranks.append(int(np.where(order == target_idx)[0][0]) + 1)
    return summarize(ranks)


def eval_pipeline(samples, corpus) -> dict:
    """Parity check: run the production rerank_docs end-to-end on the eval
    corpus, legacy-doc dicts and all. Measures what actually serves."""
    import os
    os.environ.setdefault("EMBED_CACHE_DIR", str(CACHE_DIR / "pipeline"))
    from patent_analyzer.semantic_search import rerank_docs

    pubs = list(corpus.keys())
    docs = []
    for pub in pubs:
        c = corpus[pub]
        claims = [cl for cl in (c.get("claims") or []) if isinstance(cl, str)]
        docs.append({
            "title": c.get("title") or "",
            "abstract": c.get("abstract") or "",
            "pub_num": pub,
            "claims_text": " ".join(claims),
        })

    ranks = []
    for s in samples:
        ranked = rerank_docs(s["query_claim"], docs, limit=len(docs))
        order = [d["pub_num"] for d in ranked]
        ranks.append(order.index(s["target"]) + 1)
    return summarize(ranks)


def summarize(ranks: list[int]) -> dict:
    r = np.array(ranks, dtype=float)
    return {
        "n": len(ranks),
        "recall@1": float((r <= 1).mean()),
        "recall@5": float((r <= 5).mean()),
        "recall@10": float((r <= 10).mean()),
        "recall@50": float((r <= 50).mean()),
        "mrr": float((1.0 / r).mean()),
        "median_rank": float(np.median(r)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--split", default=None, help="train/validation/test; default all")
    ap.add_argument("--mode", default="all",
                    choices=["all", "bm25", "title_abstract", "claims_chunks", "hybrid", "pipeline"])
    ap.add_argument("--model", default="minilm",
                    choices=["minilm", "text-embedding-005", "gemini-embedding-001"])
    args = ap.parse_args()

    samples = load_samples(args.limit, args.split)
    corpus = build_corpus(samples)
    print(f"samples={len(samples)}  corpus={len(corpus)} unique prior-art patents\n")

    results = {}
    if args.mode in ("all", "bm25"):
        results["bm25_keyword"] = eval_bm25(samples, corpus)
    tag = args.model.replace("text-embedding-", "te").replace("gemini-embedding-", "ge")
    if args.mode in ("all", "title_abstract"):
        results[f"{tag}_title_abstract"] = eval_embedding(samples, corpus, "title_abstract", args.model)
    if args.mode in ("all", "claims_chunks"):
        results[f"{tag}_claims_chunks"] = eval_embedding(samples, corpus, "claims_chunks", args.model)
    if args.mode in ("all", "hybrid"):
        results[f"hybrid_rrf_{tag}"] = eval_hybrid(samples, corpus, model=args.model)
    if args.mode == "pipeline":
        results["pipeline_rerank_docs"] = eval_pipeline(samples, corpus)

    print(f"\n{'method':<28}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'R@50':>7}{'MRR':>8}{'medR':>7}")
    for name, m in results.items():
        print(f"{name:<28}{m['recall@1']:>7.3f}{m['recall@5']:>7.3f}"
              f"{m['recall@10']:>7.3f}{m['recall@50']:>7.3f}{m['mrr']:>8.3f}{m['median_rank']:>7.0f}")


if __name__ == "__main__":
    main()
