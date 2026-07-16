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
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

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


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_rank(query: str, docs: list[str], k1=1.5, b=0.75) -> np.ndarray:
    """Return score per doc. Plain BM25, no deps."""
    doc_tokens = [_tokens(d) for d in docs]
    doc_lens = np.array([len(t) for t in doc_tokens], dtype=float)
    avgdl = doc_lens.mean() if len(doc_lens) else 1.0
    df = Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    n = len(docs)
    tfs = [Counter(toks) for toks in doc_tokens]
    scores = np.zeros(n)
    for term in set(_tokens(query)):
        if term not in df:
            continue
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        tf = np.array([t[term] for t in tfs], dtype=float)
        scores += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_lens / avgdl))
    return scores


def eval_embedding(samples, corpus, mode: str) -> dict:
    from patent_analyzer.semantic_search import embed_texts

    pubs = list(corpus.keys())
    all_chunks, owner = [], []
    for i, pub in enumerate(pubs):
        for ch in doc_chunks(corpus[pub], mode):
            all_chunks.append(ch)
            owner.append(i)
    owner = np.array(owner)
    print(f"[{mode}] embedding {len(all_chunks)} chunks for {len(pubs)} docs...")
    doc_vecs = embed_texts(all_chunks)
    query_vecs = embed_texts([s["query_claim"] for s in samples])

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
        scores = bm25_rank(s["query_claim"], docs)
        order = np.argsort(scores)[::-1]
        target_idx = pubs.index(s["target"])
        ranks.append(int(np.where(order == target_idx)[0][0]) + 1)
    return summarize(ranks)


def eval_hybrid(samples, corpus, k_rrf: int = 60) -> dict:
    """RRF fusion of BM25 and claims_chunks dense ranks."""
    from patent_analyzer.semantic_search import embed_texts

    pubs = list(corpus.keys())
    docs_text = [" ".join(doc_chunks(corpus[p], "claims_chunks")) for p in pubs]
    all_chunks, owner = [], []
    for i, pub in enumerate(pubs):
        for ch in doc_chunks(corpus[pub], "claims_chunks"):
            all_chunks.append(ch)
            owner.append(i)
    owner = np.array(owner)
    doc_vecs = embed_texts(all_chunks)
    query_vecs = embed_texts([s["query_claim"] for s in samples])

    ranks = []
    for qi, s in enumerate(samples):
        sims = doc_vecs @ query_vecs[qi]
        dense = np.full(len(pubs), -1.0)
        np.maximum.at(dense, owner, sims)
        sparse = bm25_rank(s["query_claim"], docs_text)
        rrf = np.zeros(len(pubs))
        for scores in (dense, sparse):
            order = np.argsort(scores)[::-1]
            pos = np.empty(len(pubs), dtype=int)
            pos[order] = np.arange(len(pubs))
            rrf += 1.0 / (k_rrf + pos + 1)
        order = np.argsort(rrf)[::-1]
        target_idx = pubs.index(s["target"])
        ranks.append(int(np.where(order == target_idx)[0][0]) + 1)
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
                    choices=["all", "bm25", "title_abstract", "claims_chunks", "hybrid"])
    args = ap.parse_args()

    samples = load_samples(args.limit, args.split)
    corpus = build_corpus(samples)
    print(f"samples={len(samples)}  corpus={len(corpus)} unique prior-art patents\n")

    results = {}
    if args.mode in ("all", "bm25"):
        results["bm25_keyword"] = eval_bm25(samples, corpus)
    if args.mode in ("all", "title_abstract"):
        results["minilm_title_abstract"] = eval_embedding(samples, corpus, "title_abstract")
    if args.mode in ("all", "claims_chunks"):
        results["minilm_claims_chunks"] = eval_embedding(samples, corpus, "claims_chunks")
    if args.mode in ("all", "hybrid"):
        results["hybrid_rrf"] = eval_hybrid(samples, corpus)

    print(f"\n{'method':<28}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'R@50':>7}{'MRR':>8}{'medR':>7}")
    for name, m in results.items():
        print(f"{name:<28}{m['recall@1']:>7.3f}{m['recall@5']:>7.3f}"
              f"{m['recall@10']:>7.3f}{m['recall@50']:>7.3f}{m['mrr']:>8.3f}{m['median_rank']:>7.0f}")


if __name__ == "__main__":
    main()
