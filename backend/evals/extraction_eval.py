#!/usr/bin/env python3
"""Extraction-stage eval on FiNE-Patents.

Task: given claim 1 of the rejected application, split it into features.
Ground truth: the examiner's feature breakdown from the ESOP annotations.
Matching: greedy 1:1 pairing by embedding similarity; precision/recall/F1
reported at several thresholds so no single cutoff decides the story.

Usage:
    python3 evals/extraction_eval.py --limit 100 --mode regex
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "eval_data" / "fine-patents" / "data" / "packaged"
CACHE_DIR = Path(__file__).parent.parent / "eval_data" / ".emb_cache"


def load_samples(limit: int | None):
    samples = []
    for app_dir in sorted(DATA_DIR.iterdir()):
        if not app_dir.is_dir():
            continue
        try:
            bd = json.loads((app_dir / "breakdown.json").read_text())
            rej = json.loads((app_dir / "rejected_patent.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if bd.get("language") != "EN":
            continue
        claims = rej.get("claims") or []
        features = [f.get("feature", "").strip() for f in (bd.get("breakdown") or [])]
        features = [f for f in features if len(f) > 15]
        if not claims or not isinstance(claims[0], str) or len(features) < 2:
            continue
        samples.append({"app": app_dir.name, "claim": claims[0], "features": features})
        if limit and len(samples) >= limit:
            break
    return samples


def predict_regex(claim: str) -> list[str]:
    from nodes.claim_mode import _parse_claim_limitations, _split_preamble
    parsed = _parse_claim_limitations(claim)
    preds = _split_preamble(parsed.get("preamble", "")) + parsed["limitations"]
    return [p for p in preds if len(p.strip()) > 15]


def embed(texts: list[str]) -> np.ndarray:
    from patent_analyzer.encoders import embed_vertex
    return embed_vertex(texts, "SEMANTIC_SIMILARITY", cache_dir=CACHE_DIR)


def greedy_match(sim: np.ndarray) -> list[tuple[int, int, float]]:
    """Greedy 1:1 pairing, best similarity first."""
    pairs = [(sim[i, j], i, j) for i in range(sim.shape[0]) for j in range(sim.shape[1])]
    pairs.sort(reverse=True)
    used_i, used_j, out = set(), set(), []
    for s, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--mode", default="regex", choices=["regex"])
    from common import add_budget_arg, arm_budget
    add_budget_arg(ap)
    args = ap.parse_args()
    arm_budget(args)

    samples = load_samples(args.limit)
    n_feat = sum(len(s["features"]) for s in samples)
    print(f"samples={len(samples)}  examiner features={n_feat} "
          f"(avg {n_feat / max(len(samples), 1):.1f}/claim)\n")

    thresholds = [0.7, 0.8, 0.9]
    agg = {t: {"tp": 0, "n_gold": 0, "n_pred": 0} for t in thresholds}
    n_pred_total = 0

    for s in samples:
        preds = predict_regex(s["claim"])
        n_pred_total += len(preds)
        if not preds:
            for t in thresholds:
                agg[t]["n_gold"] += len(s["features"])
            continue
        gold_vecs = embed(s["features"])
        pred_vecs = embed(preds)
        sim = gold_vecs @ pred_vecs.T
        matches = greedy_match(sim)
        for t in thresholds:
            tp = sum(1 for _, _, sc in matches if sc >= t)
            agg[t]["tp"] += tp
            agg[t]["n_gold"] += len(s["features"])
            agg[t]["n_pred"] += len(preds)

    print(f"predicted features: avg {n_pred_total / max(len(samples), 1):.1f}/claim "
          f"({args.mode})\n")
    print(f"{'threshold':<12}{'recall':>8}{'precision':>11}{'F1':>7}")
    for t in thresholds:
        a = agg[t]
        r = a["tp"] / a["n_gold"] if a["n_gold"] else 0.0
        p = a["tp"] / a["n_pred"] if a["n_pred"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{t:<12}{r:>8.3f}{p:>11.3f}{f1:>7.3f}")


if __name__ == "__main__":
    main()
