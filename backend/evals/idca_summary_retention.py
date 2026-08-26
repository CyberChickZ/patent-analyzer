#!/usr/bin/env python3
"""IDCA gate: how much of the invention does the summary still carry?

The summary is the only thing Phase 2 sees besides the document, so the
question is whether the facts the claims are made of survive it. Gold =
the patent's own independent-claim limitations (Pap2Pat). A limitation is
"retained" when some sentence of the summary is within TAU cosine of it
(text-embedding-005, the production encoder). Claim language is patentese
and a summary is prose, so the absolute rate is low; what the gate reads is
the difference between two models on the same papers (rate@.70, rate@.60
and the mean best cosine are all reported).

    LLM_MODEL_IDCA=gemini-3.8-flash python3 evals/idca_summary_retention.py --limit 20

Writes eval_data/runs/j5/idca_retention_<model>.json and prints the table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "j5"
TAU = 0.70


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if len(p.strip()) > 25]


def retention(summary: str, limitations: list[str], tau: float = TAU) -> dict:
    import numpy as np
    from patent_analyzer.encoders import embed_docs, embed_queries
    sents = _sentences(summary)
    if not sents or not limitations:
        return {"n": len(limitations), "retained": 0, "rate": 0.0, "per_limitation": []}
    sv = np.asarray(embed_docs(sents), dtype=np.float32)
    lv = np.asarray(embed_queries(limitations), dtype=np.float32)
    sv /= np.linalg.norm(sv, axis=1, keepdims=True) + 1e-9
    lv /= np.linalg.norm(lv, axis=1, keepdims=True) + 1e-9
    sim = lv @ sv.T
    best = sim.max(axis=1)
    rows = [{"limitation": l[:120], "cos": round(float(c), 4), "retained": bool(c >= tau)}
            for l, c in zip(limitations, best)]
    return {"n": len(limitations), "retained": int((best >= tau).sum()),
            "rate": float((best >= tau).mean()), "rate_60": float((best >= 0.60).mean()),
            "mean_cos": float(best.mean()), "per_limitation": rows}


async def one_pair(pair: dict, data: Path) -> dict:
    from nodes.idca import idca_node
    from openworld_eval import render_paper
    from pap2pat_extraction_eval import gold_claims
    patent = json.loads((data / pair["pair_id"] / "patent.json").read_text())
    claims = gold_claims(patent)
    lims = [e["text"] for c in claims[:1] for e in c["elements"]]
    if not lims:
        return {}
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(render_paper(pair["pair_id"]))
        tmp = f.name
    try:
        out = await idca_node({"input_local_path": tmp, "input_mode": "academic_paper", "events": []})
    finally:
        os.unlink(tmp)
    summary = out.get("summary", "") or ""
    r = retention(summary, lims)
    return {"pair_id": pair["pair_id"], "summary_chars": len(summary), "status": out.get("status_determination"),
            "n_limitations": r["n"], "retained": r["retained"], "rate": r["rate"], "rate_60": r["rate_60"],
            "mean_cos": r["mean_cos"], "per_limitation": r["per_limitation"]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--model", default="", help="sets LLM_MODEL_IDCA for this run")
    a = ap.parse_args()
    if a.model:
        os.environ["LLM_MODEL_IDCA"] = a.model
    from common import load_env_yaml
    load_env_yaml()
    from app.llm import stage_model, usage
    model = stage_model("idca")
    from pap2pat_extraction_eval import ensure_data
    from openworld_eval import sample_pairs
    data = ensure_data()
    pairs = sample_pairs(50, seed=42)[:a.limit]
    rows = []
    for p in pairs:
        try:
            r = await one_pair(p, data)
        except Exception as exc:
            print(f"[{p['pair_id']}] FAILED {type(exc).__name__}: {exc}")
            continue
        if r:
            rows.append(r)
            print(f"{r['pair_id']:32s} limitations {r['n_limitations']:3d} retained {r['retained']:3d} "
                  f"rate@.70 {r['rate']:.3f}  rate@.60 {r['rate_60']:.3f}  mean cos {r['mean_cos']:.3f}  "
                  f"summary {r['summary_chars']:6d} chars")
    if rows:
        n = sum(r["n_limitations"] for r in rows)
        k = sum(r["retained"] for r in rows)
        print(f"\n== idca retention  model={model}  papers={len(rows)}  limitations={n}  "
              f"retained={k}  micro rate@.70={k / n:.3f}  macro rate@.70={sum(r['rate'] for r in rows) / len(rows):.3f}  "
              f"macro rate@.60={sum(r['rate_60'] for r in rows) / len(rows):.3f}  mean cos={sum(r['mean_cos'] for r in rows) / len(rows):.3f}  "
              f"mean summary={sum(r['summary_chars'] for r in rows) // len(rows)} chars  (tau={TAU})")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / f"idca_retention_{model}.json").write_text(json.dumps(
        {"model": model, "tau": TAU, "rows": rows, "llm_usage": {m: dict(u) for m, u in usage.items()}}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
