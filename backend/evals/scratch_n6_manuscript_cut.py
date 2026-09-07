#!/usr/bin/env python3
"""N6 audit: what the manuscript adapter actually cuts out of the 8 §H1.7.1 papers.

One LLM call per paper (app.llm.classify_prior_art_sections, through llm_cache),
then print every section the classifier did not call own_work, with its reason,
and how much text the cut removed. No extraction, no search.

    python3 evals/scratch_n6_manuscript_cut.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from common import load_env_yaml

load_env_yaml()

import llm_cache  # noqa: E402
from patent_analyzer.adapters.manuscript import (  # noqa: E402
    ADAPTER_VERSION, cut_nested, outline_nested,
)
from patent_analyzer.adapters.paper import doc_from_sections, render_doc  # noqa: E402

DATA = Path("/tmp/pap2pat/Pap2Pat/data")
PAIRS = ["W1516158759-US20120194631", "W1592449185-US20150126796", "W1983490683-US20120058475",
         "W1989556855-US20120144509", "W1997025766-US20150313878", "W2043323285-US20170089878",
         "W2044092945-US20100036217", "W2069072790-US20090304751"]
OUT = Path(__file__).parent.parent / "eval_data" / "runs" / "p2p_extract" / f"n6_cut_audit_{ADAPTER_VERSION}.json"


async def main():
    llm_cache.install()
    report = {}
    for pid in PAIRS:
        paper = json.loads((DATA / pid / "paper.json").read_text())
        doc = doc_from_sections(paper.get("title", ""), paper.get("abstract", ""), paper.get("sections"))
        items = outline_nested(doc)
        before = len(render_doc(doc))
        cut, verdicts = await cut_nested(doc)
        after = len(render_doc(cut))
        print(f"\n{'=' * 100}\n{pid}  {paper.get('title', '')[:80]}")
        print(f"  {before} -> {after} chars  (-{100 * (before - after) / before:.1f}%)  "
              f"sections {len(items)} judged {len(verdicts)}  "
              f"dropped sections {len(cut['dropped_sections'])}  dropped paragraphs {len(cut['dropped_paragraphs'])}")
        by_id = {i["id"]: i for i in items}
        for sid, v in verdicts.items():
            if v["verdict"] == "own_work":
                continue
            it = by_id[sid]
            tag = "DROP-ALL " if v["verdict"] == "prior_art" else f"DROP P{v['prior_art_paragraphs']} of {len(it['paragraphs'])} "
            print(f"  [{sid}] {it['heading'][:60]!r}  {tag}\n      {v['reason']}")
            for n in (range(1, len(it["paragraphs"]) + 1) if v["verdict"] == "prior_art" else v["prior_art_paragraphs"]):
                if 1 <= n <= len(it["paragraphs"]):
                    print(f"      - P{n}: {it['paragraphs'][n - 1][:160]}")
        kept = [f"{i['id']} {i['heading']}" for i in items
                if (verdicts.get(i["id"]) or {}).get("verdict") != "prior_art"]
        report[pid] = {"before": before, "after": after, "verdicts": verdicts,
                       "dropped_sections": cut["dropped_sections"], "dropped_paragraphs": cut["dropped_paragraphs"],
                       "kept_sections": kept}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("\n" + llm_cache.summary())
    print("written:", OUT)


if __name__ == "__main__":
    asyncio.run(main())
