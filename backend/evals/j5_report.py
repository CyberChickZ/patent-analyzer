#!/usr/bin/env python3
"""Print the J5 model × stage tables from eval_data/runs/j5/*.json (no LLM calls)."""

import json
import re
from pathlib import Path

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "j5"


def _num(line: str, key: str):
    m = re.search(rf"{re.escape(key)}=([0-9.]+)", line)
    return float(m.group(1)) if m else None


def _row_after(lines: list[str], prefix: str) -> str:
    for l in lines:
        if l.strip().startswith(prefix):
            return l
    return ""


def extract_table():
    rows = []
    for f in sorted(RUN_DIR.glob("extract_*.json")):
        d = json.loads(f.read_text())
        m = d["meter"]
        fine, p2p = d.get("fine_stdout") or [], d.get("p2p_stdout") or []
        head = fine[0] if fine else ""
        f07 = _row_after(fine, "0.7")
        f08 = _row_after(fine, "0.8")
        err = next((l for l in fine if "errors:" in l), "")
        p2p_mean = next((l for l in p2p if l.strip().startswith("mean")), "")
        rows.append({
            "model": m["model"], "n": head, "f1_07": f07.split(), "f1_08": f08.split(), "errors": err.strip(),
            "calls_line": next((l for l in fine if "llm calls/sample" in l), "").strip(),
            "p2p_mean": p2p_mean.strip(), "fine_unsup": d["fine_unsupported"], "p2p_unsup": d["p2p_unsupported"],
            "retries": d.get("fine_retries"), "meter": m, "log429": None,
        })
    return rows


def main():
    print("## extract")
    for r in extract_table():
        m = r["meter"]
        print(f"### {r['model']}")
        print(f"  {r['n']}")
        print(f"  F1@0.7 (recall precision F1): {r['f1_07'][1:]}   F1@0.8: {r['f1_08'][1:]}")
        print(f"  {r['errors']}")
        print(f"  {r['calls_line']}")
        print(f"  pap2pat: {r['p2p_mean']}")
        print(f"  unsupported FiNE {r['fine_unsup']}  p2p {r['p2p_unsup']}  retries={r['retries']}")
        print(f"  meter: calls={m['calls']} 429={m['errors_429']} in={m['prompt_tokens']} out={m['output_tokens']} "
              f"thought={m['thought_tokens']} s/doc={m['seconds_per_doc']} $/doc={m['cost_per_doc']} maxtok={m['max_tokens_hits']} "
              f"other={m['other_stage_calls']}")
    print("\n## eval")
    for f in sorted(RUN_DIR.glob("eval_gemini-*.json")):
        d = json.loads(f.read_text())
        m = d["meter"]
        print(f"### {m['model']}")
        for k, v in (d.get("summary") or {}).items():
            fp, fr, ff = v["feature"]
            cp, cr, cf = v["claim"]
            q = v["quotes"]
            print(f"  {k}: feat P/R/F1 {fp:.3f}/{fr:.3f}/{ff:.3f}  claim P/R/F1 {cp:.3f}/{cr:.3f}/{cf:.3f}  "
                  f"quotes {q['passed']}/{q['quotes']} survive, q/item {q['quotes'] / max(q['positive'], 1):.2f}, downgraded {q.get('downgraded', 0)}/{q['positive']}")
        print(f"  meter: calls={m['calls']} 429={m['errors_429']} in={m['prompt_tokens']} out={m['output_tokens']} "
              f"thought={m['thought_tokens']} s/doc={m['seconds_per_doc']} $/doc={m['cost_per_doc']} maxtok={m['max_tokens_hits']}")
    print("\n## screen")
    for f in sorted(RUN_DIR.glob("screen_*_h1d.json")):
        d = json.loads(f.read_text())
        m, a = d["meter"], d["agg"]
        print(f"### {m['model']}: papers={a['papers']} gold_in={a['gold_in']} gold_worth={a['gold_worth']} gold_kept60={a['gold_kept60']} "
              f"(recorded 2.5-pro run: worth {a['recorded_gold_worth']} kept {a['recorded_gold_kept60']}) worth_total={a['worth_total']} "
              f"unanswered={a['unanswered']} wall/paper={a['wall_seconds_per_paper']}s")
        print(f"  meter: calls={m['calls']} 429={m['errors_429']} in={m['prompt_tokens']} out={m['output_tokens']} "
              f"thought={m['thought_tokens']} s(serial)/paper={m['seconds_per_doc']} $/paper={m['cost_per_doc']} maxtok={m['max_tokens_hits']}")
        for r in d["rows"]:
            print(f"    {r['key']}: {r['n_docs']} docs → worth {r['worth']} kept {r['kept']} unanswered {r['unanswered']} wall {r['wall_seconds']}s")


if __name__ == "__main__":
    main()
