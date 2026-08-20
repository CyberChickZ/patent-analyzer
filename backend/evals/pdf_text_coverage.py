#!/usr/bin/env python3
"""J4: how much of a paper does each PDF text layer keep?

Gold = Pap2Pat paper.json paragraphs (GROBID body text). A paragraph is
"kept" by a text layer when quote_verify.locate_quote finds it (difflib
local alignment >= 0.9, space-insensitive second pass). Layers:

  fitz_pdf_text   patent_analyzer.quote_verify.pdf_text (Phase 4 quote check)
  fitz_extract    app.llm._extract_pdf_text(max_pages=80, 150k) (Phase 2 extraction)
  grobid          TEI from a GROBID server (eval_data/runs/j4/grobid/<pair>.tei.xml)
  doc_json        Gemini Doc JSON (eval_data/runs/j4/doc_json/<pair>.json)

Usage: python3 evals/pdf_text_coverage.py [--pairs a,b] [--layers fitz_pdf_text,...]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from patent_analyzer.quote_verify import locate_quote, normalize

PAP2PAT = Path(os.environ.get("PAP2PAT_DIR", "/tmp/pap2pat/Pap2Pat/data"))
PDF_DIR = Path(__file__).parent.parent / "eval_data" / "pdfs" / "j4"
RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "j4"
MIN_PARA = 20
CHUNK = 100


def paper_paragraphs(pair_id: str) -> list[dict]:
    """Abstract + every body paragraph of paper.json with its section path."""
    paper = json.loads((PAP2PAT / pair_id / "paper.json").read_text())
    out = [{"section": "Abstract", "text": paper.get("abstract") or ""}]

    def walk(secs, path):
        for s in secs or []:
            title = s.get("title") or ""
            for p in s.get("paragraphs") or []:
                if p:
                    out.append({"section": " / ".join(path + [title]), "text": p})
            walk(s.get("subsections"), path + [title])
    walk(paper.get("sections"), [])
    return [p for p in out if len(normalize(p["text"])) >= MIN_PARA]


def chunk_coverage(para: str, doc_norm: str) -> float:
    """Share of ~100-char normalized, space-free chunks of the paragraph found verbatim."""
    q = normalize(para).replace(" ", "")
    chunks = [q[i:i + CHUNK] for i in range(0, len(q), CHUNK)]
    chunks = [c for c in chunks if len(c) >= 30]
    if not chunks:
        return 1.0 if q in doc_norm else 0.0
    return sum(1 for c in chunks if c in doc_norm) / len(chunks)


def coverage(paragraphs: list[dict], text: str) -> dict:
    doc_norm = normalize(text).replace(" ", "")
    rows = []
    for p in paragraphs:
        found, sim = locate_quote(p["text"], text, 0.9)
        rows.append({"section": p["section"], "found": found, "sim": sim,
                     "chunks": round(chunk_coverage(p["text"], doc_norm), 3),
                     "head": p["text"][:80]})
    n = len(rows)
    hit = sum(r["found"] for r in rows)
    return {"n": n, "hit": hit, "missed": n - hit,
            "para_cov": round(hit / n, 4) if n else None,
            "chunk_cov": round(sum(r["chunks"] for r in rows) / n, 4) if n else None,
            "chars": len(text), "rows": rows}


# ---- text layers -----------------------------------------------------------

def layer_fitz_pdf_text(pdf: Path) -> str:
    from patent_analyzer.quote_verify import pdf_text
    return pdf_text(str(pdf))


def layer_fitz_extract(pdf: Path) -> str:
    from app.llm import _EXTRACTION_DOC_CAP, _extract_pdf_text
    return _extract_pdf_text(str(pdf), max_pages=80, max_chars=_EXTRACTION_DOC_CAP)


def layer_doc_json(pdf: Path) -> str | None:
    """Gemini Doc JSON (app.llm.build_doc_json) rendered by the production adapter,
    without [S.P] markers. Written by `--build_doc_json` (one Gemini call per PDF)."""
    from patent_analyzer.adapters.docjson import render_doc_json
    p = RUN_DIR / "doc_json" / (pdf.stem + ".json")
    if not p.exists():
        return None
    return render_doc_json(json.loads(p.read_text()), markers=False)


async def build_doc_jsons(pdfs: list[Path]) -> None:
    import llm_cache
    from app.llm import build_doc_json
    llm_cache.install()
    out_dir = RUN_DIR / "doc_json"
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        p = out_dir / (pdf.stem + ".json")
        if p.exists():
            continue
        doc = await build_doc_json("", source_pdf_path=str(pdf))
        if doc is None:
            print(f"{pdf.stem}: Doc JSON call failed")
            continue
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        print(f"{pdf.stem}: {len(doc['sections'])} sections, {sum(len(s['paragraphs']) for s in doc['sections'])} paragraphs, "
              f"{len(doc['figures'])} figures, {len(doc['equations'])} equations")
    print(llm_cache.summary())


_TEI_TAG = re.compile(r"<[^>]+>")


def render_grobid_tei(xml: str) -> str:
    """Abstract, body and back-matter (annex / acknowledgement divs, not the
    bibliography) <p>/<head>/<figDesc>/<formula> text of a GROBID TEI document."""
    body = xml.split("<body>", 1)[1].split("</body>", 1)[0] if "<body>" in xml else xml
    abstract = xml.split("<abstract>", 1)[1].split("</abstract>", 1)[0] if "<abstract>" in xml else ""
    back = xml.split("<back>", 1)[1].split("</back>", 1)[0] if "<back>" in xml else ""
    back = "".join(m.group(0) for m in re.finditer(r'<div type="(?:annex|acknowledgement)".*?</div>', back, re.S))
    parts = []
    for m in re.finditer(r"<(head|p|figDesc|formula)\b[^>]*>(.*?)</\1>", abstract + body + back, re.S):
        t = _TEI_TAG.sub("", m.group(2))
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            parts.append(t)
    return "\n".join(parts) + "\n"


def layer_grobid(pdf: Path) -> str | None:
    p = RUN_DIR / "grobid" / (pdf.stem + ".tei.xml")
    if not p.exists():
        return None
    return render_grobid_tei(p.read_text())


LAYERS = {"fitz_pdf_text": layer_fitz_pdf_text, "fitz_extract": layer_fitz_extract,
          "grobid": layer_grobid, "doc_json": layer_doc_json}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="")
    ap.add_argument("--layers", default=",".join(LAYERS))
    ap.add_argument("--out", default=str(RUN_DIR / "coverage.json"))
    ap.add_argument("--build_doc_json", action="store_true", help="call Gemini for missing Doc JSONs first")
    args = ap.parse_args()
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.pairs:
        keep = set(args.pairs.split(","))
        pdfs = [p for p in pdfs if p.stem in keep]
    layers = [l for l in args.layers.split(",") if l in LAYERS]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if args.build_doc_json:
        import asyncio
        asyncio.run(build_doc_jsons(pdfs))

    result = {}
    print(f"{'pair':28} {'paras':>5} " + " ".join(f"{l:>22}" for l in layers))
    for pdf in pdfs:
        paras = paper_paragraphs(pdf.stem)
        row = {"n_paras": len(paras)}
        cells = []
        for l in layers:
            text = LAYERS[l](pdf)
            if text is None:
                cells.append(f"{'-':>22}")
                continue
            cov = coverage(paras, text)
            row[l] = cov
            cells.append(f"{cov['hit']:>3}/{cov['n']:<3} {cov['para_cov']:.2f} ch{cov['chunk_cov']:.2f} {cov['chars'] // 1000:>3}k")
        result[pdf.stem] = row
        print(f"{pdf.stem:28} {len(paras):>5} " + " ".join(cells))
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    tot = {l: [sum(r[l]["hit"] for r in result.values() if l in r), sum(r[l]["n"] for r in result.values() if l in r)]
           for l in layers}
    print("TOTAL " + " ".join(f"{l}={h}/{n} ({h / n:.3f})" for l, (h, n) in tot.items() if n))


if __name__ == "__main__":
    main()
