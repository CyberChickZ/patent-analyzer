"""Disk cache + call meter for Gemini calls during evals.

install() monkeypatches app.llm.call_llm / call_llm_with_pdfs so every
higher-level function (detect_and_summarize_invention, SSR nodes, evaluate_*)
hits the cache transparently. Re-scoring a finished run costs $0; the meter
reports how many calls actually went to Vertex.
"""

import hashlib
import json
import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR",
                                Path(__file__).parent.parent / "eval_data" / ".llm_cache"))

stats = {"hits": 0, "misses": 0, "chars_in": 0, "chars_out": 0}


def _key(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\x00")
    return h.hexdigest()


def _file_sha(path: str) -> str:
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "missing"


def install():
    import app.llm as llm

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    real_text, real_pdfs = llm.call_llm, llm.call_llm_with_pdfs

    async def cached_text(system, user, max_tokens=llm.MAX_TOKENS, thinking_budget=0, response_schema=None):
        # schema-less calls keep their historical key (cached IDCA / extraction stay valid)
        extra = [json.dumps(response_schema, sort_keys=True)] if response_schema else []
        p = CACHE_DIR / (_key("text", llm.MODEL, system, user, max_tokens, thinking_budget, *extra) + ".json")
        if p.exists():
            stats["hits"] += 1
            return json.loads(p.read_text())["text"]
        out = await real_text(system, user, max_tokens, thinking_budget, response_schema=response_schema)
        stats["misses"] += 1
        stats["chars_in"] += len(system) + len(user)
        stats["chars_out"] += len(out)
        p.write_text(json.dumps({"text": out}))
        return out

    async def cached_pdfs(system, user, pdf_paths, max_tokens=llm.MAX_TOKENS,
                          thinking_budget=0, image_parts=None):
        shas = [_file_sha(x) for x in pdf_paths]
        img = [hashlib.sha1(b).hexdigest() for b in (image_parts or [])]
        p = CACHE_DIR / (_key("pdfs", llm.MODEL, system, user, shas, img, max_tokens, thinking_budget) + ".json")
        if p.exists():
            stats["hits"] += 1
            return json.loads(p.read_text())["text"]
        out = await real_pdfs(system, user, pdf_paths, max_tokens, thinking_budget, image_parts)
        stats["misses"] += 1
        stats["chars_in"] += len(system) + len(user)
        stats["chars_out"] += len(out)
        p.write_text(json.dumps({"text": out}))
        return out

    llm.call_llm = cached_text
    llm.call_llm_with_pdfs = cached_pdfs
    llm.call_llm_with_pdf = lambda s, u, path, m=llm.MAX_TOKENS, t=0: cached_pdfs(s, u, [path], m, t)


def summary() -> str:
    return (f"llm calls: {stats['misses']} live, {stats['hits']} cached; "
            f"~{stats['chars_in'] // 4 // 1000}k tok in / ~{stats['chars_out'] // 4 // 1000}k tok out (live only)")
