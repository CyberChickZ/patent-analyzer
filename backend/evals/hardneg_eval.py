#!/usr/bin/env python3
"""E3: half-open retrieval eval — FiNE queries vs a hard-negative pool.

The closed 200-doc arena in retrieval_eval.py only ranks examiner-cited
patents against each other. Here every query (rejected claim 1) is ranked
over a pool built like the self-supervised EPO recipe (arXiv 2511.10657):
gold = examiner-cited closest prior art, plus per query ~80 lexical
more-like-this negatives and ~80 same-CPC-subclass / same-era negatives
drawn from our own BigQuery copy (amie_patents.pubs). All docs, gold
included, are title+abstract so the representation is identical.

Usage:
    python3 evals/hardneg_eval.py --limit 200 --buckets 200
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.common import norm_pub  # noqa: E402
from evals.retrieval_eval import DATA_DIR, doc_chunks, embed_vertex, load_samples  # noqa: E402
from patent_analyzer.hybrid import bm25_scores, tokenize  # noqa: E402
from patent_analyzer.recall.bigquery_patents import GC_PROJECT, guarded_query  # noqa: E402

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "e3"
STOP = set("""a an the of in on for and or by to with from is at as its via using based method system
apparatus device comprising wherein step configured adapted claim claims said which that each having
being one first second third least plurality thereof into between such than other about further whereby
when where means unit units portion portions according includes including provided wherein there their
this these those form forms formed member members part parts arranged along through over under after
before within without above below least more less same different respective respectively""".split())

BQ_BILLED = []


def _bq_form(p: str) -> str:
    p = norm_pub(p)
    m = re.match(r"^([A-Z]{2})(\d+)([A-Z]\d?)?$", p)
    if not m:
        return p
    cc, digits, kind = m.group(1), m.group(2), m.group(3) or ""
    if cc == "US" and len(digits) == 11 and digits[4] == "0":
        digits = digits[:4] + digits[5:]
    return f"{cc}-{digits}-{kind}".rstrip("-")


def query_terms(claim: str, n: int = 8) -> list[str]:
    toks = [t for t in tokenize(claim) if t not in STOP and len(t) > 3 and t.isalpha()]
    cnt = collections.Counter(toks)
    return [w for w, _ in sorted(cnt.items(), key=lambda kv: (-len(kv[0]), -kv[1], kv[0]))[:n]]


def _client():
    from google.cloud import bigquery
    return bigquery.Client(project=GC_PROJECT)


def _run(client, sql, params, max_gib):
    rows = guarded_query(client, sql, params, max_gib=max_gib)
    for job in client.list_jobs(max_results=10):
        if job.job_type == "query" and not job.dry_run and job.query == sql:
            BQ_BILLED.append((job.total_bytes_billed or 0) / 2 ** 30)
            break
    return rows


def fetch_meta(samples) -> dict:
    """family_id / priority_date / cpc_codes for rejected + gold, narrow
    columns only (title/abstract would cost 7x: 10.9 vs 1.5 GiB dry-run for
    398 numbers spread over 386 hash buckets)."""
    path = RUN_DIR / "meta.json"
    if path.exists():
        return json.loads(path.read_text())
    from google.cloud import bigquery
    nums = set()
    for s in samples:
        nums.add(_bq_form(s["rejected_pub"]))
        nums.add(_bq_form(s["target"]))
    nums = sorted(nums)
    client = _client()
    pubs_param = bigquery.ArrayQueryParameter("pubs", "STRING", nums)
    b = client.query("SELECT ARRAY(SELECT MOD(ABS(FARM_FINGERPRINT(p)), 4000) FROM UNNEST(@pubs) p) AS b",
                     job_config=bigquery.QueryJobConfig(query_parameters=[pubs_param]))
    buckets = list(b.result())[0].b
    params = [pubs_param, bigquery.ArrayQueryParameter("buckets", "INT64", buckets)]
    rows = _run(client, f"""
        SELECT publication_number, family_id, country_code, priority_date, cpc_codes
        FROM `{GC_PROJECT}.amie_patents.pubs`
        WHERE bucket IN UNNEST(@buckets) AND publication_number IN UNNEST(@pubs)""", params, max_gib=2.5)
    meta = {norm_pub(r.publication_number): {
        "family_id": r.family_id or "", "priority_date": int(r.priority_date or 0),
        "cpc_codes": list(r.cpc_codes or []), "country_code": r.country_code or ""} for r in rows}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta))
    return meta


def build_specs(samples, meta):
    specs = []
    for i, s in enumerate(samples):
        rj = meta.get(norm_pub(s["rejected_pub"]), {})
        gd = meta.get(norm_pub(s["target"]), {})
        prio = rj.get("priority_date") or int(f"20{s['app'][2:4]}0101")
        year = prio // 10000
        subs = [c[:4] for c in gd.get("cpc_codes") or [] if len(c) >= 4]
        sub = collections.Counter(subs).most_common(1)[0][0] if subs else ""
        if not sub:
            rsubs = [c[:4] for c in rj.get("cpc_codes") or [] if len(c) >= 4]
            sub = collections.Counter(rsubs).most_common(1)[0][0] if rsubs else "ZZZZ"
        specs.append({
            "qid": i, "terms": query_terms(s["query_claim"]), "sub": sub,
            "ylo": year - 5, "yhi": year, "cutoff": prio,
            "gfam": gd.get("family_id") or "-", "rfam": rj.get("family_id") or "-",
            "gpub": _bq_form(s["target"]), "rpub": _bq_form(s["rejected_pub"]),
        })
    return specs


POOL_SQL = """
WITH spec AS (
  SELECT i AS qid, SPLIT(t, ' ') AS terms, @subs[OFFSET(i)] AS sub, @ylo[OFFSET(i)] AS ylo,
         @yhi[OFFSET(i)] AS yhi, @cutoff[OFFSET(i)] AS cutoff, @gfam[OFFSET(i)] AS gfam,
         @rfam[OFFSET(i)] AS rfam, @gpub[OFFSET(i)] AS gpub, @rpub[OFFSET(i)] AS rpub
  FROM UNNEST(@terms) t WITH OFFSET i
),
base AS (
  SELECT publication_number, family_id, country_code, priority_date, title, abstract, cpc_codes
  FROM `{project}.amie_patents.pubs`
  WHERE bucket < @nb AND country_code IN UNNEST(@cc)
    AND abstract IS NOT NULL AND LENGTH(abstract) > 150 AND title IS NOT NULL
    AND REGEXP_CONTAINS(LOWER(abstract), r'\\bthe\\b') AND REGEXP_CONTAINS(LOWER(abstract), r'\\b(of|and|is)\\b')
),
n_base AS (SELECT COUNT(*) AS n FROM base),
vocab AS (SELECT DISTINCT t FROM spec, UNNEST(terms) t),
postings AS (
  SELECT b.publication_number, t
  FROM base b, UNNEST(ARRAY(
        SELECT DISTINCT x FROM UNNEST(REGEXP_EXTRACT_ALL(LOWER(CONCAT(b.title, ' ', b.abstract)), r'[a-z0-9]+')) x)) t
  WHERE t IN (SELECT t FROM vocab)
),
df AS (SELECT t, COUNT(*) AS df FROM postings GROUP BY t),
lex AS (
  SELECT s.qid, p.publication_number,
         SUM(LOG(1 + ((SELECT n FROM n_base) - df.df + 0.5) / (df.df + 0.5))) AS score
  FROM spec s, UNNEST(s.terms) qt
  JOIN postings p ON p.t = qt
  JOIN df ON df.t = qt
  GROUP BY s.qid, p.publication_number
),
lex_ranked AS (
  SELECT 'lexical' AS channel, s.qid, b.publication_number, b.family_id, b.country_code, b.priority_date,
         b.title, b.abstract, b.cpc_codes, l.score,
         ROW_NUMBER() OVER (PARTITION BY s.qid ORDER BY l.score DESC, FARM_FINGERPRINT(b.publication_number)) AS rn
  FROM lex l JOIN spec s ON s.qid = l.qid JOIN base b ON b.publication_number = l.publication_number
  WHERE b.priority_date < s.cutoff AND b.family_id NOT IN (s.gfam, s.rfam)
    AND b.publication_number NOT IN (s.gpub, s.rpub)
),
cpc_ranked AS (
  SELECT 'cpc' AS channel, s.qid, b.publication_number, b.family_id, b.country_code, b.priority_date,
         b.title, b.abstract, b.cpc_codes, 0.0 AS score,
         ROW_NUMBER() OVER (PARTITION BY s.qid
                            ORDER BY FARM_FINGERPRINT(CONCAT(b.publication_number, CAST(s.qid AS STRING)))) AS rn
  FROM base b, spec s
  WHERE DIV(b.priority_date, 10000) BETWEEN s.ylo AND s.yhi AND b.priority_date < s.cutoff
    AND b.family_id NOT IN (s.gfam, s.rfam) AND b.publication_number NOT IN (s.gpub, s.rpub)
    AND EXISTS (SELECT 1 FROM UNNEST(b.cpc_codes) c WHERE STARTS_WITH(c, s.sub))
)
SELECT * FROM lex_ranked WHERE rn <= @per_q
UNION ALL
SELECT * FROM cpc_ranked WHERE rn <= @per_q
"""


def build_pool(specs, n_buckets: int, per_q: int) -> list[dict]:
    """One scan of pubs (bucket < n_buckets == uniform hash sample) serves
    both negative channels; BigQuery bills bytes not compute, so the
    per-query lexical ranking is free. search_abstracts was rejected for
    channel (a): OR-of-terms + LIMIT returns arbitrary rows (measured: 80
    GB-country drilling patents for a focus-detector claim) and costs
    0.23 GiB per query (46 GiB for 200), AND-of-terms 18-22 GiB per query."""
    path = RUN_DIR / f"pool_b{n_buckets}_q{len(specs)}.jsonl"
    if path.exists():
        return [json.loads(l) for l in path.read_text().splitlines()]
    from google.cloud import bigquery
    P = bigquery
    params = [
        P.ArrayQueryParameter("terms", "STRING", [" ".join(s["terms"]) or "zzzz" for s in specs]),
        P.ArrayQueryParameter("subs", "STRING", [s["sub"] for s in specs]),
        P.ArrayQueryParameter("ylo", "INT64", [s["ylo"] for s in specs]),
        P.ArrayQueryParameter("yhi", "INT64", [s["yhi"] for s in specs]),
        P.ArrayQueryParameter("cutoff", "INT64", [s["cutoff"] for s in specs]),
        P.ArrayQueryParameter("gfam", "STRING", [s["gfam"] for s in specs]),
        P.ArrayQueryParameter("rfam", "STRING", [s["rfam"] for s in specs]),
        P.ArrayQueryParameter("gpub", "STRING", [s["gpub"] for s in specs]),
        P.ArrayQueryParameter("rpub", "STRING", [s["rpub"] for s in specs]),
        P.ScalarQueryParameter("nb", "INT64", n_buckets),
        P.ScalarQueryParameter("per_q", "INT64", per_q),
        P.ArrayQueryParameter("cc", "STRING", ["US", "EP", "WO"]),
    ]
    rows = _run(_client(), POOL_SQL.format(project=GC_PROJECT), params, max_gib=8)
    pool = [{"channel": r.channel, "qid": r.qid, "pub": norm_pub(r.publication_number),
             "family_id": r.family_id or "", "country_code": r.country_code,
             "priority_date": int(r.priority_date or 0), "title": r.title or "", "abstract": r.abstract or "",
             "cpc_codes": list(r.cpc_codes or []), "score": float(r.score or 0)} for r in rows]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(p) for p in pool))
    return pool


def sentence_case(title: str) -> str:
    return title.capitalize() if title.isupper() else title


def build_arena(samples, meta, pool):
    """canonical pub -> doc dict. Gold text comes from FiNE (same title+abstract
    fields; all-caps USPTO titles are sentence-cased to match BigQuery's)."""
    docs = {}
    for p in pool:
        if p["pub"] not in docs:
            docs[p["pub"]] = {"title": p["title"], "abstract": p["abstract"], "family_id": p["family_id"],
                              "cpc_codes": p["cpc_codes"], "src": "bq"}
    for s in samples:
        g = norm_pub(s["target"])
        m = meta.get(g, {})
        docs[g] = {"title": sentence_case(s["cited"].get("title") or ""), "abstract": s["cited"].get("abstract") or "",
                   "family_id": m.get("family_id") or "", "cpc_codes": m.get("cpc_codes") or [], "src": "fine"}
    return docs


def rank_metrics(order_pubs: list[str], gold: str, gold_fam: str, docs: dict) -> tuple[int, int]:
    doc_rank = order_pubs.index(gold) + 1
    fam_rank = doc_rank
    if gold_fam:
        for i, p in enumerate(order_pubs[:doc_rank]):
            if docs[p]["family_id"] == gold_fam:
                fam_rank = i + 1
                break
    return doc_rank, fam_rank


def summarize(ranks: list[int]) -> dict:
    r = np.array(ranks, dtype=float)
    return {"n": len(ranks), "R@1": float((r <= 1).mean()), "R@10": float((r <= 10).mean()),
            "R@100": float((r <= 100).mean()), "MRR": float((1.0 / r).mean()), "medR": float(np.median(r))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--buckets", type=int, default=200, help="hash-sample size: bucket < N of 4000")
    ap.add_argument("--per-q", type=int, default=80)
    ap.add_argument("--model", default="text-embedding-005")
    ap.add_argument("--skip-bm25", action="store_true")
    args = ap.parse_args()

    samples = load_samples(args.limit, None)
    for s in samples:
        s["rejected_pub"] = json.loads((DATA_DIR / s["app"] / "rejected_patent.json").read_text())["publication_number"]
    meta = fetch_meta(samples)
    specs = build_specs(samples, meta)
    n_rej = sum(1 for s in samples if norm_pub(s["rejected_pub"]) in meta)
    n_gold = sum(1 for s in samples if norm_pub(s["target"]) in meta)
    print(f"samples={len(samples)}  meta hits: rejected {n_rej}/{len(samples)}, gold {n_gold}/{len(samples)}")

    pool = build_pool(specs, args.buckets, args.per_q)
    docs = build_arena(samples, meta, pool)
    pubs = list(docs)
    idx = {p: i for i, p in enumerate(pubs)}
    by_q = collections.defaultdict(lambda: collections.Counter())
    for p in pool:
        by_q[p["qid"]][p["channel"]] += 1
    lex_n = [by_q[i]["lexical"] for i in range(len(samples))]
    cpc_n = [by_q[i]["cpc"] for i in range(len(samples))]
    stats = {
        "pool_rows": len(pool), "unique_docs": len(pubs), "gold_docs": len({norm_pub(s["target"]) for s in samples}),
        "lexical_per_q": (float(np.mean(lex_n)), int(min(lex_n)), int(max(lex_n))),
        "cpc_per_q": (float(np.mean(cpc_n)), int(min(cpc_n)), int(max(cpc_n))),
        "q_with_no_cpc_neg": int(sum(1 for n in cpc_n if n == 0)),
        "country": dict(collections.Counter(p["country_code"] for p in pool)),
        "families": len({d["family_id"] for d in docs.values() if d["family_id"]}),
    }
    print("pool:", json.dumps(stats))

    texts = [doc_chunks(docs[p], "title_abstract")[0] for p in pubs]
    queries = [s["query_claim"] for s in samples]
    n_chars = sum(len(t) for t in texts) + sum(len(q) for q in queries)
    print(f"[{args.model}] {len(texts)} docs + {len(queries)} queries, {n_chars / 1e6:.1f}M chars, "
          f"~{(len(texts) + 99) // 100 + (len(queries) + 99) // 100} requests at 100/req (cache hits free)")
    doc_vecs = embed_vertex(texts, args.model, "RETRIEVAL_DOCUMENT")
    q_vecs = embed_vertex(queries, args.model, "RETRIEVAL_QUERY")

    results, per_query = {}, []
    dense_doc, dense_fam, oracle_doc, oracle_fam = [], [], [], []
    share4 = share3 = primary4 = 0
    above = collections.Counter()
    own_doc, own_fam, own_above = [], [], []
    own = collections.defaultdict(dict)
    for p in pool:
        own[p["qid"]][p["pub"]] = p["channel"]
    gold_set = {norm_pub(s["target"]) for s in samples}
    for qi, s in enumerate(samples):
        gold = norm_pub(s["target"])
        gfam = docs[gold]["family_id"]
        sims = doc_vecs @ q_vecs[qi]
        order = [pubs[i] for i in np.argsort(-sims)]
        dr, fr = rank_metrics(order, gold, gfam, docs)
        dense_doc.append(dr)
        dense_fam.append(fr)
        rj = meta.get(norm_pub(s["rejected_pub"]), {})
        q4 = {c[:4] for c in rj.get("cpc_codes") or []}
        q3 = {c[:3] for c in q4}
        g4 = {c[:4] for c in docs[gold]["cpc_codes"]}
        hit4, hit3 = bool(q4 & g4), bool(q3 & {c[:3] for c in g4})
        share4 += hit4
        share3 += hit3
        qp, gp = (rj.get("cpc_codes") or [""])[0][:4], (docs[gold]["cpc_codes"] or [""])[0][:4]
        primary4 += bool(qp) and qp == gp
        for p in order[:dr - 1]:
            above[own[qi].get(p) or ("other_gold" if p in gold_set else "other_query")] += 1
        own_above.append(sum(1 for p in order[:dr - 1] if p in own[qi]))
        own_order = [p for p in order if p in own[qi] or p == gold]
        odr2, ofr2 = rank_metrics(own_order, gold, gfam, docs)
        own_doc.append(odr2)
        own_fam.append(ofr2)
        if q4:
            keep = [p for p in order if {c[:4] for c in docs[p]["cpc_codes"]} & q4]
            if gold in keep:
                odr, ofr = rank_metrics(keep, gold, gfam, docs)
            else:
                odr = ofr = len(pubs) + 1
        else:
            odr, ofr = dr, fr
        oracle_doc.append(odr)
        oracle_fam.append(ofr)
        per_query.append({"app": s["app"], "gold": gold, "dense_doc_rank": dr, "dense_fam_rank": fr,
                          "cpc_share4": hit4, "sub": specs[qi]["sub"], "terms": specs[qi]["terms"]})
    results["te005_title_abstract (doc)"] = summarize(dense_doc)
    results["te005_title_abstract (family)"] = summarize(dense_fam)
    results["te005 own-query pool only, 161 docs (doc)"] = summarize(own_doc)
    results["te005 own-query pool only, 161 docs (family)"] = summarize(own_fam)
    results["te005 + CPC-subclass oracle filter (doc)"] = summarize(oracle_doc)
    results["te005 + CPC-subclass oracle filter (family)"] = summarize(oracle_fam)

    if not args.skip_bm25:
        bm_doc, bm_fam = [], []
        for qi, s in enumerate(samples):
            gold = norm_pub(s["target"])
            scores = bm25_scores(s["query_claim"], texts)
            order = [pubs[i] for i in np.argsort(-scores, kind="stable")]
            dr, fr = rank_metrics(order, gold, docs[gold]["family_id"], docs)
            bm_doc.append(dr)
            bm_fam.append(fr)
            per_query[qi]["bm25_doc_rank"] = dr
        results["bm25_title_abstract (doc)"] = summarize(bm_doc)
        results["bm25_title_abstract (family)"] = summarize(bm_fam)

    n = len(samples)
    n_q_cpc = sum(1 for s in samples if meta.get(norm_pub(s["rejected_pub"]), {}).get("cpc_codes"))
    cpc = {"share_subclass4": share4 / n, "share_class3": share3 / n, "primary_subclass4": primary4 / n,
           "queries_with_cpc": n_q_cpc, "n": n,
           "codes_per_query": float(np.mean([len(meta.get(norm_pub(s["rejected_pub"]), {}).get("cpc_codes") or []) for s in samples])),
           "codes_per_gold": float(np.mean([len(docs[norm_pub(s["target"])]["cpc_codes"]) for s in samples]))}
    print(f"\nCPC oracle: query-gold share any subclass(4) {cpc['share_subclass4']:.3f}, any class(3) {cpc['share_class3']:.3f}, "
          f"primary subclass(4) {cpc['primary_subclass4']:.3f} (queries with CPC: {n_q_cpc}/{n}; "
          f"codes/query {cpc['codes_per_query']:.1f}, codes/gold {cpc['codes_per_gold']:.1f})")
    tot_above = sum(above.values()) or 1
    stats["ranked_above_gold"] = {k: (v, round(v / tot_above, 3)) for k, v in above.most_common()}
    print("docs ranked above gold (te005), by origin:", stats["ranked_above_gold"])
    stats["own_negs_above_gold"] = {"mean": float(np.mean(own_above)), "median": float(np.median(own_above)),
                                    "zero": int(sum(1 for x in own_above if x == 0))}
    print("own-query negatives ranked above gold:", stats["own_negs_above_gold"])
    print(f"\n{'method':<48}{'R@1':>7}{'R@10':>7}{'R@100':>7}{'MRR':>8}{'medR':>7}")
    for name, m in results.items():
        print(f"{name:<48}{m['R@1']:>7.3f}{m['R@10']:>7.3f}{m['R@100']:>7.3f}{m['MRR']:>8.3f}{m['medR']:>7.0f}")
    print(f"\nBQ billed this run: {sum(BQ_BILLED):.2f} GiB over {len(BQ_BILLED)} calls {['%.2f' % b for b in BQ_BILLED]}")
    (RUN_DIR / "results.json").write_text(json.dumps(
        {"results": results, "cpc": cpc, "pool": stats, "bq_billed_gib": BQ_BILLED, "per_query": per_query,
         "args": vars(args)}, indent=1))


if __name__ == "__main__":
    main()
