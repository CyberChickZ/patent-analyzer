# Stage-level Evals

Offline, per-stage benchmarks so retrieval/encoding decisions are made on
measurements instead of intuition. No LLM calls, no API cost — everything
runs locally.

## Dataset: FiNE-Patents

[FiNE-Patents](https://github.com/boschresearch/fine-patents) (Bosch Research,
SIGIR 2026, CC-BY-4.0): 3,163 real EPO applications with **examiner-written
ground truth** — for each application, the closest prior art the examiner
cited, plus feature-level annotations of which prior-art passage discloses
which claim feature.

Setup (not committed to git — ~1 GB unpacked):

```bash
cd backend
git clone --depth 1 https://github.com/boschresearch/fine-patents eval_data/fine-patents
cd eval_data/fine-patents
cat data/packaged.tar.gz.part*.bin > data/packaged.tar.gz
tar xzf data/packaged.tar.gz -C data
mv data/data/packaged data/packaged && rmdir data/data && rm data/packaged.tar.gz
```

## Retrieval eval

Task: query = rejected application's claim 1; corpus = all cited prior-art
patents in the sample; target = the examiner-cited closest prior art.

```bash
cd backend
python3 evals/retrieval_eval.py --limit 500 --mode all
```

Baseline (2026-08-26, 500 samples, 492-doc corpus, local MiniLM):

| method                        | R@1  | R@10 | R@50 | MRR  |
|-------------------------------|------|------|------|------|
| bm25_keyword                  | .318 | .620 | .834 | .420 |
| minilm_title_abstract         | .302 | .630 | .850 | .417 |
| minilm_claims_chunks          | .328 | .668 | .870 | .440 |
| hybrid_rrf (bm25 + chunks)    | .388 | .722 | .906 | .502 |

Encoder A/B (2026-09-17, 200 samples, 197-doc corpus — smaller arena, so
compare only within this table):

| method                        | R@1  | R@10 | R@50 | MRR  |
|-------------------------------|------|------|------|------|
| bm25_keyword                  | .380 | .755 | .945 | .503 |
| minilm_claims_chunks          | .450 | .780 | .955 | .563 |
| te005_claims_chunks           | .505 | .880 | .990 | .636 |
| ge001_claims_chunks           | .520 | .900 | .985 | .641 |
| hybrid_rrf_ge001              | .490 | .900 | .985 | .631 |

Takeaways: structure-aware chunking (per-claim vectors, max-pool) beats
title+abstract with every encoder; the 2020-era MiniLM is roughly tied with
plain BM25 while Vertex encoders are far ahead (gemini-embedding-001 best,
text-embedding-005 within ~2 pts); BM25 RRF fusion is a ~5-9 pt win under a
weak encoder but neutral-to-slightly-negative under strong ones — keep it
for MiniLM, re-evaluate after the pipeline encoder upgrade. Serving note:
te005 batches 100 texts/request while ge001 is 1/request on Vertex, so
te005 is the pragmatic choice for the production rerank path.

## Pipeline parity

`--mode pipeline` runs the production `semantic_search.rerank_docs` itself
(legacy-doc dicts, claims_text blob splitting, Vertex encoder, fallback
logic) against the same arena, so the number measures what actually serves:

| method                        | R@1  | R@10 | R@50 | MRR  |
|-------------------------------|------|------|------|------|
| pipeline_rerank_docs (te005)  | .500 | .895 | .990 | .636 |

(2026-09-17, 200 samples — same arena as the encoder A/B above. The old
pipeline representation, minilm_claims_chunks, scored .780 R@10 here.)

The pipeline now uses text-embedding-005 + structure-aware chunks +
max-pool; BM25 RRF fusion applies only on the MiniLM fallback path, per
the fusion nuance above. BigQuery candidates carry claims_text (8k chars)
into the rerank so patent claims become chunks.

Planned next: an extraction-stage eval using the feature-level breakdown
annotations.
