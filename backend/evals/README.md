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

Baseline (2026-08-26, 500 samples, 492-doc corpus):

| method                        | R@1  | R@10 | R@50 | MRR  |
|-------------------------------|------|------|------|------|
| bm25_keyword                  | .318 | .620 | .834 | .420 |
| minilm_title_abstract         | .302 | .630 | .850 | .417 |
| minilm_claims_chunks          | .328 | .668 | .870 | .440 |
| hybrid_rrf (bm25 + chunks)    | .388 | .722 | .906 | .502 |

Takeaways: structure-aware chunking (per-claim vectors, max-pool) beats the
current title+abstract representation on every metric; the 2020-era MiniLM
encoder is roughly tied with plain BM25, so a modern encoder has clear
headroom; hybrid RRF fusion is a ~5-9 pt free win and should go into the
main pipeline.

Planned next: gemini-embedding-001 A/B on the same task; an extraction-stage
eval using the feature-level breakdown annotations.
