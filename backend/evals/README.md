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

## Extraction eval

Task: split claim 1 into features; ground truth is the examiner's own
feature breakdown. Greedy 1:1 embedding matching (te005,
SEMANTIC_SIMILARITY), scored at several thresholds:

```bash
python3 evals/extraction_eval.py --limit 100 --mode regex
```

Baseline (2026-09-17, 100 EN samples, 502 examiner features):

| mode  | recall@0.8 | precision@0.8 | F1   |
|-------|-----------|---------------|------|
| regex | .815      | .983          | .891 |

Honest surprise: the regex claim splitter — flagged earlier as the weak
link — is a strong baseline. Precision .983 means what it splits is real;
the gap is recall (~18% of examiner features get merged into a coarser
limitation, which would coarsen evidence mapping downstream). An LLM
splitter has a measurable target to beat now.

Planned next: an LLM-based splitter mode for extraction_eval; wiring the
feature-level passage annotations into an evidence-mapping eval.

## Coverage eval (product question 2, evaluator ceiling)

`evals/coverage_eval.py`: give the evaluator the examiner-cited prior art
D1 (oracle retrieval) and ask whether it covers the features the examiner
marked as disclosed — with a verbatim quote that `quote_verify` can locate
in D1. `para_hit` = share of verified quotes that land in the paragraph
the examiner cited.

2026-09-17, 20 apps / 98 examiner-disclosed features, Gemini 2.5 Pro:

| checklist / doc_mode | cov_raw | cov_verified | para_hit | key_match |
|----------------------|--------:|-------------:|---------:|----------:|
| oracle / full_text   | .837    | **.673**     | .465     | .949 |
| oracle / abstract    | .908    | (no quotes)  | –        | 1.000 |
| regex  / full_text   | .806    | **.684**     | .429     | 1.000 |
| regex  / abstract    | .735    | (no quotes)  | –        | .955 |

Reading: without a quote requirement the model claims 91% coverage; once
every positive must carry a quote that exists in the document, 67% survive.
The 16-24 pt gap is the fabrication rate the old pipeline reported as
"coverage". Verified quotes land in the examiner's paragraph 43-47% of the
time — the rest are real quotes from elsewhere in D1, which is fine for
coverage but not for pinpoint citation. `key_match` < 1 was the
numbered-key bug fixed by `_align_checklist_keys`.

### FiNE Table 2 protocol (`--protocol fine`, 2026-09-17)

Same evaluator, scored the way FiNE-Patents scores passage retrieval
(`evaluate.py::compute_retrieval_metrics`): each predicted feature is
mapped to the examiner feature with the highest edit similarity
(many-to-one, no threshold); each verified quote is located to a passage
`(kind, number)` in the cited patent (paragraph / claim / abstract); per
examiner feature with gold passages, tp/fp/fn over passage ids give
P/R/F1, averaged within the sample and then over samples. Claim-level =
union of predicted vs union of gold passages. Gold now follows FiNE's
`locate_cited_passages`: paragraph/claim/abstract references only, on the
prior-art label whose identifier matches `cited_patent.publication_number`
(figure/page/component-only features carry no gold passage).

Sample: fixture `stage` ids restricted to FiNE test split with a rejected
version (`--sample test`, 58 apps, 227 examiner features with gold
passages). `--baseline rougeL|embed` reproduce FiNE's non-LLM baselines
(top-5, tau 0.4 / 0.5; embed uses text-embedding-005 instead of
Qwen3-Embedding-8B) without any Gemini call.

| variant (58 apps)        | feat P | feat R | feat F1 | claim P | claim R | claim F1 |
|--------------------------|-------:|-------:|--------:|--------:|--------:|---------:|
| baseline rougeL          | .057 | .134 | .069 | .124 | .258 | .148 |
| baseline embed (te005)   | .071 | .212 | .096 | .113 | .326 | .155 |
| regex  / full_text (LLM) | .158 | .104 | .113 | .278 | .185 | .195 |
| oracle / full_text (LLM) | .184 | .111 | .129 | .323 | .148 | .190 |
| FiNE Table 2 Rouge-L (N=932)      | .046 | .119 | .057 | .113 | .255 | .136 |
| FiNE Table 2 Qwen3-8B-Emb (N=932) | .083 | .227 | .106 | .137 | .357 | .174 |
| FiNE Table 2 Hier. Qwen3.5-397B   | .170 | .391 | .209 | .207 | .568 | .274 |

Our re-implementation of the two baselines lands within ~1 pt of the
paper on this 58-app subset, so the protocol port is sound. The evaluator
sits between the embedding baseline and FiNE's LLM rows on feature F1:
precision is on par with their LLMs (.16-.18 vs .13-.17) but recall is a
third of theirs (.10-.11 vs .27-.39) because we emit exactly one verified
quote per feature while FiNE models return several passages per feature.
Not directly comparable: subset (58 vs 932), Gemini 2.5 Pro vs Qwen,
single-quote output, and the regex splitter vs `re.split("[;\n]")`.
