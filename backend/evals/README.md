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

### Multi-quote + dual-threshold grounding (`--quotes multi`, 2026-09-17)

The recall gap above is structural: one quote per criterion can only ever
hit one passage, while the examiner cites 2.3 passages per feature on
average (121/227 features cite >= 2). `evals/eval_prompts.py` is an
eval-only copy of the full_text prompt that asks for `evidence_quotes`
(1-5 verbatim 10-40 word excerpts, one per disclosing passage); nodes/ and
app/ are untouched. Every quote is checked by two verifiers:
`quote_verify.locate_quote` (char-level difflib >= .9, the production
check) and `evals/quote_dual.verify_quote_dual` (span_ratio >= .70 of the
quote's content words inside one sliding window AND bigram_ratio >= .30
of its adjacent token pairs found in the document). Passing quotes are
located to a passage (`quote_location`; the dual arm falls back to
`locate_dual`, the best passage that itself passes the dual test) and the
union per criterion feeds `fine_score` unchanged. `--verifier dual|both`
re-scores the single-quote runs with the same machinery at zero cost.

| variant (58 apps / 227 feats)     | feat P | feat R | feat F1 | claim P | claim R | claim F1 | q/item | survive | pred psg |
|-----------------------------------|-------:|-------:|--------:|--------:|--------:|---------:|-------:|--------:|---------:|
| oracle single + locate (E2)       | .184 | .111 | .129 | .323 | .148 | .190 | 1.00 | 147/173 | 138 |
| oracle single + dual              | .210 | .128 | .149 | .309 | .168 | .205 | 1.00 | 172/173 | 160 |
| oracle multi  + locate            | .169 | .289 | .190 | .265 | .386 | .282 | 3.60 | 569/638 | 438 |
| oracle multi  + dual              | .178 | .331 | **.206** | .264 | .421 | .293 | 3.60 | 636/638 | 479 |
| oracle multi  + dual AND locate   | .169 | .289 | .190 | .265 | .386 | .282 | 3.60 | 569/638 | 438 |
| regex  single + locate (E2)       | .158 | .104 | .113 | .278 | .185 | .195 | 1.00 | 214/251 | 191 |
| regex  single + dual              | .207 | .150 | .158 | .308 | .233 | .234 | 1.00 | 250/251 | 222 |
| regex  multi  + locate            | .156 | .255 | .177 | .210 | .337 | .232 | 3.22 | 693/772 | 503 |
| regex  multi  + dual              | .159 | .282 | **.185** | .212 | .360 | .240 | 3.22 | 769/772 | 549 |
| regex  multi  + dual AND locate   | .156 | .255 | .177 | .210 | .337 | .232 | 3.22 | 693/772 | 503 |
| FiNE Table 2 Single-step Qwen3.5  | .148 | .389 | .192 | .202 | .576 | .271 |      |         |          |
| FiNE Table 2 Hier. Qwen3.5-397B   | .170 | .391 | .209 | .207 | .568 | .274 |      |         |          |

Reading: multi-quote roughly triples recall (.11 -> .29-.33) at 1-2 pt of
precision, moving oracle feature-F1 from .129 to .206 and regex from .113
to .185, i.e. into the FiNE LLM band (.192-.209). The dual verifier is not
what moves the number: it passes 99.7% of quotes (636/638) and adds
~4 pt of recall over locate_quote by keeping ellipsis-spliced quotes
("...") that difflib scores at .6-.9 but that sit in one passage (52 of
the 67 rescued quotes are found by the exact 60-char prefix match). AND
of both verifiers equals locate alone, since dual is a superset. The two
quotes dual rejects are genuine cross-paragraph splices. 116 Gemini calls
(58 x 2), ~2.3M tok in.


## Draft eval (Step 6 — claim drafting)

Two gates for `nodes/draft.py`. Both are about the *draft*, not the search:
gate 1 runs with no prior art at all, gate 2 only reads the drafted claims.

### Gate 1 — Dis2Pat element overlap

```bash
python3 evals/draft_eval.py --limit 5 --max-live-calls 60
```

Data: HF `lj408/Dis2Pat` test split (943 rows, CC-BY-SA-4.0; Jiang, Sun,
Goetz, arXiv 2608.21249), cached at `eval_data/dis2pat/dis2pat_test.jsonl`.
Sample: seed-42 permutation, first `--limit`. **Input is only the seven
`disclosure` fields** (title / problem / core_idea / how_it_works / novelty /
benefits / optional_variants) through `adapters.disclosure.doc_from_fields`;
no field of the patent reaches the pipeline. `DRAFT_RECHECK=0`,
`DRAFT_ADVISORY=0` — the Dis2Pat patents are on Google Patents and would
find themselves, so gate 1 does not search.

Gold: the granted `claims` string, independent claims cut with
`nodes.claim_mode._parse_claim_limitations` + `_split_preamble` (the cutter
the extraction evals use), dependent claims reduced to their added
limitation. Matching: `extraction_eval.embed` (te005) + `greedy_match`, 1:1,
tau = .7 — the same rule as the Pap2Pat coverage eval.

Columns: the drafted claim 1 (preamble + limitations), the A2 pre-search
`independent_claim_draft` (same cutter) as the control, the extraction
elements joined directly as the upper-bound candidate, gold against itself
(= 1.0), and the drafted dependents against the gold dependents.

### Gate 2 — indefiniteness rate (PEDANTIC)

```bash
python3 evals/pedantic_definiteness_eval.py calibrate --limit 100    # detector vs their labels
python3 evals/pedantic_definiteness_eval.py draft --limit 100        # our drafts
```

Detector D = the `patent_analyzer/draft/definiteness.py` rules (antecedent
basis / relative term / exemplary phrasing / 112(f)) plus the PEDANTIC
examination prompt (Knappich et al., arXiv 2505.21342;
github.com/boschresearch/pedantic-patentsemtech, MIT) with their category
list and likelihood expressions verbatim, run on Gemini through
`app.llm.definiteness_advisory`; their `get_claim` / `search_description`
tools are replaced by the parent claims and the description inline. Data:
`eval_data/pedantic/dataset.pkl` (git-lfs; fetched from the media host —
`src/pedantic` is not installed, the eval ships a pickle shim). Calibration
is reported next to the paper's Logistic-Regression baseline (F1 .563 /
AUROC .595) and their Qwen-2.5-72B + LR ensemble (F1 .588 / AUROC .603);
**D is used as a gate only if it beats the LR baseline**, otherwise the
draft numbers are reported as numbers. `draft` pools every claim of the
runs `draft_eval.py` wrote, reports the rule-flag rate before and after the
node's reword pass, D's indefinite rate, and the same for the A2 draft.

### First numbers (2026-09-18, gemini-2.5-pro, 5 Dis2Pat rows seed 42)

Gate 1 (`--limit 5`, tau .7, macro over rows / pooled over elements):

| column            | recall | prec | full | pooled R | pooled P |
|-------------------|-------:|-----:|-----:|---------:|---------:|
| draft claim 1     | .742 | .781 | .200 | .711 | .730 |
| A2 pre-search draft | .759 | .679 | .200 | .737 | .667 |
| elements direct   | .742 | .781 | .200 | .711 | .730 |
| gold self         | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| draft dependents  | .736 | .575 | .200 | .657 | .575 |

Reading: the drafted claim 1 recites the same share of the granted claim's
elements as the extraction elements do (.742 — the draft is those elements,
so this is the invariant working, not a result), at +10 pt of precision over
the A2 free-text draft for -1.7 pt of recall: the A2 draft splits into more
clauses than the granted claim has elements. Granularity 1.18 drafted
limitations per gold element, 0 pool items dropped for an unlocatable quote,
41 LLM calls for the 5 rows. `full` = .2 (one row of five recites every gold
element) — with 5 rows that is one document, not a rate.

Gate 2, calibration on 100 PEDANTIC test claims (45 indefinite / 55 definite):

| detector      | P | R | F1 | acc | AUROC |
|---------------|--:|--:|---:|----:|------:|
| rules only    | .476 | .667 | .556 | .520 | .533 |
| LLM examiner  | .466 | .756 | .576 | .500 | .568 |
| rules OR LLM  | .462 | .933 | .618 | .480 | .557 |
| paper LR baseline | | | .563 | | .595 |
| paper 72B+LR ensemble | | | .588 | | .603 |

**D is not used as a gate**: it passes the LR baseline on F1 (.618 vs .563)
but not on AUROC (.557 vs .595), and its precision is barely above the .45
base rate — it calls 73% (LLM) / 91% (union) of PEDANTIC claims indefinite
where the truth is 45%. Per-category, the antecedent rule fires on 60 of 100
claims where examiners cited antecedent basis in 9 (P .100 / R .667); the
relative-term rule is the usable one (P .429 / R .500); the 112(f) rule fired
0 times against 8 gold functional-claiming rejections — PEDANTIC's examiners
read "control logic … identifying" as a nonce placeholder, which our word
list does not cover.

Gate 2 on our drafts (90 claims from the 5 gate-1 documents):

- rule flag rate **.322 before the node's reword pass -> .044 after** (4
  antecedent flags left, all in one document; they are the node's own
  `open_flags`), so the reword loop clears 25 of 29 rule flags but not all —
  the design's "0 after reword" is not met.
- D (LLM examiner) calls 74/90 = .822 of the drafted claims indefinite,
  against .73 on PEDANTIC's 50/50 set: read as ~1.1x the detector's own
  positive rate, not as an 82% defect rate. The A2 pre-search draft scores
  3/5 = .600 (n = 5 claims, one per document).
- categories D raises on our claims: undefined_term 40, antecedent_basis 36,
  contradicting_limitations 28, relative_term 21, omission 11 — the three the
  rules cannot check are the majority, which is why they stay advisory.
