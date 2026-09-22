# AMIE — prior-art analysis for university research

AMIE takes a research paper or manuscript and produces a prior-art report: which
parts of the described work could be claimed, which earlier patents and papers
already cover them, the verbatim evidence for each finding, and a draft claim set
that avoids what was found. It is built for a technology-transfer office; the
report is written for a licensing manager or patent attorney, not for a lawyer
to rely on without review.

US patent practice (MPEP) sets the rules for claims and verdicts; the prior-art
search itself is worldwide.

## What one job does

| Step | What the program does | Output |
|---|---|---|
| 1 Read | Converts the PDF into sections, figures and references. Decides whether an invention is present and assigns a CPC class. | structured document |
| 2 Checklist | Lists 1–3 candidate inventions. For each, a list of the features it must have, every feature pinned to a sentence in the paper, written as a patent claim would be. | checklist per candidate |
| 3 Search | For each feature: builds a Google Patents query, takes the top 100 results, records the terms those patents use and reuses them in the next query. Locates the paper and its citation neighbourhood, then the patents that cite those papers. Sends every patent number found to our own BigQuery copy of the patent record and adds what they cite and what cites them. A fast model screens the pool; a claims-level judge keeps what matches. | ranked list of documents |
| 4 Evidence | Reads each delivered document against every feature. The model must quote sentences; a quote that cannot be located in the document text does not count. | feature × document matrix with verified quotes |
| 5 Verdict | A fixed rule over the verified quotes: one document covering every feature → anticipated (35 USC 102, MPEP 2131); two or three together → flagged for human review (a screening signal, not the legal obviousness test); otherwise no blocking art among the documents read. | ruling and reasons |
| 6 Draft | Claims written from the checklist and narrowed around what the found documents cover; 1 independent claim, up to 9 dependent claims; basis and 112(b) checks. | claim set + HTML/Markdown report |

The job can pause after any step. A reviewer edits the step's output or the
prompt behind it and reruns from there. Every prompt change, reviewer edit and
comment is kept on the Feedback page with a link to the job.

## Measured performance (22 Sep 2026)

Every score is "share of the examiner's items we reproduced", out of 100. The
answer key is always a human examiner's record; nothing is self-labelled.

| Step | Score | Answer key | Sample | Reference point |
|---|---|---|---|---|
| Read | 97 | FiNE-Patents examiner claim splits | 20 patents | — |
| Checklist | 93 | FiNE-Patents | 5 patents · 29 features | — |
| Search | 16 delivered · 44 in pool | Pap2Pat paper → its patent → examiner citations (SEA, ISR, EXA) | 8 papers · 32 families | one human examiner ≈ 54 |
| Evidence | 21 (P .18 / R .33) | FiNE-Patents examiner-cited passages | 58 applications · 227 features | best published .209 |
| Verdict | 44 | PANORAMA office actions | 200 claims | random .335 |
| Draft | 77 recall · 78 precision | Dis2Pat filed independent claims | 5 applications | — |

One production run: $2.81, 25 minutes, 238 model calls, 7,914 documents
screened, 120 delivered, 119 read in full. Full logs, per-case error reviews and
the measurement protocol are in `outputs/` of the mirror repository and in
`backend/evals/`.

## Repository layout

```
backend/            FastAPI + LangGraph service (Python 3.12 on Cloud Run; 3.14 locally)
  app/              HTTP API, auth (Firebase, oregonstate.edu only), job store, HITL, feedback, quota
  graph/            LangGraph wiring: 6 nodes + gate nodes (interrupt) + conditional routing
  nodes/            one module per step
  patent_analyzer/  search channels, agentic loop, BigQuery client, evidence, adjudication, drafting, report
  prompts/          prompt registry: versioned templates with contracts, stored in GCS in production
  evals/            benchmark harnesses (FiNE, Pap2Pat, PANORAMA, Dis2Pat) — every script requires --budget-usd
  tests/            649 tests: pytest tests -q
frontend/amie/      TypeScript + Vite single-page app behind an Express proxy (IAM token to the backend)
run_local.sh        backend on :8000 and UI on :5173 with AUTH_DISABLED=1
DEPLOY.md           Cloud Run deployment, environment variables, checks
```

## Run locally

```bash
cp backend/.env.yaml.example backend/.env.yaml   # fill in keys; the file is git-ignored
./run_local.sh
```

Environment variables the backend reads (names only; values never go in git):
`GC_PROJECT`, `LLM_MODEL`, `SERPAPI_KEYS`, `SERPAPI_RESERVED_KEYS`, `OPENALEX_KEY`,
`SEMANTIC_SCHOLAR_KEY`, `LENS_API_TOKEN`, `USPTO_ODP_API_KEY`, `OUTPUT_DIR`,
`SMTP_USER`, `SMTP_PASSWORD`, `DAILY_SPEND_CAP_USD`, `BQ_MAX_GIB_PER_JOB`.

## Deploy

See `DEPLOY.md`. Two Cloud Run services in `us-west1`: `patent-analyzer`
(backend, no unauthenticated access, CPU always allocated, one instance) and
`patent-analyzer-frontend`. The frontend host must be listed in Firebase
Authentication → Authorized domains.

## Data sources and limits

Google Patents via SerpAPI (250 free searches per key per month; four keys, one
reserved for demonstrations) and by direct request (blocked intermittently);
our own BigQuery tables of patents 1980→ (claims, abstracts, citations both
ways, families, US descriptions; ≈ $0.80 of scanning per job); Lens (trial);
USPTO Open Data; Semantic Scholar, OpenAlex, arXiv; paper full text through
PubMed Central, Europe PMC, Unpaywall and Crossref text-mining links (about 21
of 104 papers readable in a typical job; paywalled papers are reported as
"abstract only" and can be uploaded by the reviewer).

Known limits, in order of effect: search recall (16 of 100 delivered against a
human's ≈ 54); paper full-text access; the search/extraction model
(`gemini-2.5-pro`) retires on 20 Oct 2026.

## License

MIT — Copyright (c) 2026 Oregon State University. See `LICENSE`.
