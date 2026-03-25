# Patent Analyzer

Automated patent novelty analysis pipeline. Given a paper or patent document, performs:

1. **IDCA** — Invention Detection & Classification (patentability check)
2. **Decomposition** — Break invention into atomic components + generate evaluation checklist
3. **Prior Art Search** — SerpAPI (Google Patents + Google Scholar) with fallback strategy
4. **Evaluation** — Score each prior art document against the checklist
5. **Report** — Static HTML with clickable Markdown export

## Architecture: Fixed Code vs LLM

| Step | Type | Module |
|------|------|--------|
| Query generation | **FIXED** | `query_builder.py` |
| SerpAPI search + retry | **FIXED** | `searcher.py` |
| Result deduplication | **FIXED** | `searcher.py` |
| PDF download | **FIXED** | `searcher.py` |
| Keyword pre-filtering | **FIXED** | `prefilter.py` |
| Score computation | **FIXED** | `scorer.py` |
| Risk classification | **FIXED** | `scorer.py` |
| Results aggregation | **FIXED** | `scorer.py` |
| HTML report generation | **FIXED** | `report_generator.py` |
| Invention detection | **LLM** | `prompts/idca.py` |
| Summarization | **LLM** | `prompts/idca.py` |
| Checklist generation | **LLM** | `prompts/naa.py` |
| Delegation planning | **LLM** | `prompts/naa.py` |
| Per-doc evaluation | **LLM** | `prompts/naa.py` |
| Overall summary | **LLM** | `prompts/naa.py` |

**Key design**: LLM responses begin with a decision word (e.g., `present`/`absent`/`implied`) enabling deterministic routing without parsing free-text.

## Setup

```bash
pip install -e .
export SERPAPI_KEY=your_key_here
```

## Usage

### As Claude Code Skill
```
/patent-analyze path/to/paper.pdf
```

### As CLI Scripts
```bash
# Search
patent-search --queries-file queries.json --output search.json --log-file search.log

# Pre-filter top 200
patent-prefilter --search-results search.json --checklist-file phase2.json --output top200.json --limit 200

# Generate report
patent-report --input results.json --output report.html
```

## Search Strategy

Follows USPTO patent examination methodology (35 USC 102/103):

- **Layer 1 (102)**: Search for complete system anticipation
- **Layer 2**: Core novelty subcombinations
- **Layer 3 (103)**: Individual dependent claim features
- **Layer 4**: Broader field for obviousness combinations

Each query has **3-level fallback**: tight → medium → broad, ensuring minimum 1 result per group.

## Output

Self-contained static HTML report with:
- Executive summary + risk badge (HIGH/MEDIUM/LOW/CLEAR)
- Search group cards with query + match counts
- Prior art match cards with color-coded scores
- Per-checklist-item evaluation (check/cross + analysis)
- Click any card → Markdown popup (copy/download)
