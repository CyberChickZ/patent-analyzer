"""Global configuration constants."""

# Search
EVAL_LIMIT = 200
MAX_OR_TERMS = 3
MAX_QUERIES_PER_GROUP = 3
SERPAPI_TIMEOUT = 60
SERPAPI_MAX_RETRIES = 3
SERPAPI_RETRY_BACKOFF = [2, 5, 10]
MAX_PAGES_PATENT = 1       # 100 results/page
MAX_PAGES_PAPER = 5        # 20 results/page

# Scoring
RISK_THRESHOLDS = {
    "HIGH": 0.7,      # >70%: core ideas heavily anticipated
    "MEDIUM": 0.4,    # 40-70%: partial overlap
    "LOW": 0.2,       # 20-40%: minor overlap
    "CLEAR": 0.0,     # <20%: appears novel
}

# Output
OUTPUT_DIR = "./patent-analysis-output"
PAPERS_SUBDIR = "papers"
LOG_FILE = "search.log"
