"""Shim: the dual-threshold grounding now lives in patent_analyzer.quote_dual."""
from patent_analyzer.quote_dual import *  # noqa: F401,F403
from patent_analyzer.quote_dual import DocIndex, SPAN_TAU, BIGRAM_TAU, MIN_CONTENT, tokens, content_tokens, span_ratio, bigram_ratio, verify_quote_dual, locate_dual  # noqa: F401
