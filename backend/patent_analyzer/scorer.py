"""
Deterministic scoring aggregation — no LLM needed.

Takes evaluation results and computes:
- Per-document total scores
- Risk level classification
- Sorted rankings
- Statistics
"""

from .config import RISK_THRESHOLDS


def compute_total_score(checklist_results: dict) -> float:
    """Compute total_score. Uses 3-level scores if available, falls back to binary."""
    if not checklist_results:
        return 0.0
    has_scores = any(
        isinstance(v, dict) and "score" in v
        for v in checklist_results.values())
    if has_scores:
        css, ewss, _dc = compute_css_ewss(checklist_results)
        return ewss
    matches = sum(1 for v in checklist_results.values()
                  if isinstance(v, dict) and v.get("match", False))
    return matches / len(checklist_results)


def compute_css_ewss(
    checklist_results: dict,
    ssr_criteria: list | None = None,
) -> tuple[float, float, int]:
    """Compute Conservative Similarity Score and Evidence-Weighted Similarity Score.

    CSS: unknown/absent treated as 0, denominator = all criteria
    EWSS: only criteria with score > 0 count in denominator

    When ssr_criteria is provided, uses weights from it. Otherwise uniform weights.
    """
    if not checklist_results:
        return 0.0, 0.0, 0

    weights = {}
    if ssr_criteria:
        for c in ssr_criteria:
            if isinstance(c, dict):
                key = c.get("criterion", c.get("id", ""))
                weights[key] = c.get("weight", 1.0)

    total_weight = 0.0
    css_num = 0.0
    ewss_num = 0.0
    ewss_denom = 0.0

    for key, val in checklist_results.items():
        if not isinstance(val, dict):
            continue
        score = val.get("score")
        if score is None:
            score = 2 if val.get("match", False) else 0

        w = _find_weight(key, weights) if weights else 1.0
        total_weight += w

        if score >= 2:
            css_num += w
            ewss_num += w
            ewss_denom += w
        elif score == 1:
            ewss_num += 0.5 * w
            ewss_denom += w
        # score == 0: contributes to total_weight but not numerators

    css = css_num / total_weight if total_weight > 0 else 0.0
    ewss = ewss_num / ewss_denom if ewss_denom > 0 else 0.0
    denom_count = sum(
        1 for v in checklist_results.values()
        if isinstance(v, dict) and (v.get("score") or 0) > 0
    )
    return round(css, 4), round(ewss, 4), denom_count


def _find_weight(key: str, weights: dict) -> float:
    if key in weights:
        return weights[key]
    key_lower = key.lower().strip()
    for wk, wv in weights.items():
        if wk.lower().strip() == key_lower:
            return wv
        if key_lower[:40] in wk.lower() or wk.lower()[:40] in key_lower:
            return wv
    return 1.0


def classify_risk(score: float) -> str:
    """
    Classify risk level based on highest prior art score.
    Deterministic threshold-based.
    """
    for level in ("HIGH", "MEDIUM", "LOW"):
        if score > RISK_THRESHOLDS[level]:
            return level
    return "CLEAR"


def risk_color(level: str) -> str:
    """Return hex color for risk level. Deterministic."""
    return {
        "HIGH": "#ff3b30",
        "MEDIUM": "#ff9500",
        "LOW": "#34c759",
        "CLEAR": "#30d158",
    }.get(level, "#8899aa")


def score_color(score: float) -> str:
    """Return hex color for a similarity score. Deterministic."""
    if score > 0.7:
        return "#ff3b30"
    if score > 0.4:
        return "#ff9500"
    return "#34c759"


def aggregate_evaluations(eval_batches: list[dict]) -> dict:
    """
    Merge multiple evaluation batch files into a single scoring report.
    Sort by total_score descending.

    Input: list of batch dicts, each with "evaluations" key
    Output: {
        "scoring_report": [...sorted evaluations...],
        "stats": {...}
    }
    """
    all_evals = []
    for batch in eval_batches:
        for ev in batch.get("evaluations", []):
            # Recompute total_score deterministically
            cr = ev.get("checklist_results", ev.get("similarity_categories", {}))
            ev["similarity_score"] = compute_total_score(cr)
            ev["similarity_categories"] = cr
            all_evals.append(ev)

    # Sort by score descending
    all_evals.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

    # Compute stats
    scores = [e.get("similarity_score", 0) for e in all_evals]
    top_score = max(scores) if scores else 0
    avg_score = sum(scores) / len(scores) if scores else 0

    # Rename fields for report compatibility
    for ev in all_evals:
        ev["id"] = ev.get("pub_num", ev.get("id", ""))
        ev["manuscript_type"] = ev.get("match_type", ev.get("manuscript_type", ""))

    return {
        "scoring_report": all_evals,
        "stats": {
            "total_evaluated": len(all_evals),
            "top_score": round(top_score, 4),
            "avg_score": round(avg_score, 4),
            "risk_level": classify_risk(top_score),
            "risk_color": risk_color(classify_risk(top_score)),
            "high_risk_count": sum(1 for s in scores if s > RISK_THRESHOLDS["HIGH"]),
            "medium_risk_count": sum(1 for s in scores if RISK_THRESHOLDS["MEDIUM"] < s <= RISK_THRESHOLDS["HIGH"]),
            "low_risk_count": sum(1 for s in scores if RISK_THRESHOLDS["LOW"] < s <= RISK_THRESHOLDS["MEDIUM"]),
            "clear_count": sum(1 for s in scores if s <= RISK_THRESHOLDS["LOW"]),
        },
    }


def merge_into_final_results(phase1: dict, phase2: dict, search: dict, evaluation: dict) -> dict:
    """
    Merge all phase outputs into final results.json for report generation.
    Fully deterministic — just data merging.
    """
    from datetime import datetime, timezone

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase1": phase1,
        "phase2": phase2,
        "search": {
            "groups": search.get("groups", []),
            "summary": search.get("summary", {}),
        },
        "evaluation": evaluation,
    }
