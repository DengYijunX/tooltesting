from typing import Any, Dict, List


def check_knowledge_sufficiency(
    retrieved_docs: List[Dict[str, Any]],
    retrieval_summary: str,
    min_docs: int = 2,
    min_top_score: float = 0.5,
) -> Dict[str, Any]:
    issues: List[str] = []

    if len(retrieved_docs) < min_docs:
        issues.append("retrieved_docs_too_few")

    if not retrieval_summary.strip():
        issues.append("retrieval_summary_empty")

    top_score = None
    if retrieved_docs:
        top_score = retrieved_docs[0].get("metadata", {}).get("rerank_score")

    if top_score is None:
        issues.append("top_rerank_score_missing")
    else:
        try:
            if float(top_score) < min_top_score:
                issues.append("top_rerank_score_too_low")
        except (TypeError, ValueError):
            issues.append("top_rerank_score_invalid")

    source_count = len({
        doc.get("metadata", {}).get("source")
        for doc in retrieved_docs
        if doc.get("metadata", {}).get("source")
    })
    if source_count == 0:
        issues.append("retrieval_sources_missing")

    content_lengths = [
        len((doc.get("content") or "").strip())
        for doc in retrieved_docs
    ]
    if not any(length >= 80 for length in content_lengths):
        issues.append("retrieved_content_too_short")

    return {
        "knowledge_sufficiency": len(issues) == 0,
        "knowledge_sufficiency_issues": issues,
        "knowledge_stats": {
            "retrieved_doc_count": len(retrieved_docs),
            "source_count": source_count,
            "top_rerank_score": top_score,
            "max_content_length": max(content_lengths) if content_lengths else 0,
        },
    }
