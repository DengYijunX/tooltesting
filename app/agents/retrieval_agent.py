from typing import Dict, Any, List, Optional

from app.config import INDEX_DIR, RECALL_K, FINAL_TOP_N, ENABLED_SUBJECTS
from app.retriever import retrieve_with_scores
from app.reranker import SiliconFlowReranker


def run_retrieval_agent(topic: str, subject: Optional[str] = None) -> Dict[str, Any]:
    if subject and subject not in ENABLED_SUBJECTS:
        return {
            "retrieved_docs": [],
            "retrieval_summary": f"当前学科 {subject} 尚未接入知识库。",
            "status": "not_enabled",
            "subject": subject,
        }

    metadata_filter = None
    if subject and subject != "all":
        metadata_filter = {"subject": subject}

    docs_and_scores = retrieve_with_scores(
        query=topic,
        save_path=INDEX_DIR,
        k=RECALL_K,
        metadata_filter=metadata_filter,
    )

    recalled_docs = []
    for doc, score in docs_and_scores:
        doc.metadata["faiss_score"] = float(score)
        recalled_docs.append(doc)

    reranker = SiliconFlowReranker()
    final_docs = reranker.rerank(
        query=topic,
        docs=recalled_docs,
        top_n=FINAL_TOP_N,
    )

    serialized_docs: List[Dict[str, Any]] = []
    for doc in final_docs:
        serialized_docs.append({
            "content": doc.page_content,
            "metadata": {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "subject": doc.metadata.get("subject"),
                "file_type": doc.metadata.get("file_type"),
                "faiss_score": doc.metadata.get("faiss_score"),
                "rerank_score": doc.metadata.get("rerank_score"),
            }
        })

    summary_lines = []
    for i, d in enumerate(serialized_docs, 1):
        meta = d["metadata"]
        summary_lines.append(
            f"[片段{i}] source={meta.get('source')} chunk={meta.get('chunk_id')} "
            f"subject={meta.get('subject')} rerank_score={meta.get('rerank_score')}"
        )

    return {
        "retrieved_docs": serialized_docs,
        "retrieval_summary": "\n".join(summary_lines),
        "status": "ok",
        "subject": subject or "all",
    }