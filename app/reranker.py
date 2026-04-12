from typing import List
import requests
from langchain_core.documents import Document

from app.config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    RERANK_MODEL,
    validate_config,
)


class SiliconFlowReranker:
    def __init__(self):
        validate_config()
        self.api_key = SILICONFLOW_API_KEY
        self.base_url = SILICONFLOW_BASE_URL.rstrip("/")
        self.model = RERANK_MODEL
        self.endpoint = f"{self.base_url}/rerank"

    def rerank(self, query: str, docs: List[Document], top_n: int = 4) -> List[Document]:
        if not docs:
            return []

        documents = [doc.page_content for doc in docs]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        # 兼容不同返回字段风格
        results = data.get("results") or data.get("data") or []
        reranked_docs: List[Document] = []

        for item in results:
            idx = item.get("index", item.get("document_index"))
            score = item.get("relevance_score", item.get("score", 0.0))
            if idx is None or idx < 0 or idx >= len(docs):
                continue

            doc = docs[idx]
            doc.metadata["rerank_score"] = float(score)
            reranked_docs.append(doc)

        return reranked_docs