from typing import Optional, Dict, List, Tuple
from langchain_core.documents import Document

from app.vectorstore import load_faiss


def retrieve(
    query: str,
    save_path: str,
    k: int = 4,
    metadata_filter: Optional[Dict] = None,
) -> List[Document]:
    vs = load_faiss(save_path)

    if metadata_filter:
        return vs.similarity_search(query, k=k, filter=metadata_filter)

    return vs.similarity_search(query, k=k)


def retrieve_with_scores(
    query: str,
    save_path: str,
    k: int = 4,
    metadata_filter: Optional[Dict] = None,
) -> List[Tuple[Document, float]]:
    vs = load_faiss(save_path)

    if metadata_filter:
        return vs.similarity_search_with_score(query, k=k, filter=metadata_filter)

    return vs.similarity_search_with_score(query, k=k)