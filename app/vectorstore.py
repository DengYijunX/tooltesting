from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.embeddings import SiliconFlowEmbeddings


def build_faiss_from_documents(docs: List[Document]) -> FAISS:
    embeddings = SiliconFlowEmbeddings()
    return FAISS.from_documents(docs, embeddings)


def save_faiss(vectorstore: FAISS, save_path: str) -> None:
    vectorstore.save_local(save_path)


def load_faiss(save_path: str) -> FAISS:
    embeddings = SiliconFlowEmbeddings()
    return FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )