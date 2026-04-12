from typing import List
from langchain_core.documents import Document

from app.llm import SiliconFlowLLM


def format_context(docs: List[Document]) -> str:
    blocks = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        chunk_id = doc.metadata.get("chunk_id")
        faiss_score = doc.metadata.get("faiss_score")
        rerank_score = doc.metadata.get("rerank_score")

        ref = f"{source}"
        if page is not None:
            ref += f" | page={page}"
        if chunk_id is not None:
            ref += f" | chunk={chunk_id}"
        if faiss_score is not None:
            ref += f" | faiss_score={faiss_score:.4f}"
        if rerank_score is not None:
            ref += f" | rerank_score={rerank_score:.4f}"

        blocks.append(f"[片段{i}] {ref}\n{doc.page_content}")

    return "\n\n".join(blocks)


def answer_with_sources(question: str, docs: List[Document]) -> str:
    context = format_context(docs)

    prompt = f"""你是一个基于检索上下文作答的助手。
请严格依据给定上下文回答，不要编造。
如果上下文不足以回答，请明确写出“根据当前检索到的资料，无法确定”。

问题：
{question}

上下文：
{context}

请按以下结构输出：
1. 简明答案
2. 依据说明
3. 引用来源（列出片段编号与文件名）
"""

    llm = SiliconFlowLLM()
    messages = [
        {"role": "system", "content": "你是一个严格依据检索资料回答问题的助手。"},
        {"role": "user", "content": prompt},
    ]
    return llm.chat(messages)