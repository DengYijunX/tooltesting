import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from app.config import INDEX_DIR, RECALL_K, FINAL_TOP_N, validate_config
from app.retriever import retrieve_with_scores
from app.reranker import SiliconFlowReranker
from app.qa import answer_with_sources


def main():
    validate_config()

    question = input("请输入问题：").strip()
    if not question:
        print("问题不能为空。")
        return

    print("\n===== 第一步：FAISS 初始召回 =====")
    docs_and_scores = retrieve_with_scores(
        query=question,
        save_path=INDEX_DIR,
        k=RECALL_K,
    )

    recalled_docs = []
    for i, (doc, score) in enumerate(docs_and_scores, 1):
        doc.metadata["faiss_score"] = float(score)
        recalled_docs.append(doc)
        print(
            f"{i}. faiss_score={score:.4f} | "
            f"source={doc.metadata.get('source')} | "
            f"page={doc.metadata.get('page')} | "
            f"subject={doc.metadata.get('subject')}"
        )

    print("\n===== 第二步：Rerank 重排 =====")
    reranker = SiliconFlowReranker()
    final_docs = reranker.rerank(
        query=question,
        docs=recalled_docs,
        top_n=FINAL_TOP_N,
    )

    for i, doc in enumerate(final_docs, 1):
        print(
            f"{i}. rerank_score={doc.metadata.get('rerank_score', 0.0):.4f} | "
            f"source={doc.metadata.get('source')} | "
            f"page={doc.metadata.get('page')} | "
            f"subject={doc.metadata.get('subject')}"
        )

    print("\n===== 第三步：生成答案 =====")
    answer = answer_with_sources(question, final_docs)
    print(answer)


if __name__ == "__main__":
    main()