import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from app.config import INDEX_DIR, SUPPORTED_SUBJECTS, ENABLED_SUBJECTS
from app.retriever import retrieve_with_scores
from app.reranker import SiliconFlowReranker
from app.qa import answer_with_sources


def choose_subject():
    print("请选择检索范围：")
    print("1. psychology")
    print("2. physics")
    print("3. math")
    print("4. finance")
    print("5. mixed")
    print("6. all")

    choice = input("请输入选项编号：").strip()

    mapping = {
        "1": "psychology",
        "2": "physics",
        "3": "math",
        "4": "finance",
        "5": "mixed",
        "6": "all",
    }
    subject = mapping.get(choice, "all")

    if subject not in SUPPORTED_SUBJECTS:
        subject = "all"

    return subject


def main():
    subject = choose_subject()
    question = input("请输入问题：").strip()

    if not question:
        print("问题不能为空。")
        return

    if subject not in ENABLED_SUBJECTS:
        print(f"\n当前学科 {subject} 尚未接入知识库。")
        print(f"当前已启用学科：{', '.join(ENABLED_SUBJECTS)}")
        return

    metadata_filter = None
    if subject != "all":
        metadata_filter = {"subject": subject}

    print("\n===== 第一步：FAISS 初始召回 =====")
    docs_and_scores = retrieve_with_scores(
        query=question,
        save_path=INDEX_DIR,
        k=8,
        metadata_filter=metadata_filter,
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
    final_docs = reranker.rerank(question, recalled_docs, top_n=4)

    for i, doc in enumerate(final_docs, 1):
        print(
            f"{i}. rerank_score={doc.metadata.get('rerank_score', 0.0):.4f} | "
            f"source={doc.metadata.get('source')} | "
            f"page={doc.metadata.get('page')} | "
            f"subject={doc.metadata.get('subject')}"
        )

    print("\n===== 第三步：生成答案 =====\n")
    answer = answer_with_sources(question, final_docs)
    print(answer)


if __name__ == "__main__":
    main()