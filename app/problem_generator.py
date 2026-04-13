from app.config import INDEX_DIR, RECALL_K, FINAL_TOP_N
from app.retriever import retrieve_with_scores
from app.reranker import SiliconFlowReranker
from app.llm import SiliconFlowLLM
from app.prompt_builder import build_problem_generation_prompt


def generate_problem(topic: str) -> str:
    # 1. 初始召回
    docs_and_scores = retrieve_with_scores(
        query=topic,
        save_path=INDEX_DIR,
        k=RECALL_K,
    )

    recalled_docs = []
    for doc, score in docs_and_scores:
        doc.metadata["faiss_score"] = float(score)
        recalled_docs.append(doc)

    # 2. rerank
    reranker = SiliconFlowReranker()
    final_docs = reranker.rerank(
        query=topic,
        docs=recalled_docs,
        top_n=FINAL_TOP_N,
    )

    # 3. 构造 prompt
    prompt = build_problem_generation_prompt(topic, final_docs)

    # 4. 调 LLM 生成题目
    llm = SiliconFlowLLM()
    messages = [
        {
            "role": "system",
            "content": "你是一个严谨的编程题命题助手，擅长将学科知识转化为可计算、可判题的编程题。"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    return llm.chat(messages)