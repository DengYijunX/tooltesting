from typing import Dict, Any, List

from app.llm import SiliconFlowLLM


def _format_docs(docs: List[Dict[str, Any]]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d["metadata"]
        ref = f"{meta.get('source')} | chunk={meta.get('chunk_id')}"
        parts.append(f"[资料片段{i}] {ref}\n{d['content']}")
    return "\n\n".join(parts)


def run_background_agent(topic: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    context = _format_docs(retrieved_docs)

    prompt = f"""你是题目背景生成代理。
请根据下面资料，围绕主题“{topic}”生成简洁、可靠、可追溯的题目背景。

要求：
1. 背景必须基于资料。
2. 不要编造超出资料的内容。
3. 控制在一小段即可。

资料：
{context}
"""

    llm = SiliconFlowLLM()
    result = llm.chat(
        [
            {"role": "system", "content": "你是一个严谨的题目背景生成代理。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )

    return {
        "problem_background": result,
        "background_evidence_summary": "背景依据来自当前检索片段。"
    }