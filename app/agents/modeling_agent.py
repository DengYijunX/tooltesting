from typing import Dict, Any, List

from app.llm import SiliconFlowLLM


def _format_docs(docs: List[Dict[str, Any]]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d["metadata"]
        ref = f"{meta.get('source')} | chunk={meta.get('chunk_id')}"
        parts.append(f"[资料片段{i}] {ref}\n{d['content']}")
    return "\n\n".join(parts)


def run_modeling_agent(topic: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    context = _format_docs(retrieved_docs)

    prompt = f"""你是知识建模代理。
请根据资料，围绕主题“{topic}”，输出适合生成编程题的知识骨架。

要求：
1. 不要直接写完整题面。
2. 要把学科知识转成可编程处理的结构。
3. 不要涉及真实临床诊断。
4. 输出必须包含以下部分：

# 核心概念
# 可量化对象
# 可规则化条件
# 适合的题型
# 推荐的输入输出设计
# 一个建议题目方向

资料如下：
{context}
"""

    llm = SiliconFlowLLM()
    result = llm.chat([
        {"role": "system", "content": "你是严谨的知识建模代理。"},
        {"role": "user", "content": prompt},
    ])

    return {"knowledge_schema": result}