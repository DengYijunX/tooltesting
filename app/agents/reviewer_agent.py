from typing import Dict, Any, List

from app.llm import SiliconFlowLLM


def _format_docs_for_review(docs: List[Dict[str, Any]]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d["metadata"]
        ref = f"{meta.get('source')} | chunk={meta.get('chunk_id')}"
        parts.append(f"[资料片段{i}] {ref}\n{d['content']}")
    return "\n\n".join(parts)


def run_reviewer_agent(topic: str, retrieved_docs: List[Dict[str, Any]], draft_problem: str) -> Dict[str, Any]:
    context = _format_docs_for_review(retrieved_docs)

    prompt = f"""你是题目审核代理。
请审核下面这道题是否满足要求。

审核标准：
1. 是否基于给定资料
2. 是否为真正的编程题
3. 输入输出是否明确
4. 规则是否有歧义
5. 是否避免了真实临床诊断

请按如下格式输出：

# 审核结论
通过 或 不通过

# 问题列表
（若通过可写“无”）

# 修改建议
（若通过可写“无”）

资料：
{context}

题目草案：
{draft_problem}
"""

    llm = SiliconFlowLLM()
    result = llm.chat([
        {"role": "system", "content": "你是严格的题目审核代理。"},
        {"role": "user", "content": prompt},
    ])

    passed = "通过" in result and "不通过" not in result

    return {
        "review_result": result,
        "review_passed": passed
    }