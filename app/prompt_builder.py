from typing import List
from langchain_core.documents import Document


def format_docs_for_generation(docs: List[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        chunk_id = doc.metadata.get("chunk_id")

        ref = f"{source}"
        if page is not None:
            ref += f" | page={page}"
        if chunk_id is not None:
            ref += f" | chunk={chunk_id}"

        blocks.append(f"[资料片段{i}] {ref}\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_problem_generation_prompt(topic: str, docs: List[Document]) -> str:
    context = format_docs_for_generation(docs)

    prompt = f"""你是一个“学科知识驱动的编程题生成助手”。

现在需要根据给定的学科知识资料，围绕主题“{topic}”生成一道编程题。

要求如下：
1. 题目必须以给定资料为背景，不能脱离资料随意编造。
2. 题目本质上必须是一道“可编程求解”的题，而不是心理学问答题。
3. 题目规则必须明确，输入输出必须唯一可判定。
4. 尽量把学科知识转化为：
   - 分类判断
   - 统计分析
   - 规则模拟
   - 数据处理
5. 不要设计成医学诊断题，不要涉及真实临床判断。
6. 题目难度控制在“基础到中等编程题”。
7. 输出时使用规范格式。

请严格按照下面结构输出：

# 题目名称

# 题目背景
（结合资料，简要说明背景）

# 任务描述
（明确要求程序做什么）

# 输入格式

# 输出格式

# 样例输入

# 样例输出

# 样例说明

# 数据范围

# 出题依据
（说明这道题主要依据了哪些资料片段）

以下是可用资料：
{context}
"""
    return prompt