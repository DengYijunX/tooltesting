from typing import Dict, Any

from app.llm import SiliconFlowLLM


def run_drafting_agent(topic: str, knowledge_schema: str) -> Dict[str, Any]:
    prompt = f"""你是编程题命题代理。
请根据下面的知识骨架，围绕主题“{topic}”生成一道可判题的编程题。

要求：
1. 保留学科背景，但题目本质必须是编程题。
2. 输入输出必须清晰、唯一可判定。
3. 难度控制在基础到中等。
4. 不要写成心理学论述题。

请输出以下结构：

# 题目名称
# 题目背景
# 任务描述
# 输入格式
# 输出格式
# 样例输入
# 样例输出
# 样例说明
# 数据范围

知识骨架如下：
{knowledge_schema}
"""

    llm = SiliconFlowLLM()
    result = llm.chat([
        {"role": "system", "content": "你是严谨的编程题命题代理。"},
        {"role": "user", "content": prompt},
    ])

    return {"draft_problem": result}