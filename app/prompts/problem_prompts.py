import json
from typing import Any, Dict


def _format_task_model(task_model: Dict[str, Any] | None) -> str:
    if not task_model:
        return "未提供任务模型。请按主题和背景生成可判题的标准编程题。"
    return json.dumps(task_model, ensure_ascii=False, indent=2)


def build_problem_prompt(
    topic: str,
    problem_background: str,
    retrieval_summary: str = "",
    task_model: Dict[str, Any] | None = None,
) -> str:
    task_model_text = _format_task_model(task_model)
    return f"""你是“标准编程题题面草稿生成代理”。

你的任务是：围绕主题“{topic}”，根据给定背景、检索摘要和任务模型，生成一道标准编程题的简洁草稿。

严格要求：
1. 题目的核心考点必须严格对应主题“{topic}”。
2. 必须是一道可判题的编程题，不能是问答题或开放题。
3. 输入输出必须明确、唯一、无歧义。
4. 不要输出 JSON。
5. 不要输出 Markdown 代码块。
6. 不要输出 LaTeX 公式写法，例如 \\(F = ma\\)，请直接写成普通文本：F = ma。
7. 只允许按下面固定格式输出，不要增加任何额外内容。
8. 不要模拟对话，不要出现 user、assistant、用户、助手、答案 等角色标记。
9. 样例输入和样例输出只能写纯样例内容，不要加引号，不要加解释，不要额外补一行引号或说明文字。
10. 如果输入格式写了“两行”或“第二行”，样例输入必须真的分成多行。
11. 数据范围不能为空；如果输入格式已经包含取值范围，请把它整理到“数据范围”字段。
12. 题目必须包含明确计算规则或公式，说明如何从输入得到输出；不能只写“模拟”“计算最终结果”。
13. 题目必须围绕数据结构与算法生成；学科知识可以作为背景，但最终必须能用程序判题。
14. 如果任务模型中 has_formula 为 true，题面必须直接使用 knowledge_rule 作为计算规则，不要自行发明其他公式。
15. 如果任务模型中 requested_algorithm 不是 auto，必须优先满足 algorithm_model；公式只能作为每条记录、每个事件或每个查询里的计算步骤。
16. 如果任务模型中 strategy 为 contextual_algorithm，学科知识只作为软背景，重点放在计数、排序、模拟、前缀和、二分、动态规划等可判题算法规则。
17. 样例输入和样例输出必须只包含原始数据，不要写“一行”“两行”“输入：”“输出：”等说明文字。
18. 如果 algorithm_model 为 sorting，输入格式必须包含第一行 n，后面 n 行记录；样例输入第一行也必须是 n。
19. 如果输出编号，样例输出中的编号必须来自样例输入中真实存在的编号，不能输出不存在的编号。
20. 如果任务模型提供 sample_rows 和 sample_outputs，优先使用它们构造样例，不要随意改数字。
21. 如果资料不足以支持可靠出题，请输出：
题目名称：资料不足
题目描述：资料不足，无法可靠生成题目。

请严格按下面格式输出：

题目名称：...
题目描述：...
输入格式：...
输出格式：...
样例输入：...
样例输出：...
数据范围：...

背景如下：
{problem_background}

检索摘要如下：
{retrieval_summary}

任务模型如下：
{task_model_text}
"""
