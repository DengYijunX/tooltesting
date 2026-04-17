import json


def build_solution_prompt(problem_statement: dict) -> str:
    title = problem_statement.get("title", "").strip()
    description = problem_statement.get("description", "").strip()
    input_format = problem_statement.get("input_format", "").strip()
    output_format = problem_statement.get("output_format", "").strip()
    constraints = problem_statement.get("constraints", "").strip()
    sample_input = problem_statement.get("sample_input", "").strip()
    sample_output = problem_statement.get("sample_output", "").strip()
    sample_explanation = problem_statement.get("sample_explanation", "").strip()

    schema_example = {
        "language": "python",
        "explanation": "先根据题意分析输入输出关系，再写出直接求解逻辑。",
        "code": "F, m = map(float, input().split())\na = F / m\nprint(f\"{a:.2f}\")"
    }

    return f"""你是“编程题参考答案生成代理”。

你的任务是：根据给定题面，生成这道题的参考解法与 Python 标准答案。

严格要求：
1. 必须严格依据题面，不得改题。
2. 必须生成可直接运行的 Python 代码。
3. 代码必须从标准输入读取数据，向标准输出打印结果。
4. 不要输出伪代码。
5. 不要输出 Markdown，不要输出代码块，不要输出额外解释。
6. explanation 字段写简洁思路说明。
7. code 字段只放完整 Python 代码字符串。
8. 如果题面信息不足以生成可靠答案，请只输出：
{{"error": "insufficient_information"}}

你必须只输出一个合法 JSON 对象。

输出 JSON 示例：
{json.dumps(schema_example, ensure_ascii=False, indent=2)}

下面是题目信息：

# title
{title}

# description
{description}

# input_format
{input_format}

# output_format
{output_format}

# constraints
{constraints}

# sample_input
{sample_input}

# sample_output
{sample_output}

# sample_explanation
{sample_explanation}
"""