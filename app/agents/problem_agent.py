import re
from typing import Dict, Any, List

from app.llm import SiliconFlowLLM
from app.prompts.problem_prompts import build_problem_prompt


REQUIRED_FIELDS = [
    "title",
    "description",
    "input_format",
    "output_format",
    "sample_input",
    "sample_output",
    "constraints",
]


def _clean_text(text: str) -> str:
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")


def _sanitize_output(text: str) -> str:
    text = _clean_text(text)

    # 去掉可能混入的 Markdown 代码块
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # 去掉常见 LaTeX 包裹
    text = text.replace(r"\(", "(")
    text = text.replace(r"\)", ")")
    text = text.replace(r"\[", "[")
    text = text.replace(r"\]", "]")

    # 去掉多余的星号强调
    text = text.replace("**", "")

    return text.strip()


def _extract_field(text: str, field_name: str, next_fields: List[str]) -> str:
    next_pattern = "|".join(re.escape(f) for f in next_fields) if next_fields else r"\Z"
    pattern = rf"{re.escape(field_name)}：\s*(.*?)(?=\n(?:{next_pattern})：|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_problem_draft(text: str) -> Dict[str, Any]:
    text = _sanitize_output(text)

    field_labels = [
        "题目名称",
        "题目描述",
        "输入格式",
        "输出格式",
        "样例输入",
        "样例输出",
        "数据范围",
    ]

    parsed = {
        "title": _extract_field(text, "题目名称", field_labels[1:]),
        "description": _extract_field(text, "题目描述", field_labels[2:]),
        "input_format": _extract_field(text, "输入格式", field_labels[3:]),
        "output_format": _extract_field(text, "输出格式", field_labels[4:]),
        "sample_input": _extract_field(text, "样例输入", field_labels[5:]),
        "sample_output": _extract_field(text, "样例输出", field_labels[6:]),
        "constraints": _extract_field(text, "数据范围", []),
        "sample_explanation": "",
    }

    return parsed


def _validate_problem_statement(obj: Dict[str, Any]) -> Dict[str, Any]:
    errors = []

    if not isinstance(obj, dict):
        return {
            "valid": False,
            "errors": ["problem_statement_not_dict"],
        }

    for field in REQUIRED_FIELDS:
        value = obj.get(field, "")
        if not isinstance(value, str) or not value.strip():
            errors.append(f"missing_or_empty_field: {field}")

    if obj.get("title", "").strip() == "资料不足":
        errors.append("insufficient_information")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def _repair_problem_draft(raw_output: str) -> str:
    repair_prompt = f"""请把下面内容修复为严格符合格式的题面草稿。

要求：
1. 不要输出 JSON
2. 不要输出 Markdown
3. 不要输出代码块
4. 必须严格只包含这 7 行字段：
题目名称：...
题目描述：...
输入格式：...
输出格式：...
样例输入：...
样例输出：...
数据范围：...

待修复内容：
{raw_output}
"""

    llm = SiliconFlowLLM()
    repaired = llm.chat(
        messages=[
            {
                "role": "system",
                "content": "你是题面草稿修复代理，只输出固定格式题面草稿。",
            },
            {
                "role": "user",
                "content": repair_prompt,
            },
        ],
        temperature=0.0,
        max_tokens=700,
    )
    return repaired


def run_problem_agent(
    topic: str,
    problem_background: str,
    retrieval_summary: str = ""
) -> Dict[str, Any]:
    prompt = build_problem_prompt(
        topic=topic,
        problem_background=problem_background,
        retrieval_summary=retrieval_summary,
    )

    llm = SiliconFlowLLM()
    raw_output = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个严谨的标准编程题题面草稿生成代理。"
                    "你只输出固定格式题面草稿，不输出额外内容。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
        max_tokens=700,
    )

    parsed = parse_problem_draft(raw_output)
    validation = _validate_problem_statement(parsed)

    if not validation["valid"]:
        repaired_output = _repair_problem_draft(raw_output)
        repaired_parsed = parse_problem_draft(repaired_output)
        repaired_validation = _validate_problem_statement(repaired_parsed)

        if repaired_validation["valid"]:
            parsed = repaired_parsed
            validation = repaired_validation
            raw_output = repaired_output

    return {
        "problem_statement": parsed if validation["valid"] else {},
        "raw_problem_output": raw_output,
        "problem_statement_valid": validation["valid"],
        "problem_statement_errors": validation["errors"],
    }