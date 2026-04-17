import json
import re
from typing import Dict, Any, List

from app.llm import SiliconFlowLLM
from app.prompts.solution_prompts import build_solution_prompt


REQUIRED_FIELDS = [
    "language",
    "explanation",
    "code",
]


def _clean_text(text: str) -> str:
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")


def _extract_json_object(text: str) -> str:
    """
    尝试从模型输出中提取第一个 JSON 对象。
    """
    text = _clean_text(text)

    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0).strip()

    return text


def _sanitize_json_like_text(text: str) -> str:
    """
    清理常见的伪 JSON 问题。
    """
    text = _clean_text(text)

    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # 去掉常见 LaTeX 风格括号
    text = text.replace(r"\(", "(")
    text = text.replace(r"\)", ")")
    text = text.replace(r"\[", "[")
    text = text.replace(r"\]", "]")

    # 去掉非法反斜杠，只保留 JSON 常见合法转义
    text = re.sub(r'\\(?!["\\/bfnrtu])', '', text)

    return text.strip()


def _validate_solution_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []

    if not isinstance(obj, dict):
        return {
            "valid": False,
            "errors": ["solution_object_not_dict"],
        }

    if obj.get("error") == "insufficient_information":
        return {
            "valid": False,
            "errors": ["insufficient_information"],
        }

    for field in REQUIRED_FIELDS:
        value = obj.get(field, "")
        if not isinstance(value, str) or not value.strip():
            errors.append(f"missing_or_empty_field: {field}")

    language = str(obj.get("language", "")).strip().lower()
    if language and language != "python":
        errors.append("language_not_python")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def _normalize_solution_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "language": "",
        "explanation": "",
        "code": "",
    }
    for field in REQUIRED_FIELDS:
        value = obj.get(field, "")
        normalized[field] = value.strip() if isinstance(value, str) else ""
    return normalized


def _repair_solution_json(raw_output: str) -> str:
    repair_prompt = f"""请把下面内容修复成一个合法 JSON 对象。

要求：
1. 只输出 JSON 对象
2. 必须包含以下字段：
language, explanation, code
3. 每个字段的值都必须是字符串
4. language 必须为 "python"
5. 不要输出 Markdown，不要输出解释
6. 如果内容中出现 ```python 代码块，请把代码提取到 code 字段中
7. 如果内容中出现 \\( \\) \\[ \\] 等公式包裹，请改写成普通文本，不要保留反斜杠

待修复内容：
{raw_output}
"""

    llm = SiliconFlowLLM()
    repaired = llm.chat(
        messages=[
            {
                "role": "system",
                "content": "你是 JSON 修复代理，只输出合法 JSON。"
            },
            {
                "role": "user",
                "content": repair_prompt
            },
        ],
        temperature=0.0,
        max_tokens=1200,
    )
    return repaired


def run_solution_agent(problem_statement: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_solution_prompt(problem_statement)

    llm = SiliconFlowLLM()
    raw_output = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个严谨的编程题参考答案生成代理。"
                    "你只输出合法 JSON，不输出任何额外内容。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
        max_tokens=1400,
    )

    parsed_json_text = _extract_json_object(raw_output)
    parsed_json_text = _sanitize_json_like_text(parsed_json_text)

    try:
        parsed_obj = json.loads(parsed_json_text)
    except json.JSONDecodeError:
        repaired_output = _repair_solution_json(raw_output)
        repaired_json_text = _extract_json_object(repaired_output)
        repaired_json_text = _sanitize_json_like_text(repaired_json_text)

        try:
            parsed_obj = json.loads(repaired_json_text)
            raw_output = repaired_output
        except json.JSONDecodeError:
            return {
                "language": "python",
                "solution_explanation": "",
                "reference_code": "",
                "raw_solution_output": raw_output,
                "solution_valid": False,
                "solution_errors": ["json_decode_failed"],
            }

    validation = _validate_solution_obj(parsed_obj)

    if not validation["valid"]:
        repaired_output = _repair_solution_json(
            json.dumps(parsed_obj, ensure_ascii=False)
        )
        repaired_json_text = _extract_json_object(repaired_output)
        repaired_json_text = _sanitize_json_like_text(repaired_json_text)

        try:
            repaired_obj = json.loads(repaired_json_text)
            repaired_validation = _validate_solution_obj(repaired_obj)

            if repaired_validation["valid"]:
                parsed_obj = repaired_obj
                validation = repaired_validation
                raw_output = repaired_output
        except json.JSONDecodeError:
            pass

    normalized = _normalize_solution_obj(parsed_obj) if validation["valid"] else {
        "language": "python",
        "explanation": "",
        "code": "",
    }

    return {
        "language": normalized["language"] or "python",
        "solution_explanation": normalized["explanation"],
        "reference_code": normalized["code"],
        "raw_solution_output": raw_output,
        "solution_valid": validation["valid"],
        "solution_errors": validation["errors"],
    }