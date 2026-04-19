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


MAX_PREVIOUS_OUTPUT_CHARS = 400
MAX_RAW_REPAIR_CHARS = 2000


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


def _truncate_text(text: str, limit: int) -> str:
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...(truncated)"


def _sample_input_line_count(problem_statement: Dict[str, Any]) -> int:
    sample_input = _clean_text(str(problem_statement.get("sample_input", "")))
    if not sample_input:
        return 0
    return len(sample_input.split("\n"))


def _estimate_input_calls(code: str) -> int:
    return code.count("input(") + code.count("stdin.readline(")


def _reads_all_input(code: str) -> bool:
    return any(token in code for token in ["sys.stdin.read(", "sys.stdin.buffer.read("])


def _repair_multiline_input_code(
    problem_statement: Dict[str, Any],
    code: str,
) -> str:
    normalized_code = _clean_text(code)
    expected_lines = _sample_input_line_count(problem_statement)

    if expected_lines < 2:
        return code
    if _reads_all_input(normalized_code) or _estimate_input_calls(normalized_code) >= expected_lines:
        return code

    lines = normalized_code.split("\n")
    for index, line in enumerate(lines):
        stripped = line.strip()
        map_match = re.match(
            r"^([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)+)\s*=\s*map\(\s*(int|float)\s*,\s*input\(\)\.split\(\)\s*\)\s*$",
            stripped,
        )
        if map_match:
            variables = [item.strip() for item in map_match.group(1).split(",")]
            caster = map_match.group(2)
            if len(variables) == expected_lines:
                replacement = [f"{name} = {caster}(input().strip())" for name in variables]
                updated_lines = lines[:index] + replacement + lines[index + 1 :]
                candidate = "\n".join(updated_lines)
                try:
                    compile(candidate, "<rule_repaired_code>", "exec")
                    return candidate
                except Exception:
                    return code

        split_match = re.match(
            r"^([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)+)\s*=\s*input\(\)\.split\(\)\s*$",
            stripped,
        )
        if split_match:
            variables = [item.strip() for item in split_match.group(1).split(",")]
            if len(variables) == expected_lines:
                replacement = [f"{name} = input().strip()" for name in variables]
                updated_lines = lines[:index] + replacement + lines[index + 1 :]
                candidate = "\n".join(updated_lines)
                try:
                    compile(candidate, "<rule_repaired_code>", "exec")
                    return candidate
                except Exception:
                    return code

    return code


def _build_solution_retry_directives(
    problem_statement: Dict[str, Any],
    feedback_issues: List[str],
) -> List[str]:
    directives: List[str] = []
    input_format = _clean_text(str(problem_statement.get("input_format", "")))
    sample_input = _clean_text(str(problem_statement.get("sample_input", "")))
    line_count = _sample_input_line_count(problem_statement)

    if any("json_decode_failed" in issue for issue in feedback_issues):
        directives.append("必须只输出一个合法 JSON 对象，且顶层只有 language、explanation、code 三个字符串字段。")

    if any("reference_code_compile_failed" in issue or "python_code_compile_failed" in issue for issue in feedback_issues):
        directives.append("code 必须是可直接编译运行的完整 Python 程序，不要输出半成品、伪代码或损坏字符串。")

    if any("reference_code_input_read_missing" in issue for issue in feedback_issues):
        directives.append("code 必须显式从标准输入读取数据。")

    if any("reference_code_output_write_missing" in issue for issue in feedback_issues):
        directives.append("code 必须显式向标准输出打印结果。")

    if any("missing_or_empty_field: language" in issue for issue in feedback_issues):
        directives.append('language 字段必须是字符串 "python"。')

    if any("missing_or_empty_field: code" in issue for issue in feedback_issues):
        directives.append(
            "code 字段不能为空。必须输出完整可运行的 Python 程序，不要返回 insufficient_information 或空字符串。"
        )

    if any("missing_or_empty_field: explanation" in issue for issue in feedback_issues):
        directives.append("explanation 字段不能为空，使用 1 到 2 句话说明解法。")

    if any("reference_code_input_line_count_mismatch" in issue for issue in feedback_issues):
        directives.append(
            "题面输入结构与代码读取方式必须一致。不要把多行输入误写成单次 input().split() 读取。"
        )
        if input_format:
            directives.append(f"当前 input_format：{input_format}")
        if sample_input:
            directives.append(f"当前 sample_input：\n{sample_input}")
        if line_count >= 2:
            directives.append(
                f"sample_input 明显有 {line_count} 行。code 必须至少逐行读取 {line_count} 次，或者一次性读取全部输入后按行解析。"
            )

    if any("solution_explanation_empty" in issue for issue in feedback_issues):
        directives.append("explanation 不能为空，使用 1 到 2 句话说明解法。")

    if not directives:
        directives.append("严格按题面重新生成合法 JSON，不要重复之前的错误。")

    return directives


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
    raw_output = _truncate_text(raw_output, MAX_RAW_REPAIR_CHARS)
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


def _regenerate_solution_json(
    problem_statement: Dict[str, Any],
    raw_output: str = "",
    feedback_issues: List[str] | None = None,
) -> str:
    prompt = build_solution_prompt(problem_statement)
    directives = _build_solution_retry_directives(problem_statement, feedback_issues or [])
    prompt += (
        "\n\n上一次输出存在格式或内容问题。请忽略损坏内容，直接重新生成一个合法 JSON 对象。"
        "\n修复要求：\n"
        + "\n".join(f"{i + 1}. {directive}" for i, directive in enumerate(directives))
        + "\n不要输出任何额外说明。"
    )
    if raw_output.strip():
        prompt += "\n上一次输出摘要：\n" + _truncate_text(raw_output, MAX_PREVIOUS_OUTPUT_CHARS)

    llm = SiliconFlowLLM()
    regenerated = llm.chat(
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
        temperature=0.0,
        max_tokens=1000,
    )
    return regenerated


def _build_solution_retry_prompt(
    problem_statement: Dict[str, Any],
    feedback_issues: List[str],
    previous_output: str,
) -> str:
    base_prompt = build_solution_prompt(problem_statement)
    issues_text = "\n".join(f"- {issue}" for issue in feedback_issues) if feedback_issues else "- previous_output_invalid"
    previous_output = _truncate_text(previous_output, MAX_PREVIOUS_OUTPUT_CHARS) if previous_output.strip() else "(empty)"
    directives = _build_solution_retry_directives(problem_statement, feedback_issues)
    return (
        f"{base_prompt}\n\n"
        "上一次参考答案存在问题，请只根据下面错误定向修复，不要重复错误。\n"
        "错误列表：\n"
        f"{issues_text}\n\n"
        "额外硬约束：\n"
        + "\n".join(f"{i + 1}. {directive}" for i, directive in enumerate(directives))
        + "\n\n上一次输出摘要：\n"
        f"{previous_output}\n"
    )


def generate_solution_output(
    problem_statement: Dict[str, Any],
    feedback_issues: List[str] | None = None,
    previous_output: str = "",
) -> str:
    if feedback_issues:
        prompt = _build_solution_retry_prompt(
            problem_statement=problem_statement,
            feedback_issues=feedback_issues,
            previous_output=previous_output,
        )
        temperature = 0.0
        max_tokens = 800
    else:
        prompt = build_solution_prompt(problem_statement)
        temperature = 0.1
        max_tokens = 1400

    llm = SiliconFlowLLM()
    return llm.chat(
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
        temperature=temperature,
        max_tokens=max_tokens,
    )


def validate_solution_obj(solution_obj: Dict[str, Any]) -> Dict[str, Any]:
    return _validate_solution_obj(solution_obj)


def run_solution_agent(
    problem_statement: Dict[str, Any],
    feedback_issues: List[str] | None = None,
    previous_output: str = "",
) -> Dict[str, Any]:
    raw_output = generate_solution_output(
        problem_statement=problem_statement,
        feedback_issues=feedback_issues,
        previous_output=previous_output,
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
            regenerated_output = _regenerate_solution_json(
                problem_statement,
                raw_output,
                feedback_issues=feedback_issues,
            )
            regenerated_json_text = _extract_json_object(regenerated_output)
            regenerated_json_text = _sanitize_json_like_text(regenerated_json_text)

            try:
                parsed_obj = json.loads(regenerated_json_text)
                raw_output = regenerated_output
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

        if not validation["valid"]:
            regenerated_output = _regenerate_solution_json(
                problem_statement,
                raw_output,
                feedback_issues=feedback_issues or validation["errors"],
            )
            regenerated_json_text = _extract_json_object(regenerated_output)
            regenerated_json_text = _sanitize_json_like_text(regenerated_json_text)

            try:
                regenerated_obj = json.loads(regenerated_json_text)
                regenerated_validation = _validate_solution_obj(regenerated_obj)
                if regenerated_validation["valid"]:
                    parsed_obj = regenerated_obj
                    validation = regenerated_validation
                    raw_output = regenerated_output
            except json.JSONDecodeError:
                pass

    if validation["valid"]:
        normalized = _normalize_solution_obj(parsed_obj)
        repaired_code = _repair_multiline_input_code(
            problem_statement,
            normalized["code"],
        )
        if repaired_code != normalized["code"]:
            normalized["code"] = repaired_code
            raw_output = json.dumps(
                {
                    "language": normalized["language"] or "python",
                    "explanation": normalized["explanation"],
                    "code": normalized["code"],
                },
                ensure_ascii=False,
            )
    else:
        normalized = {
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
