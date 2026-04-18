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


FIELD_ALIASES = {
    "title": ["题目名称"],
    "description": ["题目描述", "任务描述"],
    "input_format": ["输入格式"],
    "output_format": ["输出格式"],
    "sample_input": ["样例输入"],
    "sample_output": ["样例输出"],
    "constraints": ["数据范围"],
}


ROLE_MARKERS = {"user", "assistant", "system", "用户", "助手", "答案"}
FIELD_LABEL_TOKENS = [
    "题目名称：",
    "题目描述：",
    "任务描述：",
    "输入格式：",
    "输出格式：",
    "样例输入：",
    "样例输出：",
    "数据范围：",
]
SUSPICIOUS_FRAGMENTS = [
    "请纠正",
    "符合输入格式",
    "符合输出格式",
    "不要输出",
    "根据给定背景",
    "检索摘要如下",
]


def _clean_text(text: str) -> str:
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")


def _input_format_expects_multiple_lines(input_format: str) -> bool:
    return "两行" in input_format or "第二行" in input_format


def _output_format_expects_single_line(output_format: str) -> bool:
    return "一行" in output_format or "1行" in output_format


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

    cleaned_lines = []
    for line in text.split("\n"):
        if line.strip().lower() in ROLE_MARKERS:
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    return text.strip()


def _trim_blank_lines(lines: List[str]) -> List[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _collapse_duplicate_halves(lines: List[str]) -> List[str]:
    collapsed = list(lines)
    while len(collapsed) >= 2 and len(collapsed) % 2 == 0:
        half = len(collapsed) // 2
        if collapsed[:half] == collapsed[half:]:
            collapsed = collapsed[:half]
            continue
        break
    return collapsed


def _collapse_adjacent_duplicates(lines: List[str]) -> List[str]:
    deduped: List[str] = []
    for line in lines:
        if deduped and line == deduped[-1]:
            continue
        deduped.append(line)
    return deduped


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1].strip()
    return stripped


def _normalize_sample_lines(lines: List[str]) -> List[str]:
    normalized: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in {"'", '"', "'''", '"""'}:
            continue
        # 去掉模型常混进样例里的“第一行：”“输入：”等标签，只保留字面值
        stripped = re.sub(
            r"^(?:第[一二三四五六七八九十\d]+行|样例输入|样例输出|输入|输出)\s*[：:]\s*",
            "",
            stripped,
        )
        normalized.append(_strip_wrapping_quotes(stripped))
    return _trim_blank_lines(normalized)


def _contains_suspicious_sample_content(value: str) -> bool:
    if not value.strip():
        return False
    lines = [line.strip() for line in _clean_text(value).split("\n")]
    if any(line in {"'", '"', "'''", '"""'} for line in lines):
        return True
    if any(re.match(r"^(?:第[一二三四五六七八九十\d]+行|输入|输出)\s*[：:]", line) for line in lines):
        return True
    if re.search(r"(^|\n)\s*['\"]", value) and not re.search(r"(^|\n)\s*[-+]?\d", value):
        return True
    if re.search(r"['\"]\s*($|\n)", value) and not re.search(r"\d\s*['\"]\s*($|\n)", value):
        return True
    return False


def _normalize_field_value(field_name: str, value: str) -> str:
    text = _sanitize_output(value)
    lines = [line.rstrip() for line in text.split("\n")]
    lines = _trim_blank_lines(lines)

    if field_name in {"title", "description", "input_format", "output_format", "constraints"}:
        lines = _collapse_duplicate_halves(lines)
        lines = _collapse_adjacent_duplicates(lines)
    elif field_name in {"sample_input", "sample_output"}:
        lines = _normalize_sample_lines(lines)

    if field_name == "title":
        first_non_empty = next((line.strip() for line in lines if line.strip()), "")
        return first_non_empty

    return "\n".join(lines).strip()


def _match_field_label(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    for key, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            for sep in ("：", ":"):
                prefix = f"{alias}{sep}"
                if stripped.startswith(prefix):
                    return key, stripped[len(prefix):].strip()
    return None


def _empty_buffers() -> Dict[str, List[str]]:
    return {
        "title": [],
        "description": [],
        "input_format": [],
        "output_format": [],
        "sample_input": [],
        "sample_output": [],
        "constraints": [],
    }


def _buffers_have_content(buffers: Dict[str, List[str]]) -> bool:
    return any(any(line.strip() for line in values) for values in buffers.values())


def _build_candidate_from_buffers(buffers: Dict[str, List[str]]) -> Dict[str, Any]:
    parsed = {}
    for key, lines in buffers.items():
        parsed[key] = _normalize_field_value(key, "\n".join(lines))
    parsed["sample_explanation"] = ""
    return parsed


def _contains_suspicious_content(value: str) -> bool:
    if not isinstance(value, str):
        return False

    lowered = value.lower()
    if any(marker in lowered for marker in ROLE_MARKERS):
        return True
    if any(token in value for token in FIELD_LABEL_TOKENS):
        return True
    if any(fragment in value for fragment in SUSPICIOUS_FRAGMENTS):
        return True
    if "```" in value:
        return True
    return False


def _looks_garbled(value: str) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    if re.search(r"[，,。]{2,}", value):
        return True
    if re.search(r"\b\d+[A-Za-z]{2,}\b|\b[A-Za-z]{2,}\d+\b", value):
        return True
    if re.search(r"\b(?:rer|mere)\b", value.lower()):
        return True
    return False


def _candidate_score(candidate: Dict[str, Any]) -> int:
    score = sum(1 for field in REQUIRED_FIELDS if candidate.get(field, "").strip())
    penalties = 0
    for field in REQUIRED_FIELDS:
        value = candidate.get(field, "")
        if _contains_suspicious_content(value):
            penalties += 2
        if field != "title" and _looks_garbled(value):
            penalties += 1
    return score - penalties


def parse_problem_draft(text: str) -> Dict[str, Any]:
    text = _sanitize_output(text)

    candidates: List[Dict[str, Any]] = []
    buffers = _empty_buffers()
    current_key = None

    for line in text.split("\n"):
        matched = _match_field_label(line)
        if matched:
            current_key, value = matched
            if current_key == "title" and _buffers_have_content(buffers):
                candidates.append(_build_candidate_from_buffers(buffers))
                buffers = _empty_buffers()
            if value:
                buffers[current_key].append(value)
            continue

        if current_key is not None:
            buffers[current_key].append(line)

    if _buffers_have_content(buffers):
        candidates.append(_build_candidate_from_buffers(buffers))

    if not candidates:
        return {
            "title": "",
            "description": "",
            "input_format": "",
            "output_format": "",
            "sample_input": "",
            "sample_output": "",
            "constraints": "",
            "sample_explanation": "",
        }

    best_index, best_candidate = max(
        enumerate(candidates),
        key=lambda item: (_candidate_score(item[1]), item[0]),
    )
    _ = best_index
    return best_candidate


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

    title = obj.get("title", "").strip()
    if "\n" in title:
        errors.append("title_multiline")

    for field in REQUIRED_FIELDS:
        value = obj.get(field, "")
        if isinstance(value, str) and value.strip():
            if _contains_suspicious_content(value):
                errors.append(f"suspicious_content: {field}")
            if field != "title" and _looks_garbled(value):
                errors.append(f"garbled_content: {field}")
            if field in {"sample_input", "sample_output"} and _contains_suspicious_sample_content(value):
                errors.append(f"suspicious_sample_content: {field}")

    input_format = obj.get("input_format", "").strip()
    output_format = obj.get("output_format", "").strip()
    sample_input = obj.get("sample_input", "").strip()
    sample_output = obj.get("sample_output", "").strip()
    if sample_input and _input_format_expects_multiple_lines(input_format):
        if len(sample_input.split("\n")) < 2:
            errors.append("sample_input_line_count_mismatch")
    if sample_output and _output_format_expects_single_line(output_format):
        if len(sample_output.split("\n")) > 1:
            errors.append("sample_output_line_count_mismatch")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def _repair_problem_draft(
    topic: str,
    problem_background: str,
    retrieval_summary: str,
    raw_output: str,
) -> str:
    repair_prompt = f"""请基于主题、背景和检索摘要，把下面的损坏输出修复或重写为严格符合格式的题面草稿。

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
5. 不要输出 user、assistant、用户、助手、答案 等角色标记
6. 各字段内容必须自洽，样例输入输出必须与输入输出格式一致
7. 不要重复字段内容，不要保留脏文本，不要把修复说明写进字段里
8. 样例输入和样例输出只允许保留纯样例内容，不要添加引号，不要添加解释，不要多出单独的引号行
9. 如果当前信息不足以可靠生成题面，请输出：
题目名称：资料不足
题目描述：资料不足，无法可靠生成题目。
输入格式：
输出格式：
样例输入：
样例输出：
数据范围：

主题：
{topic}

背景：
{problem_background}

检索摘要：
{retrieval_summary}

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


def _build_problem_retry_prompt(
    topic: str,
    problem_background: str,
    retrieval_summary: str,
    feedback_issues: List[str],
    previous_output: str,
) -> str:
    base_prompt = build_problem_prompt(
        topic=topic,
        problem_background=problem_background,
        retrieval_summary=retrieval_summary,
    )
    issues_text = "\n".join(f"- {issue}" for issue in feedback_issues) if feedback_issues else "- previous_output_invalid"
    previous_output = previous_output.strip() or "(empty)"
    return (
        f"{base_prompt}\n\n"
        "上一次题面草稿存在问题，请只根据下面错误定向修复，不要重复错误。\n"
        "错误列表：\n"
        f"{issues_text}\n\n"
        "上一次输出：\n"
        f"{previous_output}\n"
    )


def generate_problem_output(
    topic: str,
    problem_background: str,
    retrieval_summary: str = "",
    feedback_issues: List[str] | None = None,
    previous_output: str = "",
) -> str:
    if feedback_issues:
        prompt = _build_problem_retry_prompt(
            topic=topic,
            problem_background=problem_background,
            retrieval_summary=retrieval_summary,
            feedback_issues=feedback_issues,
            previous_output=previous_output,
        )
        temperature = 0.0
    else:
        prompt = build_problem_prompt(
            topic=topic,
            problem_background=problem_background,
            retrieval_summary=retrieval_summary,
        )
        temperature = 0.1

    llm = SiliconFlowLLM()
    return llm.chat(
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
        temperature=temperature,
        max_tokens=700,
    )


def validate_problem_statement(problem_statement: Dict[str, Any]) -> Dict[str, Any]:
    return _validate_problem_statement(problem_statement)


def run_problem_agent(
    topic: str,
    problem_background: str,
    retrieval_summary: str = "",
    feedback_issues: List[str] | None = None,
    previous_output: str = "",
) -> Dict[str, Any]:
    raw_output = generate_problem_output(
        topic=topic,
        problem_background=problem_background,
        retrieval_summary=retrieval_summary,
        feedback_issues=feedback_issues,
        previous_output=previous_output,
    )

    parsed = parse_problem_draft(raw_output)
    validation = _validate_problem_statement(parsed)

    if not validation["valid"]:
        repaired_output = _repair_problem_draft(
            topic=topic,
            problem_background=problem_background,
            retrieval_summary=retrieval_summary,
            raw_output=raw_output,
        )
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
