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
    "资料不足",
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
        if stripped in {"'", '"', "'''", '"""', "一行", "两行", "三行"}:
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


def _has_explicit_computation_rule(obj: Dict[str, Any]) -> bool:
    text = "\n".join(
        str(obj.get(field, ""))
        for field in ("description", "input_format", "output_format", "constraints")
    )
    if not text.strip():
        return False

    rule_markers = [
        "=",
        "等于",
        "增加",
        "减少",
        "更新",
        "累加",
        "平均",
        "总和",
        "差值",
        "乘以",
        "除以",
        "公式",
        "统计",
        "排序",
        "区间",
        "最大",
        "最小",
        "次数",
        "前缀和",
    ]
    return any(marker in text for marker in rule_markers)


def _derive_constraints_from_input_format(input_format: str) -> str:
    text = _clean_text(input_format)
    if not text:
        return ""

    constraint_lines: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if (
            "≤" in stripped
            or "<=" in stripped
            or "范围" in stripped
            or "正整数" in stripped
            or "正实数" in stripped
            or "非负" in stripped
        ):
            constraint_lines.append(stripped)

    if constraint_lines:
        return "；".join(constraint_lines)

    return "输入数据满足题目输入格式要求。"


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
    if not parsed.get("constraints", "").strip():
        parsed["constraints"] = _derive_constraints_from_input_format(
            parsed.get("input_format", "")
        )
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

    if not _has_explicit_computation_rule(obj):
        errors.append("computation_rule_missing")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def _validate_task_model_alignment(
    obj: Dict[str, Any],
    task_model: Dict[str, Any] | None,
) -> List[str]:
    if not task_model or not isinstance(obj, dict):
        return []

    errors: List[str] = []
    algorithm_model = str(task_model.get("algorithm_model", "")).strip()
    knowledge_rule = str(task_model.get("knowledge_rule", "")).strip()
    output_variable = str(task_model.get("output_variable", "")).strip()
    description = str(obj.get("description", ""))
    input_format = str(obj.get("input_format", ""))
    sample_input = str(obj.get("sample_input", "")).strip()
    sample_output = str(obj.get("sample_output", "")).strip()

    if knowledge_rule:
        rule_lhs = knowledge_rule.split("=", 1)[0].strip()
        has_rule_text = knowledge_rule in description
        has_rule_lhs = bool(rule_lhs and rule_lhs in description)
        has_output_var = bool(output_variable and output_variable in description)
        if not (has_rule_text or has_rule_lhs or has_output_var):
            errors.append("task_model_formula_rule_mismatch")

    if algorithm_model == "sorting":
        if "第一行" not in input_format or "n" not in input_format:
            errors.append("sorting_input_format_missing_n")

        sample_lines = [line.strip() for line in sample_input.split("\n") if line.strip()]
        n = None
        if sample_lines:
            first_parts = sample_lines[0].split()
            if len(first_parts) == 1 and first_parts[0].isdigit():
                n = int(first_parts[0])
            else:
                errors.append("sorting_sample_missing_n")
        else:
            errors.append("sorting_sample_empty")

        if n is not None:
            records = sample_lines[1:]
            if len(records) != n:
                errors.append("sorting_sample_record_count_mismatch")

            record_ids = set()
            for index, row in enumerate(records, start=1):
                parts = row.split()
                if parts and re.fullmatch(r"-?\d+", parts[0]):
                    record_ids.add(int(parts[0]))
                else:
                    record_ids.add(index)

            output_ids = []
            for token in sample_output.split():
                if re.fullmatch(r"-?\d+", token):
                    output_ids.append(int(token))

            if output_ids and any(output_id not in record_ids for output_id in output_ids):
                errors.append("sorting_sample_output_unknown_id")

    return errors


def _validate_problem_statement_for_task_model(
    obj: Dict[str, Any],
    task_model: Dict[str, Any] | None,
) -> Dict[str, Any]:
    validation = _validate_problem_statement(obj)
    if validation["valid"]:
        alignment_errors = _validate_task_model_alignment(obj, task_model)
        if alignment_errors:
            validation = {
                "valid": False,
                "errors": alignment_errors,
            }
    return validation


def _repair_problem_draft(
    topic: str,
    problem_background: str,
    retrieval_summary: str,
    raw_output: str,
    task_model: Dict[str, Any] | None = None,
) -> str:
    task_model_text = str(task_model or {})
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
9. 数据范围不能为空；如果输入格式中已经写了取值范围，请把范围整理到数据范围字段
10. 题目必须包含明确计算规则或公式，说明如何从输入得到输出；不能只写“模拟”“计算最终结果”
11. 必须遵守任务模型；如果任务模型指定 algorithm_model，题面必须体现该算法类型
12. 如果任务模型包含 knowledge_rule，题面必须直接使用该规则，不要自行发明公式
13. 如果当前信息不足以可靠生成题面，请输出：
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

任务模型：
{task_model_text}

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


def _format_problem_draft(problem_statement: Dict[str, Any]) -> str:
    return "\n".join([
        f"题目名称：{problem_statement.get('title', '')}",
        f"题目描述：{problem_statement.get('description', '')}",
        f"输入格式：{problem_statement.get('input_format', '')}",
        f"输出格式：{problem_statement.get('output_format', '')}",
        f"样例输入：\n{problem_statement.get('sample_input', '')}",
        f"样例输出：\n{problem_statement.get('sample_output', '')}",
        f"数据范围：{problem_statement.get('constraints', '')}",
    ])


def _build_task_model_problem_fallback(task_model: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not task_model:
        return None

    topic = str(task_model.get("topic", "知识点")).strip() or "知识点"
    algorithm_model = str(task_model.get("algorithm_model", "simulation")).strip()
    task_rule = str(task_model.get("task_rule", "")).strip()
    knowledge_rule = str(task_model.get("knowledge_rule", "")).strip()
    input_variables = list(task_model.get("input_variables", []) or [])
    output_variable = str(task_model.get("output_variable", "answer")).strip() or "answer"
    sample_rows = [str(row) for row in task_model.get("sample_rows", []) or []]
    sample_outputs = [str(row) for row in task_model.get("sample_outputs", []) or []]
    rule_text = task_rule or f"按照题目规则计算 {output_variable}。"

    if knowledge_rule and algorithm_model == "formula_calculation":
        rows = sample_rows[:3] or ["1 1", "2 2", "3 3"]
        outputs = sample_outputs[: len(rows)] or ["1.00"] * len(rows)
        variables = "、".join(input_variables) if input_variables else "一组参数"
        return {
            "title": f"{topic}批量计算",
            "description": (
                f"给定 n 组与 {topic} 相关的数据。"
                f"{rule_text} 请输出每组数据的计算结果。"
            ),
            "input_format": (
                "第一行输入一个整数 n，表示数据组数。\n"
                f"接下来 n 行，每行输入 {variables}。"
            ),
            "output_format": "输出 n 行，每行一个计算结果，结果保留两位小数。",
            "sample_input": f"{len(rows)}\n" + "\n".join(rows),
            "sample_output": "\n".join(outputs),
            "constraints": "1 <= n <= 1000，输入数值范围保证计算结果可用双精度浮点数表示。",
            "sample_explanation": "",
        }

    if algorithm_model == "sorting":
        if knowledge_rule and sample_rows and sample_outputs:
            indexed_rows = [f"{index + 1} {row}" for index, row in enumerate(sample_rows[:3])]
            sortable = []
            for index, value in enumerate(sample_outputs[: len(indexed_rows)], start=1):
                try:
                    numeric = float(value)
                except ValueError:
                    numeric = 0.0
                sortable.append((index, numeric))
            sorted_ids = [str(index) for index, _ in sorted(sortable, key=lambda item: (-item[1], item[0]))]
            sample_input = f"{len(indexed_rows)}\n" + "\n".join(indexed_rows)
            sample_output = " ".join(sorted_ids)
            variables = "、".join(input_variables) if input_variables else "若干参数"
            input_format = (
                "第一行输入一个整数 n，表示记录数量。\n"
                f"接下来 n 行，每行输入一个编号 id 以及 {variables}。"
            )
        else:
            sample_input = "3\n1 80\n2 95\n3 95"
            sample_output = "2 3 1"
            input_format = "第一行输入一个整数 n，表示记录数量。\n接下来 n 行，每行输入编号 id 和指标值 score。"
        return {
            "title": f"{topic}指标排序",
            "description": f"{rule_text} 请输出排序后的编号序列。",
            "input_format": input_format,
            "output_format": "一行输出排序后的编号，编号之间用一个空格分隔。",
            "sample_input": sample_input,
            "sample_output": sample_output,
            "constraints": "1 <= n <= 1000，编号为正整数，输入指标范围保证可比较。",
            "sample_explanation": "",
        }

    if algorithm_model in {"counting", "hash_count"}:
        return {
            "title": f"{topic}记录统计",
            "description": f"给定 n 条与 {topic} 相关的类别记录。{rule_text} 如果次数相同，输出字典序最小的类别。",
            "input_format": "第一行输入一个整数 n，表示记录数量。\n接下来 n 行，每行输入一个类别名称 s。",
            "output_format": "一行输出出现次数最多的类别名称和出现次数，中间用一个空格分隔。",
            "sample_input": "5\nA\nB\nA\nC\nA",
            "sample_output": "A 3",
            "constraints": "1 <= n <= 100000，类别名称只包含大小写英文字母和数字，长度不超过 30。",
            "sample_explanation": "",
        }

    if algorithm_model == "prefix_sum":
        if knowledge_rule and sample_rows and sample_outputs:
            rows = sample_rows[:3]
            values = []
            for value in sample_outputs[: len(rows)]:
                try:
                    values.append(float(value))
                except ValueError:
                    values.append(0.0)
            query_outputs = [sum(values[0:2]), sum(values[1:3])]
            return {
                "title": f"{topic}区间指标查询",
                "description": f"给定 n 条记录。{rule_text} 对计算出的指标建立前缀和，回答 q 个区间总和查询。",
                "input_format": (
                    "第一行输入两个整数 n 和 q，表示记录数量和查询数量。\n"
                    "接下来 n 行，每行输入一条记录的参数。\n"
                    "接下来 q 行，每行输入两个整数 l 和 r，表示查询区间。"
                ),
                "output_format": "输出 q 行，每行一个区间指标总和，结果保留两位小数。",
                "sample_input": f"{len(rows)} 2\n" + "\n".join(rows) + "\n1 2\n2 3",
                "sample_output": "\n".join(f"{value:.2f}" for value in query_outputs),
                "constraints": "1 <= n, q <= 100000，1 <= l <= r <= n，输入数值范围保证计算结果可用双精度浮点数表示。",
                "sample_explanation": "",
            }
        return {
            "title": f"{topic}区间总和",
            "description": f"给定一个与 {topic} 相关的数值序列，使用前缀和回答多个区间总和查询。",
            "input_format": "第一行输入两个整数 n 和 q。\n第二行输入 n 个整数 a_i。\n接下来 q 行，每行输入 l 和 r。",
            "output_format": "输出 q 行，每行一个区间和。",
            "sample_input": "5 2\n1 2 3 4 5\n1 3\n2 5",
            "sample_output": "6\n14",
            "constraints": "1 <= n, q <= 100000，1 <= a_i <= 1000，1 <= l <= r <= n。",
            "sample_explanation": "",
        }

    if algorithm_model == "binary_search":
        return {
            "title": f"{topic}阈值查找",
            "description": f"给定一个非递减指标序列和目标值 k。{rule_text} 请用二分查找思想找到第一个大于等于 k 的位置；如果不存在，输出 -1。",
            "input_format": "第一行输入两个整数 n 和 k。\n第二行输入 n 个非递减整数 a_i。",
            "output_format": "一行输出第一个满足 a_i >= k 的位置编号，位置从 1 开始；如果不存在输出 -1。",
            "sample_input": "5 7\n1 3 7 7 10",
            "sample_output": "3",
            "constraints": "1 <= n <= 100000，0 <= k, a_i <= 1000000000，序列 a_i 非递减。",
            "sample_explanation": "",
        }

    if algorithm_model == "sliding_window":
        return {
            "title": f"{topic}连续区间分析",
            "description": f"给定一个非负指标序列和上限 k。{rule_text} 使用滑动窗口求总和不超过 k 的最长连续区间长度。",
            "input_format": "第一行输入两个整数 n 和 k。\n第二行输入 n 个非负整数 a_i。",
            "output_format": "一行输出满足区间和不超过 k 的最长连续区间长度。",
            "sample_input": "5 7\n2 1 3 4 1",
            "sample_output": "3",
            "constraints": "1 <= n <= 100000，0 <= a_i <= 1000000，0 <= k <= 1000000000。",
            "sample_explanation": "",
        }

    if algorithm_model == "greedy":
        return {
            "title": f"{topic}活动选择",
            "description": f"给定若干与 {topic} 相关的活动时间段。{rule_text} 请选择最多数量的互不重叠活动并输出数量。",
            "input_format": "第一行输入一个整数 n，表示活动数量。\n接下来 n 行，每行输入两个整数 l 和 r，表示活动开始和结束时间。",
            "output_format": "一行输出最多能选择的活动数量。",
            "sample_input": "4\n1 3\n2 4\n3 5\n6 7",
            "sample_output": "3",
            "constraints": "1 <= n <= 100000，0 <= l < r <= 1000000000。",
            "sample_explanation": "",
        }

    if algorithm_model == "dynamic_programming":
        return {
            "title": f"{topic}资源分配",
            "description": f"给定若干与 {topic} 相关的任务，每个任务有消耗和收益。{rule_text} 在总资源不超过 W 的条件下，使用动态规划求最大收益。",
            "input_format": "第一行输入两个整数 n 和 W。\n接下来 n 行，每行输入两个整数 cost 和 value，表示任务消耗和收益。",
            "output_format": "一行输出可获得的最大总收益。",
            "sample_input": "3 5\n2 3\n3 4\n4 5",
            "sample_output": "7",
            "constraints": "1 <= n <= 200，1 <= W <= 10000，1 <= cost <= W，1 <= value <= 100000。",
            "sample_explanation": "",
        }

    return {
        "title": f"{topic}过程模拟",
        "description": f"给定一个初始值和 n 次操作。{rule_text} 请输出所有操作后的最终结果。",
        "input_format": "第一行输入两个整数 n 和 x，表示操作次数和初始值。\n第二行输入 n 个整数 delta_i，表示每次操作对 x 的增量。",
        "output_format": "一行输出最终的 x。",
        "sample_input": "3 10\n2 -5 4",
        "sample_output": "11",
        "constraints": "1 <= n <= 100000，-1000 <= delta_i <= 1000，结果在 32 位有符号整数范围内。",
        "sample_explanation": "",
    }


def _build_rule_based_problem_fallback(topic: str) -> Dict[str, Any] | None:
    if "遗忘曲线" in topic:
        return {
            "title": "遗忘曲线保留率计算",
            "description": (
                "给定若干份学习材料的初始记忆保持率 p、每日遗忘率 r 和经过天数 d。"
                "按照简化遗忘曲线公式 p = p * (1 - r)^d 计算经过 d 天后的记忆保持率。"
                "请输出每份材料最终的记忆保持率。"
            ),
            "input_format": (
                "第一行输入一个整数 n，表示学习材料数量。\n"
                "接下来 n 行，每行输入一个实数 p、一个实数 r 和一个整数 d，分别表示初始保持率、每日遗忘率和经过天数。"
            ),
            "output_format": "输出 n 行，每行一个最终记忆保持率，结果保留两位小数。",
            "sample_input": "3\n100 0.10 2\n80 0.05 3\n60 0.20 1",
            "sample_output": "81.00\n68.59\n48.00",
            "constraints": "1 <= n <= 1000，0 <= p <= 100，0 <= r <= 1，0 <= d <= 365。",
            "sample_explanation": "",
        }

    if "条件作用" not in topic:
        return None

    return {
        "title": f"{topic}概率更新",
        "description": (
            f"给定 {topic} 学习过程中的初始反应概率 p、学习率 r 和 n 次刺激结果。"
            "每次结果 x_i 为 1 表示有效配对或强化出现，此时更新 p = p + r * (1 - p)；"
            "x_i 为 0 表示未出现有效配对或强化，此时更新 p = p * (1 - r)。"
            "请计算 n 次更新后的最终反应概率。"
        ),
        "input_format": (
            "第一行输入一个整数 n 和两个实数 p、r，分别表示刺激次数、初始反应概率和学习率。\n"
            "第二行输入 n 个整数 x_i，每个 x_i 只能为 0 或 1。"
        ),
        "output_format": "一行输出最终反应概率，结果保留两位小数。",
        "sample_input": "3 0.20 0.50\n1 0 1",
        "sample_output": "0.65",
        "constraints": "1 <= n <= 1000，0 <= p <= 1，0 <= r <= 1，x_i 为 0 或 1。",
        "sample_explanation": "",
    }


def _build_problem_retry_prompt(
    topic: str,
    problem_background: str,
    retrieval_summary: str,
    feedback_issues: List[str],
    previous_output: str,
    task_model: Dict[str, Any] | None = None,
) -> str:
    base_prompt = build_problem_prompt(
        topic=topic,
        problem_background=problem_background,
        retrieval_summary=retrieval_summary,
        task_model=task_model,
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
    task_model: Dict[str, Any] | None = None,
) -> str:
    if feedback_issues:
        prompt = _build_problem_retry_prompt(
            topic=topic,
            problem_background=problem_background,
            retrieval_summary=retrieval_summary,
            feedback_issues=feedback_issues,
            previous_output=previous_output,
            task_model=task_model,
        )
        temperature = 0.0
    else:
        prompt = build_problem_prompt(
            topic=topic,
            problem_background=problem_background,
            retrieval_summary=retrieval_summary,
            task_model=task_model,
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
    task_model: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    raw_output = generate_problem_output(
        topic=topic,
        problem_background=problem_background,
        retrieval_summary=retrieval_summary,
        feedback_issues=feedback_issues,
        previous_output=previous_output,
        task_model=task_model,
    )

    parsed = parse_problem_draft(raw_output)
    validation = _validate_problem_statement_for_task_model(parsed, task_model)

    if not validation["valid"]:
        repaired_output = _repair_problem_draft(
            topic=topic,
            problem_background=problem_background,
            retrieval_summary=retrieval_summary,
            raw_output=raw_output,
            task_model=task_model,
        )
        repaired_parsed = parse_problem_draft(repaired_output)
        repaired_validation = _validate_problem_statement_for_task_model(
            repaired_parsed,
            task_model,
        )

        if repaired_validation["valid"]:
            parsed = repaired_parsed
            validation = repaired_validation
            raw_output = repaired_output

    if not validation["valid"] and task_model:
        fallback_problem = _build_task_model_problem_fallback(task_model)
        if fallback_problem is not None:
            fallback_validation = _validate_problem_statement_for_task_model(
                fallback_problem,
                task_model,
            )
            if fallback_validation["valid"]:
                parsed = fallback_problem
                validation = fallback_validation
                raw_output = _format_problem_draft(fallback_problem)

    should_use_rule_fallback = (
        not task_model
        and (
        "computation_rule_missing" in validation["errors"]
        or any(error.startswith("suspicious_content:") for error in validation["errors"])
        )
    )
    if not validation["valid"] and should_use_rule_fallback:
        fallback_problem = _build_rule_based_problem_fallback(topic)
        if fallback_problem is not None:
            fallback_validation = _validate_problem_statement(fallback_problem)
            if fallback_validation["valid"]:
                parsed = fallback_problem
                validation = fallback_validation
                raw_output = _format_problem_draft(fallback_problem)

    return {
        "problem_statement": parsed if validation["valid"] else {},
        "raw_problem_output": raw_output,
        "problem_statement_valid": validation["valid"],
        "problem_statement_errors": validation["errors"],
    }
