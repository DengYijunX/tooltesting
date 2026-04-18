import re
from typing import Any, Dict, List


SUSPICIOUS_TEXT_FRAGMENTS = [
    "user",
    "assistant",
    "请纠正",
    "不要输出",
    "根据给定背景",
]


def _normalize_text(text: str) -> str:
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")


def _contains_input_read(code: str) -> bool:
    return any(token in code for token in ["input(", "sys.stdin", "stdin.readline"])


def _contains_output_write(code: str) -> bool:
    return any(token in code for token in ["print(", "sys.stdout.write"])


def _looks_suspicious(text: str) -> bool:
    lowered = text.lower()
    if any(fragment in lowered for fragment in SUSPICIOUS_TEXT_FRAGMENTS):
        return True
    if "```" in text:
        return True
    if re.search(r"[，,。]{2,}", text):
        return True
    return False


def _estimate_code_input_calls(code: str) -> int:
    return code.count("input(") + code.count("stdin.readline(")


def _code_reads_all_input(code: str) -> bool:
    return any(token in code for token in ["sys.stdin.read(", "sys.stdin.buffer.read("])


def _expected_input_line_count(problem_statement: Dict[str, Any]) -> int:
    sample_input = _normalize_text(str(problem_statement.get("sample_input", "")))
    if sample_input:
        return len(sample_input.split("\n"))
    return 0


def _input_format_expects_multiple_lines(input_format: str) -> bool:
    return "两行" in input_format or "第二行" in input_format


def _output_format_expects_single_line(output_format: str) -> bool:
    return "一行" in output_format or "1行" in output_format


def check_problem_solution_consistency(
    problem_statement: Dict[str, Any],
    reference_code: str,
    solution_explanation: str,
) -> Dict[str, Any]:
    issues: List[str] = []

    required_problem_fields = [
        "title",
        "description",
        "input_format",
        "output_format",
        "sample_input",
        "sample_output",
        "constraints",
    ]

    for field in required_problem_fields:
        value = _normalize_text(str(problem_statement.get(field, "")))
        if not value:
            issues.append(f"problem_field_missing: {field}")
        elif _looks_suspicious(value):
            issues.append(f"problem_field_suspicious: {field}")

    normalized_code = _normalize_text(reference_code)
    if not normalized_code:
        issues.append("reference_code_empty")
    else:
        try:
            compile(normalized_code, "<reference_code>", "exec")
        except Exception:
            issues.append("reference_code_compile_failed")

        if not _contains_input_read(normalized_code):
            issues.append("reference_code_input_read_missing")
        if not _contains_output_write(normalized_code):
            issues.append("reference_code_output_write_missing")

        input_format = _normalize_text(str(problem_statement.get("input_format", "")))
        expected_input_lines = _expected_input_line_count(problem_statement)
        input_call_count = _estimate_code_input_calls(normalized_code)
        reads_all_input = _code_reads_all_input(normalized_code)

        if (
            (_input_format_expects_multiple_lines(input_format) or expected_input_lines >= 2)
            and not reads_all_input
            and input_call_count < 2
        ):
            issues.append("reference_code_input_line_count_mismatch")

    if not _normalize_text(solution_explanation):
        issues.append("solution_explanation_empty")

    sample_input = _normalize_text(str(problem_statement.get("sample_input", "")))
    sample_output = _normalize_text(str(problem_statement.get("sample_output", "")))
    if sample_input and not sample_output:
        issues.append("sample_output_missing_for_sample_input")
    if sample_output and not sample_input:
        issues.append("sample_input_missing_for_sample_output")
    if sample_input:
        sample_input_lines = len(sample_input.split("\n"))
        input_format = _normalize_text(str(problem_statement.get("input_format", "")))
        if _input_format_expects_multiple_lines(input_format) and sample_input_lines < 2:
            issues.append("sample_input_line_count_mismatch")
    if sample_output:
        sample_output_lines = len(sample_output.split("\n"))
        output_format = _normalize_text(str(problem_statement.get("output_format", "")))
        if _output_format_expects_single_line(output_format) and sample_output_lines > 1:
            issues.append("sample_output_line_count_mismatch")

    return {
        "consistency_passed": len(issues) == 0,
        "consistency_issues": issues,
    }
