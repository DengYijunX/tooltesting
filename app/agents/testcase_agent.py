from typing import Any, Dict, List


def run_testcase_agent(
    problem_statement: Dict[str, Any],
    reference_code: str,
) -> Dict[str, Any]:
    test_cases: List[Dict[str, Any]] = []
    issues: List[str] = []

    sample_input = str(problem_statement.get("sample_input", "")).strip()
    sample_output = str(problem_statement.get("sample_output", "")).strip()

    if sample_input and sample_output:
        test_cases.append({
            "case_id": 1,
            "case_type": "sample",
            "input": sample_input,
            "expected_output": sample_output,
        })
    else:
        issues.append("sample_case_missing")

    if not reference_code.strip():
        issues.append("reference_code_empty")

    return {
        "test_cases": test_cases,
        "testcase_generation_passed": len(test_cases) > 0 and len(issues) == 0,
        "testcase_generation_issues": issues,
    }
