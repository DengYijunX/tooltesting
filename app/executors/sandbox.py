import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List


def _normalize_output(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def run_code_in_sandbox(
    reference_code: str,
    test_cases: List[Dict[str, Any]],
    timeout_seconds: int = 5,
) -> Dict[str, Any]:
    if not reference_code.strip():
        return {
            "passed": False,
            "total_cases": len(test_cases),
            "passed_cases": 0,
            "details": [{"error": "reference_code_empty"}],
        }

    os.makedirs("storage/sandbox_runs", exist_ok=True)
    details: List[Dict[str, Any]] = []
    passed_cases = 0

    with tempfile.TemporaryDirectory(dir="storage/sandbox_runs") as temp_dir:
        script_path = os.path.join(temp_dir, "solution.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(reference_code)

        for case in test_cases:
            case_input = str(case.get("input", ""))
            expected_output = _normalize_output(str(case.get("expected_output", "")))

            try:
                completed = subprocess.run(
                    [sys.executable, script_path],
                    input=case_input,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                actual_output = _normalize_output(completed.stdout)
                stderr_output = _normalize_output(completed.stderr)

                passed = (
                    completed.returncode == 0
                    and actual_output == expected_output
                )
                if passed:
                    passed_cases += 1

                details.append({
                    "case_id": case.get("case_id"),
                    "case_type": case.get("case_type"),
                    "passed": passed,
                    "returncode": completed.returncode,
                    "input": case_input,
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                    "stderr": stderr_output,
                })
            except subprocess.TimeoutExpired:
                details.append({
                    "case_id": case.get("case_id"),
                    "case_type": case.get("case_type"),
                    "passed": False,
                    "error": "timeout",
                })

    return {
        "passed": passed_cases == len(test_cases) and len(test_cases) > 0,
        "total_cases": len(test_cases),
        "passed_cases": passed_cases,
        "details": details,
    }
