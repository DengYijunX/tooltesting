from typing import Any, Dict, List


def build_final_review(state: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    suggestions: List[str] = []

    if not state.get("knowledge_sufficiency", False):
        issues.append("knowledge_sufficiency_failed")
        suggestions.append("补充检索资料或降低题目生成难度。")

    if not state.get("problem_statement_valid", False):
        issues.append("problem_statement_invalid")
        suggestions.append("加强题面生成约束，确保样例与字段完整。")

    if not state.get("solution_valid", False):
        issues.append("solution_invalid")
        suggestions.append("加强答案 JSON 输出约束，并增加重新生成兜底。")

    if not state.get("consistency_passed", False):
        issues.append("consistency_check_failed")
        suggestions.append("检查题面字段、样例与代码读写方式是否一致。")

    if not state.get("testcase_generation_passed", False):
        issues.append("testcase_generation_failed")
        suggestions.append("至少保证样例输入输出可以转成测试用例。")

    sandbox_result = state.get("sandbox_result", {})
    if not sandbox_result.get("passed", False):
        issues.append("sandbox_execution_failed")
        suggestions.append("修复参考代码或调整题面样例，确保样例可跑通。")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "suggestions": suggestions,
    }
