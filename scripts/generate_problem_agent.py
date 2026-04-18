import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from app.config import SUPPORTED_SUBJECTS, ENABLED_SUBJECTS
from app.workflows.problem_generation_workflow import run_problem_generation_workflow


def choose_subject():
    print("请选择知识库范围：")
    print("1. psychology")
    print("2. physics")
    print("3. math")
    print("4. finance")
    print("5. mixed")
    print("6. all")

    choice = input("请输入选项编号：").strip()

    mapping = {
        "1": "psychology",
        "2": "physics",
        "3": "math",
        "4": "finance",
        "5": "mixed",
        "6": "all",
    }
    subject = mapping.get(choice, "all")

    if subject not in SUPPORTED_SUBJECTS:
        subject = "all"

    return subject


def print_failure_result(result: dict):
    final_result = result.get("final_result", {})
    error_type = final_result.get("error", "unknown_error")

    print("\n===== 生成失败 =====\n")
    print("错误类型：", error_type)

    if error_type == "subject_not_enabled":
        print("提示信息：", final_result.get("message", ""))

    elif error_type == "problem_statement_invalid":
        print("详细信息：", final_result.get("details", []))
        print("\n===== 原始题面输出 =====\n")
        print(final_result.get("raw_problem_output", ""))

    elif error_type == "knowledge_insufficient":
        print("详细信息：", final_result.get("details", []))
        print("\n===== 知识充分性统计 =====\n")
        print(json.dumps(final_result.get("knowledge_stats", {}), ensure_ascii=False, indent=2))

    elif error_type == "solution_invalid":
        print("详细信息：", final_result.get("details", []))
        print("\n===== 原始参考答案输出 =====\n")
        print(final_result.get("raw_solution_output", ""))

        if final_result.get("reference_code"):
            print("\n===== 解析出的参考代码 =====\n")
            print(final_result.get("reference_code"))

    elif error_type in {"consistency_invalid", "testcase_invalid", "sandbox_invalid", "final_review_invalid"}:
        print("详细信息：")
        print(json.dumps(final_result.get("details", []), ensure_ascii=False, indent=2))
        if final_result.get("suggestions"):
            print("\n===== 修复建议 =====\n")
            print(json.dumps(final_result.get("suggestions", []), ensure_ascii=False, indent=2))

    else:
        print("返回内容：")
        print(json.dumps(final_result, ensure_ascii=False, indent=2))

    print("\n===== 审计文件 =====")
    print(result.get("audit_file", final_result.get("audit_file", "")))


def print_success_result(result: dict):
    print("\n===== 题目背景 =====\n")
    print(result.get("problem_background", ""))

    print("\n===== 标准题面 =====\n")
    print(json.dumps(result.get("problem_statement", {}), ensure_ascii=False, indent=2))

    print("\n===== 参考答案说明 =====\n")
    print(result.get("solution_explanation", ""))

    print("\n===== 参考代码 =====\n")
    print(result.get("reference_code", ""))

    print("\n===== 一致性检查 =====\n")
    print(json.dumps({
        "consistency_passed": result.get("consistency_passed"),
        "consistency_issues": result.get("consistency_issues", []),
    }, ensure_ascii=False, indent=2))

    print("\n===== 测试用例 =====\n")
    print(json.dumps(result.get("test_cases", []), ensure_ascii=False, indent=2))

    print("\n===== 沙箱结果 =====\n")
    print(json.dumps(result.get("sandbox_result", {}), ensure_ascii=False, indent=2))

    print("\n===== 最终审核 =====\n")
    print(json.dumps(result.get("final_review", {}), ensure_ascii=False, indent=2))

    print("\n===== 审计文件 =====")
    print(result.get("audit_file", ""))


def main():
    subject = choose_subject()
    topic = input("请输入主题：").strip()

    if not topic:
        print("主题不能为空。")
        return

    if subject not in ENABLED_SUBJECTS:
        print(f"\n当前学科 {subject} 尚未接入知识库。")
        print(f"当前已启用学科：{', '.join(ENABLED_SUBJECTS)}")
        return

    result = run_problem_generation_workflow(topic=topic, subject=subject)

    final_result = result.get("final_result", {})
    if isinstance(final_result, dict) and final_result.get("error"):
        print_failure_result(result)
        return

    print_success_result(result)


if __name__ == "__main__":
    main()
