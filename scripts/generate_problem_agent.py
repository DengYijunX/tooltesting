import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from app.workflows.problem_workflow import run_problem_workflow


def main():
    topic = input("请输入主题：").strip()
    if not topic:
        print("主题不能为空。")
        return

    result = run_problem_workflow(topic)

    print("\n===== 最终结果 =====\n")
    print(result["final_problem"])

    print("\n===== 审计文件 =====")
    print(result["audit_file"])


if __name__ == "__main__":
    main()