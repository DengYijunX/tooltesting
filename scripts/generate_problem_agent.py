import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from app.config import SUPPORTED_SUBJECTS, ENABLED_SUBJECTS
from app.workflows.problem_workflow import run_problem_workflow


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

    result = run_problem_workflow(topic=topic, subject=subject)

    print("\n===== 最终结果 =====\n")
    print(result["final_problem"])

    print("\n===== 审计文件 =====")
    print(result["audit_file"])


if __name__ == "__main__":
    main()