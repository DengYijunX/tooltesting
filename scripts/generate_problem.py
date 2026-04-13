import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from app.problem_generator import generate_problem


def main():
    topic = input("请输入主题：").strip()
    if not topic:
        print("主题不能为空。")
        return

    result = generate_problem(topic)
    print("\n===== 生成结果 =====\n")
    print(result)


if __name__ == "__main__":
    main()