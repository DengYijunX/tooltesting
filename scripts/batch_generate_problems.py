import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from app.config import ENABLED_SUBJECTS
from app.workflows.problem_generation_workflow import run_problem_generation_workflow


DEFAULT_TOPIC_POOLS: Dict[str, List[str]] = {
    "physics": [
        "牛顿第二定律",
        "动能定理",
        "功和功率",
        "加速度",
    ],
    "psychology": [
        "遗忘曲线",
        "经典条件作用",
        "操作性条件作用",
        "观察法",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量测试题目生成工作流")
    parser.add_argument(
        "--subject",
        choices=ENABLED_SUBJECTS,
        required=True,
        help="学科范围",
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        help="自定义主题，可重复传入多次",
    )
    parser.add_argument(
        "--use-default-topics",
        action="store_true",
        help="使用脚本内置的测试主题池",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="限制运行主题数量，0 表示不限制",
    )
    parser.add_argument(
        "--output",
        default="",
        help="汇总结果输出路径，默认写入 audit_logs/batch_*.json",
    )
    parser.add_argument(
        "--algorithm",
        default="auto",
        help="题目算法类型，例如 auto、formula_calculation、sorting、hash_count、prefix_sum",
    )
    return parser.parse_args()


def build_topic_list(subject: str, custom_topics: List[str], use_default_topics: bool, limit: int) -> List[str]:
    topics: List[str] = []

    if use_default_topics:
        topics.extend(DEFAULT_TOPIC_POOLS.get(subject, []))

    for topic in custom_topics:
        topic = topic.strip()
        if topic and topic not in topics:
            topics.append(topic)

    if limit > 0:
        topics = topics[:limit]

    return topics


def summarize_result(topic: str, result: Dict[str, object]) -> Dict[str, object]:
    final_result = result.get("final_result", {})
    error = final_result.get("error") if isinstance(final_result, dict) else None
    return {
        "topic": topic,
        "success": not bool(error),
        "error": error or "",
        "audit_file": result.get("audit_file", final_result.get("audit_file", "") if isinstance(final_result, dict) else ""),
        "problem_title": result.get("problem_statement", {}).get("title", ""),
        "consistency_passed": bool(result.get("consistency_passed", False)),
        "sandbox_passed": bool(result.get("sandbox_result", {}).get("passed", False)),
    }


def build_output_path(subject: str, output: str) -> str:
    if output:
        return output
    os.makedirs("audit_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("audit_logs", f"batch_{subject}_{timestamp}.json")


def main() -> int:
    args = parse_args()
    topics = build_topic_list(
        subject=args.subject,
        custom_topics=args.topic,
        use_default_topics=args.use_default_topics,
        limit=args.limit,
    )

    if not topics:
        print("未提供任何主题。请使用 --topic 或 --use-default-topics。")
        return 1

    summaries: List[Dict[str, object]] = []
    success_count = 0

    print(f"开始批量测试：subject={args.subject}, topics={len(topics)}")

    for index, topic in enumerate(topics, start=1):
        print(f"\n[{index}/{len(topics)}] 生成主题：{topic}")
        result = run_problem_generation_workflow(
            topic=topic,
            subject=args.subject,
            requested_algorithm=args.algorithm,
        )
        summary = summarize_result(topic, result)
        summaries.append(summary)
        if summary["success"]:
            success_count += 1
            print(
                f"  成功 | 标题={summary['problem_title']} | "
                f"consistency={summary['consistency_passed']} | sandbox={summary['sandbox_passed']}"
            )
        else:
            print(
                f"  失败 | error={summary['error']} | audit={summary['audit_file']}"
            )

    report = {
        "subject": args.subject,
        "algorithm": args.algorithm,
        "topics": topics,
        "total": len(topics),
        "success_count": success_count,
        "failure_count": len(topics) - success_count,
        "results": summaries,
    }

    output_path = build_output_path(args.subject, args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== 批量测试汇总 =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n汇总文件：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
