import re
from typing import Any, Dict, List


SUPPORTED_ALGORITHMS = {
    "auto",
    "formula_calculation",
    "sorting",
    "counting",
    "hash_count",
    "simulation",
    "prefix_sum",
    "greedy",
    "binary_search",
    "sliding_window",
    "dynamic_programming",
    "dp",
}

ALGORITHM_ALIASES = {
    "公式": "formula_calculation",
    "公式计算": "formula_calculation",
    "排序": "sorting",
    "计数": "counting",
    "哈希": "hash_count",
    "哈希表": "hash_count",
    "模拟": "simulation",
    "前缀和": "prefix_sum",
    "贪心": "greedy",
    "二分": "binary_search",
    "滑动窗口": "sliding_window",
    "动态规划": "dynamic_programming",
}


def _format_docs_text(retrieved_docs: List[Dict[str, Any]]) -> str:
    return "\n".join(str(doc.get("content", "")) for doc in retrieved_docs)


def _normalize_algorithm(value: str | None, notes: str = "") -> str:
    candidates = [value or "", notes or ""]
    for candidate in candidates:
        text = candidate.strip()
        if not text:
            continue
        lowered = text.lower().replace("-", "_").replace(" ", "_")
        if lowered in SUPPORTED_ALGORITHMS:
            if lowered == "dp":
                return "dynamic_programming"
            return lowered
        for alias, normalized in ALGORITHM_ALIASES.items():
            if alias in text:
                return normalized
    return "auto"


def _detect_formula_spec(topic: str, text: str) -> Dict[str, Any]:
    merged = f"{topic}\n{text}"
    compact = re.sub(r"\s+", "", merged)

    if "动能" in compact and ("mv" in compact or "Ek" in compact or "动能定理" in compact):
        return {
            "name": "动能变化量",
            "formula": "delta_E = 0.5 * m * (v2 * v2 - v1 * v1)",
            "input_variables": ["m", "v1", "v2"],
            "output_variable": "delta_E",
            "unit_hint": "结果保留两位小数",
            "sample_rows": ["2 3 5", "1 0 10", "4 6 2"],
            "sample_outputs": ["16.00", "50.00", "-64.00"],
        }

    if "牛顿第二定律" in compact or "F=ma" in compact or "F＝ma" in compact:
        return {
            "name": "牛顿第二定律",
            "formula": "a = F / m",
            "input_variables": ["m", "F"],
            "output_variable": "a",
            "unit_hint": "结果保留两位小数",
            "sample_rows": ["3 6", "5 20", "2 7"],
            "sample_outputs": ["2.00", "4.00", "3.50"],
        }

    if "功率" in compact and ("P=W/t" in compact or "P＝W/t" in compact or "功和功率" in compact):
        return {
            "name": "平均功率",
            "formula": "P = W / t",
            "input_variables": ["W", "t"],
            "output_variable": "P",
            "unit_hint": "结果保留两位小数",
            "sample_rows": ["100 5", "60 4", "81 9"],
            "sample_outputs": ["20.00", "15.00", "9.00"],
        }

    if "加速度" in compact and ("速度变化量" in compact or "v" in compact):
        return {
            "name": "加速度",
            "formula": "a = (v2 - v1) / t",
            "input_variables": ["v1", "v2", "t"],
            "output_variable": "a",
            "unit_hint": "结果保留两位小数",
            "sample_rows": ["0 10 5", "3 9 2", "8 2 3"],
            "sample_outputs": ["2.00", "3.00", "-2.00"],
        }

    if "遗忘曲线" in compact:
        return {
            "name": "记忆保持率",
            "formula": "p = p * (1 - r) ^ d",
            "input_variables": ["p", "r", "d"],
            "output_variable": "p",
            "unit_hint": "结果保留两位小数",
            "sample_rows": ["100 0.10 2", "80 0.05 3", "60 0.20 1"],
            "sample_outputs": ["81.00", "68.59", "48.00"],
        }

    return {}


def _default_algorithm(subject: str, has_formula: bool) -> str:
    if has_formula:
        return "formula_calculation"
    if subject == "psychology":
        return "hash_count"
    return "simulation"


def _task_rule_for_algorithm(
    topic: str,
    algorithm_model: str,
    formula_spec: Dict[str, Any],
) -> str:
    formula = formula_spec.get("formula", "")
    output_variable = formula_spec.get("output_variable", "score")

    if algorithm_model == "formula_calculation":
        return f"对每组输入数据按公式 {formula} 计算 {output_variable} 并输出。"
    if algorithm_model == "sorting":
        if formula:
            return (
                f"每条记录先按公式 {formula} 计算 {output_variable}，"
                f"再按 {output_variable} 从大到小排序，数值相同按编号从小到大排序，输出排序后的编号。"
            )
        return f"统计每个与 {topic} 相关对象的指标，并按指标从大到小排序，指标相同按编号从小到大排序。"
    if algorithm_model in {"counting", "hash_count"}:
        if formula:
            return (
                f"每条记录先按公式 {formula} 计算 {output_variable}，"
                f"再统计 {output_variable} 落入不同等级区间的数量并输出。"
            )
        return f"给定若干条与 {topic} 相关的记录，使用计数或哈希统计每类记录出现次数并输出最高频类别。"
    if algorithm_model == "prefix_sum":
        if formula:
            return (
                f"先对每条记录按公式 {formula} 计算 {output_variable}，"
                "再建立前缀和，回答多个区间内指标总和查询。"
            )
        return f"给定 {topic} 相关的数值序列，建立前缀和并回答多个区间总和查询。"
    if algorithm_model == "simulation":
        if formula:
            return (
                f"按时间顺序读取事件，每次按公式 {formula} 计算本次变化量，"
                "累计并输出最终结果。"
            )
        return f"按输入顺序模拟 {topic} 相关状态变化，并输出最终状态。"
    if algorithm_model == "binary_search":
        if formula:
            return (
                f"每条记录可按公式 {formula} 计算指标，"
                "要求在有序候选值中用二分思想找到满足目标条件的最小值。"
            )
        return f"围绕 {topic} 的单调条件设计判定函数，用二分查找满足条件的最小值。"
    if algorithm_model == "sliding_window":
        return f"给定 {topic} 相关序列，使用滑动窗口找到满足条件的最短或最长连续区间。"
    if algorithm_model == "greedy":
        return f"给定若干 {topic} 相关任务或事件，按局部最优规则选择，输出可达到的最大收益或最多数量。"
    if algorithm_model == "dynamic_programming":
        return f"给定 {topic} 相关阶段和选择，使用动态规划计算最优值。"
    return f"把 {topic} 转换为明确的输入、状态变化和唯一输出。"


def run_task_modeling_agent(
    topic: str,
    subject: str,
    retrieved_docs: List[Dict[str, Any]],
    problem_background: str = "",
    requested_algorithm: str = "auto",
    notes: str = "",
) -> Dict[str, Any]:
    docs_text = _format_docs_text(retrieved_docs)
    formula_spec = _detect_formula_spec(topic, f"{problem_background}\n{docs_text}")
    has_formula = bool(formula_spec)
    normalized_algorithm = _normalize_algorithm(requested_algorithm, notes)

    if normalized_algorithm == "formula_calculation" and not has_formula:
        algorithm_model = _default_algorithm(subject, has_formula)
    elif normalized_algorithm == "auto":
        algorithm_model = _default_algorithm(subject, has_formula)
    else:
        algorithm_model = normalized_algorithm

    if has_formula and normalized_algorithm not in {"auto", "formula_calculation"}:
        strategy = "formula_plus_algorithm"
    elif has_formula:
        strategy = "formula_driven"
    else:
        strategy = "contextual_algorithm"

    task_model = {
        "strategy": strategy,
        "topic": topic,
        "subject": subject,
        "requested_algorithm": normalized_algorithm,
        "algorithm_model": algorithm_model,
        "domain_context": (
            f"题目以“{topic}”作为学科背景；知识点用于解释题目场景，"
            "最终题目必须仍然是可判题的数据结构与算法题。"
        ),
        "has_formula": has_formula,
        "formula_name": formula_spec.get("name", ""),
        "knowledge_rule": formula_spec.get("formula", ""),
        "input_variables": formula_spec.get("input_variables", []),
        "output_variable": formula_spec.get("output_variable", "answer"),
        "task_rule": _task_rule_for_algorithm(topic, algorithm_model, formula_spec),
        "tie_rule": "若排序或比较结果相同，默认按输入编号从小到大处理。",
        "constraints_hint": "优先使用整数或有限位小数；浮点输出默认保留两位小数；保证样例可手算验证。",
        "sample_rows": formula_spec.get("sample_rows", []),
        "sample_outputs": formula_spec.get("sample_outputs", []),
    }

    errors: List[str] = []
    if not task_model["algorithm_model"]:
        errors.append("algorithm_model_missing")
    if strategy in {"formula_driven", "formula_plus_algorithm"} and not task_model["knowledge_rule"]:
        errors.append("formula_rule_missing")
    if not task_model["task_rule"]:
        errors.append("task_rule_missing")

    return {
        "task_model": task_model,
        "task_model_valid": not errors,
        "task_model_errors": errors,
    }
