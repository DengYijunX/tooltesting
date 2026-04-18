from typing import Dict, Any, List

from app.utils.audit_logger import AuditLogger
from app.agents.retrieval_agent import run_retrieval_agent
from app.agents.background_agent import run_background_agent
from app.agents.problem_agent import run_problem_agent
from app.agents.solution_agent import run_solution_agent
from app.agents.testcase_agent import run_testcase_agent
from app.executors.sandbox import run_code_in_sandbox
from app.validators.consistency import check_problem_solution_consistency
from app.validators.knowledge import check_knowledge_sufficiency
from app.validators.review import build_final_review


MAX_PROBLEM_ATTEMPTS = 2
MAX_SOLUTION_ATTEMPTS = 2
MAX_REPAIR_ROUNDS = 2


def _finalize_step_exception(
    logger: AuditLogger,
    state: Dict[str, Any],
    step_name: str,
    input_data: Dict[str, Any],
    exc: Exception,
) -> Dict[str, Any]:
    error_message = f"{type(exc).__name__}: {exc}"
    state["final_result"] = {
        "error": "agent_exception",
        "step": step_name,
        "message": error_message,
        "audit_file": state["audit_file"],
    }
    logger.log_step(
        step_name=step_name,
        input_data=input_data,
        output_data={},
        error=error_message,
    )
    return state


def _route_consistency_issues(issues: List[str]) -> str:
    problem_markers = (
        "problem_field_",
        "sample_",
    )
    solution_markers = (
        "reference_code_",
        "solution_explanation_",
    )

    if any(issue.startswith(solution_markers) for issue in issues):
        return "solution"
    if any(issue.startswith(problem_markers) for issue in issues):
        return "problem"
    return "solution"


def _build_consistency_feedback(issues: List[str], source_step: str) -> List[str]:
    return [f"{source_step}: {issue}" for issue in issues]


def _truncate_feedback(text: str, limit: int = 300) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _build_problem_feedback(
    issues: List[str],
    state: Dict[str, Any],
    source_step: str,
) -> List[str]:
    problem_statement = state.get("problem_statement", {})
    feedback: List[str] = []

    for issue in issues:
        if issue.startswith("problem_field_missing:"):
            field = issue.split(":", 1)[1].strip()
            feedback.append(f"{source_step}: 题面字段缺失，必须补全 {field}。")
        elif issue == "sample_input_line_count_mismatch":
            feedback.append(
                f"{source_step}: input_format 声明了多行输入，但 sample_input 的行数不匹配。"
                "必须让样例输入的换行结构与题面描述一致。"
            )
        elif issue == "sample_output_line_count_mismatch":
            feedback.append(
                f"{source_step}: output_format 声明了单行输出，但 sample_output 却是多行。"
                "必须让样例输出只保留题目要求的那一行结果。"
            )
        elif issue.startswith("suspicious_sample_content:"):
            field = issue.split(":", 1)[1].strip()
            feedback.append(
                f"{source_step}: {field} 含有可疑样例内容。样例里不要带引号、解释文本或单独的引号残片。"
            )
        elif issue.startswith("sample_"):
            feedback.append(f"{source_step}: 样例输入输出不完整，必须提供与题面一致的 sample_input 和 sample_output。")
        elif issue.startswith("problem_field_suspicious:"):
            field = issue.split(":", 1)[1].strip()
            feedback.append(f"{source_step}: 字段 {field} 含有脏文本或角色标记，必须清理。")
        else:
            feedback.append(f"{source_step}: {issue}")

    if problem_statement:
        feedback.append(
            "请保证题面字段自洽，尤其是 input_format、output_format、sample_input、sample_output 必须互相一致。"
        )

    return feedback


def _build_solution_feedback(
    issues: List[str],
    state: Dict[str, Any],
    source_step: str,
) -> List[str]:
    problem_statement = state.get("problem_statement", {})
    input_format = str(problem_statement.get("input_format", "")).strip()
    sample_input = str(problem_statement.get("sample_input", "")).strip()
    feedback: List[str] = []

    for issue in issues:
        if issue == "reference_code_input_line_count_mismatch":
            feedback.append(
                f"{source_step}: 题面要求的输入结构与代码读取方式不一致。"
                f"当前 input_format 为：{input_format}"
            )
            if sample_input:
                feedback.append(
                    f"{source_step}: 当前 sample_input 为多行输入：{sample_input}"
                )
            feedback.append(
                f"{source_step}: 如果 sample_input 是多行，代码必须逐行读取，"
                "至少调用足够次数的 input()，或者一次性读取全部输入后按行解析；"
                "禁止继续使用单次 input().split() 读取整份多行样例。"
            )
        elif issue == "reference_code_compile_failed":
            feedback.append(f"{source_step}: 返回的 code 不可编译，必须修复为可直接运行的 Python 代码。")
        elif issue == "reference_code_input_read_missing":
            feedback.append(f"{source_step}: 代码没有正确读取标准输入，必须显式读取输入数据。")
        elif issue == "reference_code_output_write_missing":
            feedback.append(f"{source_step}: 代码没有正确输出结果，必须向标准输出打印答案。")
        elif issue == "solution_explanation_empty":
            feedback.append(f"{source_step}: explanation 不能为空，至少用 1 到 2 句话说明思路。")
        elif issue == "python_code_compile_failed":
            feedback.append(f"{source_step}: 上一轮参考代码编译失败，必须返回完整且可编译的 Python 代码。")
        elif issue == "solution_compile_failed":
            feedback.append(f"{source_step}: 重新生成时优先修复语法和变量定义错误。")
        elif issue == "json_decode_failed":
            feedback.append(f"{source_step}: 上一轮 JSON 非法。必须只输出合法 JSON 对象，包含 language、explanation、code 三个字符串字段。")
        else:
            feedback.append(f"{source_step}: {issue}")

    return feedback


def _route_sandbox_failure(sandbox_result: Dict[str, Any]) -> str:
    details = sandbox_result.get("details", [])
    if not details:
        return "solution"
    if any(detail.get("error") for detail in details):
        return "solution"
    if any(detail.get("returncode", 0) != 0 for detail in details):
        return "solution"
    if any(str(detail.get("stderr", "")).strip() for detail in details):
        return "solution"
    return "problem"


def _build_sandbox_feedback(sandbox_result: Dict[str, Any]) -> List[str]:
    feedback: List[str] = []
    for detail in sandbox_result.get("details", []):
        case_id = detail.get("case_id")
        if detail.get("error"):
            feedback.append(f"sandbox_case_{case_id}: {detail['error']}")
            continue
        if detail.get("stderr"):
            feedback.append(
                f"sandbox_case_{case_id}_stderr: {_truncate_feedback(detail['stderr'])}"
            )
        if detail.get("expected_output") != detail.get("actual_output"):
            feedback.append(
                "sandbox_output_mismatch: "
                f"expected={detail.get('expected_output', '')!r}, "
                f"actual={detail.get('actual_output', '')!r}"
            )
    return feedback or ["sandbox_failed"]


def _build_problem_feedback_from_sandbox(
    sandbox_result: Dict[str, Any],
    state: Dict[str, Any],
    source_step: str,
) -> List[str]:
    problem_statement = state.get("problem_statement", {})
    sample_input = _truncate_feedback(str(problem_statement.get("sample_input", "")), 120)
    sample_output = _truncate_feedback(str(problem_statement.get("sample_output", "")), 120)
    feedback: List[str] = []

    for detail in sandbox_result.get("details", []):
        if detail.get("passed"):
            continue
        if detail.get("expected_output") != detail.get("actual_output"):
            actual_output = _truncate_feedback(str(detail.get("actual_output", "")), 120)
            expected_output = _truncate_feedback(str(detail.get("expected_output", "")), 120)
            case_input = _truncate_feedback(str(detail.get("input", "")), 120)
            feedback.append(
                f"{source_step}: 样例语义不自洽。当前测试输入 {case_input!r} 的期望输出是 {expected_output!r}，"
                f"但参考代码实际输出 {actual_output!r}。"
            )
            feedback.append(
                f"{source_step}: 请优先修正题面语义、sample_input、sample_output，使其与题目描述和标准解法一致。"
            )

    if sample_input or sample_output:
        feedback.append(
            f"{source_step}: 当前题面样例为 sample_input={sample_input!r}, sample_output={sample_output!r}。"
        )
    feedback.append(
        "请不要只改标题或措辞；必须让题目描述、输入输出格式、样例输入输出与参考解法表达的是同一个计算任务。"
    )

    return feedback


def _run_problem_attempts(
    logger: AuditLogger,
    state: Dict[str, Any],
    topic: str,
    subject: str,
    max_attempts: int,
    initial_feedback: List[str] | None = None,
    previous_output: str = "",
    round_index: int = 0,
) -> Dict[str, Any]:
    feedback = list(initial_feedback or [])
    prior_output = previous_output
    last_output: Dict[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        problem_output = run_problem_agent(
            topic=topic,
            problem_background=state["problem_background"],
            retrieval_summary=state.get("retrieval_summary", ""),
            feedback_issues=feedback or None,
            previous_output=prior_output,
        )
        logger.log_step(
            step_name=f"problem_agent_round_{round_index}_attempt_{attempt}",
            input_data={
                "topic": topic,
                "subject": subject,
                "problem_background": state.get("problem_background", ""),
                "feedback_issues": feedback,
            },
            output_data=problem_output,
        )

        last_output = problem_output
        if problem_output.get("problem_statement_valid", False):
            return problem_output

        feedback = _build_problem_feedback(
            problem_output.get("problem_statement_errors", []),
            state,
            f"problem_attempt_{attempt}",
        )
        prior_output = problem_output.get("raw_problem_output", "")

    return last_output or {
        "problem_statement": {},
        "raw_problem_output": "",
        "problem_statement_valid": False,
        "problem_statement_errors": ["problem_generation_failed"],
    }


def _run_solution_attempts(
    logger: AuditLogger,
    state: Dict[str, Any],
    topic: str,
    subject: str,
    max_attempts: int,
    initial_feedback: List[str] | None = None,
    previous_output: str = "",
    round_index: int = 0,
) -> Dict[str, Any]:
    feedback = list(initial_feedback or [])
    prior_output = previous_output
    last_output: Dict[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        solution_output = run_solution_agent(
            state["problem_statement"],
            feedback_issues=feedback or None,
            previous_output=prior_output,
        )
        logger.log_step(
            step_name=f"solution_agent_round_{round_index}_attempt_{attempt}",
            input_data={
                "topic": topic,
                "subject": subject,
                "problem_statement": state.get("problem_statement", {}),
                "feedback_issues": feedback,
            },
            output_data=solution_output,
        )

        last_output = solution_output
        if solution_output.get("solution_valid", False):
            return solution_output

        feedback = _build_solution_feedback(
            solution_output.get("solution_errors", []),
            state,
            f"solution_attempt_{attempt}",
        )
        prior_output = solution_output.get("raw_solution_output", "")

    return last_output or {
        "language": "python",
        "solution_explanation": "",
        "reference_code": "",
        "raw_solution_output": "",
        "solution_valid": False,
        "solution_errors": ["solution_generation_failed"],
    }


def is_valid_python_code(code: str) -> bool:
    try:
        compile(code, "<generated_code>", "exec")
        return True
    except Exception:
        return False

def run_problem_generation_workflow(topic: str, subject: str = "all") -> Dict[str, Any]:
    logger = AuditLogger(topic=topic)

    state: Dict[str, Any] = {
        "topic": topic,
        "subject": subject,
        "audit_file": logger.filepath,
    }

    # Step 1: retrieval
    try:
        retrieval_output = run_retrieval_agent(topic, subject=subject)
    except Exception as exc:
        return _finalize_step_exception(
            logger,
            state,
            "retrieval_agent",
            {"topic": topic, "subject": subject},
            exc,
        )
    state.update(retrieval_output)
    logger.log_step(
        step_name="retrieval_agent",
        input_data={"topic": topic, "subject": subject},
        output_data=retrieval_output,
    )

    # 学科未启用 / 检索直接失败
    if state.get("status") == "not_enabled":
        state["final_result"] = {
            "error": "subject_not_enabled",
            "message": state.get("retrieval_summary", ""),
            "audit_file": state["audit_file"],
        }
        logger.log_step(
            step_name="finalize_subject_not_enabled",
            input_data={"topic": topic, "subject": subject},
            output_data=state["final_result"],
        )
        return state

    # Step 1.5: knowledge sufficiency
    knowledge_output = check_knowledge_sufficiency(
        retrieved_docs=state.get("retrieved_docs", []),
        retrieval_summary=state.get("retrieval_summary", ""),
    )
    state.update(knowledge_output)
    logger.log_step(
        step_name="knowledge_sufficiency_check",
        input_data={
            "retrieved_docs_count": len(state.get("retrieved_docs", [])),
            "retrieval_summary": state.get("retrieval_summary", ""),
        },
        output_data=knowledge_output,
    )

    if not state.get("knowledge_sufficiency", False):
        state["final_result"] = {
            "error": "knowledge_insufficient",
            "details": state.get("knowledge_sufficiency_issues", []),
            "knowledge_stats": state.get("knowledge_stats", {}),
            "audit_file": state["audit_file"],
        }
        logger.log_step(
            step_name="finalize_knowledge_insufficient",
            input_data={"topic": topic, "subject": subject},
            output_data=state["final_result"],
        )
        return state

    # Step 2: background
    try:
        background_output = run_background_agent(topic, state["retrieved_docs"])
    except Exception as exc:
        return _finalize_step_exception(
            logger,
            state,
            "background_agent",
            {
                "topic": topic,
                "subject": subject,
                "retrieved_docs_count": len(state.get("retrieved_docs", [])),
            },
            exc,
        )
    state.update(background_output)
    logger.log_step(
        step_name="background_agent",
        input_data={
            "topic": topic,
            "subject": subject,
            "retrieved_docs_count": len(state["retrieved_docs"]),
        },
        output_data=background_output,
    )

    problem_feedback: List[str] = []
    problem_previous_output = ""
    solution_feedback: List[str] = []
    solution_previous_output = ""
    should_regenerate_problem = True

    for repair_round in range(MAX_REPAIR_ROUNDS + 1):
        round_index = repair_round + 1

        if should_regenerate_problem:
            try:
                problem_output = _run_problem_attempts(
                    logger=logger,
                    state=state,
                    topic=topic,
                    subject=subject,
                    max_attempts=MAX_PROBLEM_ATTEMPTS,
                    initial_feedback=problem_feedback,
                    previous_output=problem_previous_output,
                    round_index=round_index,
                )
            except Exception as exc:
                return _finalize_step_exception(
                    logger,
                    state,
                    "problem_agent",
                    {
                        "topic": topic,
                        "subject": subject,
                        "problem_background": state.get("problem_background", ""),
                        "feedback_issues": problem_feedback,
                    },
                    exc,
                )
            state.update(problem_output)
            problem_feedback = []
            problem_previous_output = state.get("raw_problem_output", "")
            solution_feedback = []
            solution_previous_output = ""
            should_regenerate_problem = False

            print("problem_statement_valid:", state.get("problem_statement_valid"))
            print("problem_statement_errors:", state.get("problem_statement_errors"))

            if not state.get("problem_statement_valid", False):
                state["final_result"] = {
                    "error": "problem_statement_invalid",
                    "details": state.get("problem_statement_errors", []),
                    "raw_problem_output": state.get("raw_problem_output", ""),
                    "audit_file": state["audit_file"],
                }
                logger.log_step(
                    step_name="finalize_problem_invalid",
                    input_data={
                        "topic": topic,
                        "subject": subject,
                    },
                    output_data=state["final_result"],
                )
                return state

        try:
            solution_output = _run_solution_attempts(
                logger=logger,
                state=state,
                topic=topic,
                subject=subject,
                max_attempts=MAX_SOLUTION_ATTEMPTS,
                initial_feedback=solution_feedback,
                previous_output=solution_previous_output,
                round_index=round_index,
            )
        except Exception as exc:
            return _finalize_step_exception(
                logger,
                state,
                "solution_agent",
                {
                    "topic": topic,
                    "subject": subject,
                    "problem_statement": state.get("problem_statement", {}),
                    "feedback_issues": solution_feedback,
                },
                exc,
            )
        state.update(solution_output)
        solution_feedback = []
        solution_previous_output = state.get("raw_solution_output", "")

        print("solution_valid:", state.get("solution_valid"))
        print("solution_errors:", state.get("solution_errors"))

        if not state.get("solution_valid", False):
            state["final_result"] = {
                "error": "solution_invalid",
                "details": state.get("solution_errors", []),
                "raw_solution_output": state.get("raw_solution_output", ""),
                "audit_file": state["audit_file"],
            }
            logger.log_step(
                step_name="finalize_solution_invalid",
                input_data={
                    "topic": topic,
                    "subject": subject,
                },
                output_data=state["final_result"],
            )
            return state

        consistency_output = check_problem_solution_consistency(
            problem_statement=state.get("problem_statement", {}),
            reference_code=state.get("reference_code", ""),
            solution_explanation=state.get("solution_explanation", ""),
        )
        state.update(consistency_output)
        logger.log_step(
            step_name=f"consistency_check_round_{round_index}",
            input_data={
                "problem_statement": state.get("problem_statement", {}),
                "reference_code": state.get("reference_code", ""),
                "solution_explanation": state.get("solution_explanation", ""),
            },
            output_data=consistency_output,
        )

        if not state.get("consistency_passed", False):
            if repair_round < MAX_REPAIR_ROUNDS:
                route = _route_consistency_issues(state.get("consistency_issues", []))
                feedback = _build_problem_feedback(
                    state.get("consistency_issues", []),
                    state,
                    "consistency_check",
                )
                if route == "problem":
                    problem_feedback = feedback
                    problem_previous_output = state.get("raw_problem_output", "")
                    should_regenerate_problem = True
                else:
                    solution_feedback = _build_solution_feedback(
                        state.get("consistency_issues", []),
                        state,
                        "consistency_check",
                    )
                    solution_previous_output = state.get("raw_solution_output", "")
                continue

            state["final_result"] = {
                "error": "consistency_invalid",
                "details": state.get("consistency_issues", []),
                "audit_file": state["audit_file"],
            }
            logger.log_step(
                step_name="finalize_consistency_invalid",
                input_data={
                    "topic": topic,
                    "subject": subject,
                },
                output_data=state["final_result"],
            )
            return state

        if not state.get("reference_code", "").strip():
            state["final_result"] = {
                "error": "solution_invalid",
                "details": ["reference_code_empty"],
                "raw_solution_output": state.get("raw_solution_output", ""),
                "audit_file": state["audit_file"],
            }
            logger.log_step(
                step_name="finalize_solution_invalid",
                input_data={
                    "topic": topic,
                    "subject": subject,
                },
                output_data=state["final_result"],
            )
            return state

        if not is_valid_python_code(state["reference_code"]):
            if repair_round < MAX_REPAIR_ROUNDS:
                solution_feedback = ["solution_compile_failed", "python_code_compile_failed"]
                solution_previous_output = state.get("raw_solution_output", "")
                continue

            state["final_result"] = {
                "error": "solution_invalid",
                "details": ["python_code_compile_failed"],
                "raw_solution_output": state.get("raw_solution_output", ""),
                "reference_code": state.get("reference_code", ""),
                "audit_file": state["audit_file"],
            }
            logger.log_step(
                step_name="finalize_solution_compile_failed",
                input_data={
                    "topic": topic,
                    "subject": subject,
                },
                output_data=state["final_result"],
            )
            return state

        testcase_output = run_testcase_agent(
            problem_statement=state.get("problem_statement", {}),
            reference_code=state.get("reference_code", ""),
        )
        state.update(testcase_output)
        logger.log_step(
            step_name=f"testcase_agent_round_{round_index}",
            input_data={
                "problem_statement": state.get("problem_statement", {}),
                "reference_code": state.get("reference_code", ""),
            },
            output_data=testcase_output,
        )

        if not state.get("testcase_generation_passed", False):
            if repair_round < MAX_REPAIR_ROUNDS:
                problem_feedback = _build_problem_feedback(
                    state.get("testcase_generation_issues", []),
                    state,
                    "testcase_generation",
                )
                problem_previous_output = state.get("raw_problem_output", "")
                should_regenerate_problem = True
                continue

            state["final_result"] = {
                "error": "testcase_invalid",
                "details": state.get("testcase_generation_issues", []),
                "audit_file": state["audit_file"],
            }
            logger.log_step(
                step_name="finalize_testcase_invalid",
                input_data={
                    "topic": topic,
                    "subject": subject,
                },
                output_data=state["final_result"],
            )
            return state

        sandbox_result = run_code_in_sandbox(
            reference_code=state.get("reference_code", ""),
            test_cases=state.get("test_cases", []),
        )
        state["sandbox_result"] = sandbox_result
        logger.log_step(
            step_name=f"sandbox_executor_round_{round_index}",
            input_data={
                "reference_code": state.get("reference_code", ""),
                "test_cases": state.get("test_cases", []),
            },
            output_data={"sandbox_result": sandbox_result},
        )

        if not sandbox_result.get("passed", False):
            if repair_round < MAX_REPAIR_ROUNDS:
                route = _route_sandbox_failure(sandbox_result)
                if route == "problem":
                    problem_feedback = _build_problem_feedback_from_sandbox(
                        sandbox_result,
                        state,
                        "sandbox_executor",
                    )
                    problem_previous_output = state.get("raw_problem_output", "")
                    should_regenerate_problem = True
                else:
                    feedback = _build_sandbox_feedback(sandbox_result)
                    solution_feedback = _build_solution_feedback(
                        feedback,
                        state,
                        "sandbox_executor",
                    )
                    solution_previous_output = state.get("raw_solution_output", "")
                continue

            state["final_result"] = {
                "error": "sandbox_invalid",
                "details": sandbox_result.get("details", []),
                "audit_file": state["audit_file"],
            }
            logger.log_step(
                step_name="finalize_sandbox_invalid",
                input_data={
                    "topic": topic,
                    "subject": subject,
                },
                output_data=state["final_result"],
            )
            return state

        break

    # Step 8: review aggregation
    final_review = build_final_review(state)
    state["final_review"] = final_review
    logger.log_step(
        step_name="review_aggregator",
        input_data={
            "knowledge_sufficiency": state.get("knowledge_sufficiency"),
            "problem_statement_valid": state.get("problem_statement_valid"),
            "solution_valid": state.get("solution_valid"),
            "consistency_passed": state.get("consistency_passed"),
            "testcase_generation_passed": state.get("testcase_generation_passed"),
            "sandbox_result": state.get("sandbox_result", {}),
        },
        output_data={"final_review": final_review},
    )

    if not final_review.get("valid", False):
        state["final_result"] = {
            "error": "final_review_invalid",
            "details": final_review.get("issues", []),
            "suggestions": final_review.get("suggestions", []),
            "audit_file": state["audit_file"],
        }
        logger.log_step(
            step_name="finalize_review_invalid",
            input_data={
                "topic": topic,
                "subject": subject,
            },
            output_data=state["final_result"],
        )
        return state

    # Step 9: finalize success
    state["final_result"] = {
        "topic": topic,
        "subject": subject,
        "background": state.get("problem_background", ""),
        "problem_statement": state.get("problem_statement", {}),
        "reference_solution": {
            "language": state.get("language", "python"),
            "explanation": state.get("solution_explanation", ""),
            "code": state.get("reference_code", ""),
        },
        "knowledge_check": {
            "knowledge_sufficiency": state.get("knowledge_sufficiency", False),
            "issues": state.get("knowledge_sufficiency_issues", []),
            "stats": state.get("knowledge_stats", {}),
        },
        "consistency_check": {
            "consistency_passed": state.get("consistency_passed", False),
            "issues": state.get("consistency_issues", []),
        },
        "test_cases": state.get("test_cases", []),
        "sandbox_result": state.get("sandbox_result", {}),
        "final_review": state.get("final_review", {}),
        "audit_file": state["audit_file"],
    }

    logger.log_step(
        step_name="finalize_success",
        input_data={"topic": topic, "subject": subject},
        output_data=state["final_result"],
    )

    return state
