from typing import Dict, Any

from app.utils.audit_logger import AuditLogger
from app.agents.retrieval_agent import run_retrieval_agent
from app.agents.background_agent import run_background_agent
from app.agents.problem_agent import run_problem_agent
from app.agents.solution_agent import run_solution_agent

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
    retrieval_output = run_retrieval_agent(topic, subject=subject)
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

    # Step 2: background
    background_output = run_background_agent(topic, state["retrieved_docs"])
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

    # Step 3: problem
    problem_output = run_problem_agent(
        topic=topic,
        problem_background=state["problem_background"],
        retrieval_summary=state.get("retrieval_summary", "")
    )
    state.update(problem_output)
    logger.log_step(
        step_name="problem_agent",
        input_data={
            "topic": topic,
            "subject": subject,
            "problem_background": state.get("problem_background", ""),
        },
        output_data=problem_output,
    )

    # 调试输出：临时保留，方便你看
    print("problem_statement_valid:", state.get("problem_statement_valid"))
    print("problem_statement_errors:", state.get("problem_statement_errors"))

    # 关键：题面无效时，立刻终止，不再继续 solution_agent
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

    # Step 4: solution
    solution_output = run_solution_agent(state["problem_statement"])
    state.update(solution_output)
    logger.log_step(
        step_name="solution_agent",
        input_data={
            "topic": topic,
            "subject": subject,
            "problem_statement": state["problem_statement"],
        },
        output_data=solution_output,
    )

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

    # Step 5: finalize success
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
        "audit_file": state["audit_file"],
    }

    logger.log_step(
        step_name="finalize_success",
        input_data={"topic": topic, "subject": subject},
        output_data=state["final_result"],
    )

    return state