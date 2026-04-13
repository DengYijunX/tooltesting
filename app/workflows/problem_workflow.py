from typing import Dict, Any

from app.utils.audit_logger import AuditLogger
from app.agents.retrieval_agent import run_retrieval_agent
from app.agents.modeling_agent import run_modeling_agent
from app.agents.drafting_agent import run_drafting_agent
from app.agents.reviewer_agent import run_reviewer_agent


def run_problem_workflow(topic: str) -> Dict[str, Any]:
    logger = AuditLogger(topic=topic)

    state: Dict[str, Any] = {
        "topic": topic,
        "audit_file": logger.filepath,
    }

    # Step 1: retrieval
    retrieval_output = run_retrieval_agent(topic)
    state.update(retrieval_output)
    logger.log_step(
        step_name="retrieval_agent",
        input_data={"topic": topic},
        output_data=retrieval_output,
    )

    # Step 2: modeling
    modeling_output = run_modeling_agent(topic, state["retrieved_docs"])
    state.update(modeling_output)
    logger.log_step(
        step_name="modeling_agent",
        input_data={
            "topic": topic,
            "retrieved_docs_count": len(state["retrieved_docs"]),
        },
        output_data=modeling_output,
    )

    # Step 3: drafting
    drafting_output = run_drafting_agent(topic, state["knowledge_schema"])
    state.update(drafting_output)
    logger.log_step(
        step_name="drafting_agent",
        input_data={
            "topic": topic,
            "knowledge_schema": state["knowledge_schema"],
        },
        output_data=drafting_output,
    )

    # Step 4: review
    review_output = run_reviewer_agent(topic, state["retrieved_docs"], state["draft_problem"])
    state.update(review_output)
    logger.log_step(
        step_name="reviewer_agent",
        input_data={
            "topic": topic,
            "draft_problem": state["draft_problem"],
        },
        output_data=review_output,
    )

    # Step 5: finalize
    if state["review_passed"]:
        state["final_problem"] = state["draft_problem"]
    else:
        state["final_problem"] = (
            "题目审核未通过，请根据审计日志查看问题并修改。\n\n"
            + state["draft_problem"]
            + "\n\n【审核意见】\n"
            + state["review_result"]
        )

    logger.log_step(
        step_name="finalize",
        input_data={"review_passed": state["review_passed"]},
        output_data={"final_problem": state["final_problem"]},
    )

    return state