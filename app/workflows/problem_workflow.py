from typing import Dict, Any

from app.utils.audit_logger import AuditLogger
from app.agents.retrieval_agent import run_retrieval_agent
from app.agents.modeling_agent import run_modeling_agent
from app.agents.drafting_agent import run_drafting_agent


def run_problem_workflow(topic: str, subject: str = "all") -> Dict[str, Any]:
    logger = AuditLogger(topic=topic)

    state: Dict[str, Any] = {
        "topic": topic,
        "subject": subject,
        "audit_file": logger.filepath,
    }

    retrieval_output = run_retrieval_agent(topic, subject=subject)
    state.update(retrieval_output)
    logger.log_step(
        step_name="retrieval_agent",
        input_data={"topic": topic, "subject": subject},
        output_data=retrieval_output,
    )

    if state.get("status") == "not_enabled":
        state["final_problem"] = retrieval_output["retrieval_summary"]
        logger.log_step(
            step_name="finalize",
            input_data={"topic": topic, "subject": subject},
            output_data={"final_problem": state["final_problem"]},
        )
        return state

    modeling_output = run_modeling_agent(topic, state["retrieved_docs"])
    state.update(modeling_output)
    logger.log_step(
        step_name="modeling_agent",
        input_data={
            "topic": topic,
            "subject": subject,
            "retrieved_docs_count": len(state["retrieved_docs"]),
        },
        output_data=modeling_output,
    )

    drafting_output = run_drafting_agent(topic, state["knowledge_schema"])
    state.update(drafting_output)
    logger.log_step(
        step_name="drafting_agent",
        input_data={
            "topic": topic,
            "subject": subject,
            "knowledge_schema": state["knowledge_schema"],
        },
        output_data=drafting_output,
    )

    state["final_problem"] = state["draft_problem"]

    logger.log_step(
        step_name="finalize",
        input_data={"topic": topic, "subject": subject},
        output_data={"final_problem": state["final_problem"]},
    )

    return state