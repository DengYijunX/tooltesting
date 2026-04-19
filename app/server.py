from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ENABLED_SUBJECTS
from app.executors.sandbox import run_code_in_sandbox
from app.utils.problem_store import (
    get_problem,
    list_problems,
    save_problem_from_workflow_result,
)
from app.workflows.problem_generation_workflow import run_problem_generation_workflow


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class GenerateProblemRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=80)
    subject: str = Field("physics", min_length=1, max_length=32)
    mode: str = Field("verified", max_length=32)
    algorithm: str = Field("auto", max_length=64)
    notes: str = Field("", max_length=1000)


class SubmitSolutionRequest(BaseModel):
    language: str = Field("python", max_length=32)
    code: str = Field(..., min_length=1, max_length=20000)


app = FastAPI(
    title="RAG Problem Generation Platform",
    version="0.1.0",
    description="基于本地知识库和多阶段验证工作流的编程题生成服务。",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "enabled_subjects": ENABLED_SUBJECTS,
    }


@app.post("/api/generate-problem")
def generate_problem(request: GenerateProblemRequest) -> Dict[str, Any]:
    topic = request.topic.strip()
    subject = request.subject.strip()
    algorithm = request.algorithm.strip() or "auto"
    notes = request.notes.strip()

    if not topic:
        raise HTTPException(status_code=400, detail="topic 不能为空")
    if subject not in ENABLED_SUBJECTS:
        raise HTTPException(
            status_code=400,
            detail=f"subject {subject} 未启用，可用范围：{', '.join(ENABLED_SUBJECTS)}",
        )

    result = run_problem_generation_workflow(
        topic=topic,
        subject=subject,
        requested_algorithm=algorithm,
        notes=notes,
    )
    final_result = result.get("final_result", {})
    error = final_result.get("error") if isinstance(final_result, dict) else None
    problem_record: Dict[str, Any] | None = None

    if not error:
        try:
            problem_record = save_problem_from_workflow_result(result)
            if isinstance(final_result, dict):
                final_result["problem_id"] = problem_record["id"]
                final_result["problem_url"] = f"/problems/{problem_record['id']}"
        except Exception as exc:
            error = "problem_save_failed"
            result["final_result"] = {
                "error": error,
                "message": f"{type(exc).__name__}: {exc}",
                "audit_file": result.get("audit_file", ""),
            }

    return {
        "ok": not bool(error),
        "error": error or "",
        "result": result,
        "problem": problem_record,
    }


@app.get("/api/problems")
def problems(limit: int = 30) -> Dict[str, Any]:
    return {
        "problems": list_problems(limit=limit),
    }


@app.get("/api/problems/{problem_id}")
def problem_detail(problem_id: str) -> Dict[str, Any]:
    try:
        return {
            "problem": get_problem(problem_id, include_private=False),
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="题目不存在")


@app.post("/api/problems/{problem_id}/submit")
def submit_solution(problem_id: str, request: SubmitSolutionRequest) -> Dict[str, Any]:
    if request.language.strip().lower() != "python":
        raise HTTPException(status_code=400, detail="当前只支持 Python 判题")

    try:
        problem = get_problem(problem_id, include_private=True)
    except KeyError:
        raise HTTPException(status_code=404, detail="题目不存在")

    test_cases = problem.get("test_cases", [])
    if not test_cases:
        raise HTTPException(status_code=400, detail="题目没有可用测试用例")

    sandbox_result = run_code_in_sandbox(
        reference_code=request.code,
        test_cases=test_cases,
    )
    return {
        "passed": bool(sandbox_result.get("passed", False)),
        "sandbox_result": sandbox_result,
    }


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html 不存在")
    return FileResponse(index_path)


@app.get("/problems/{problem_id}")
def problem_page(problem_id: str) -> FileResponse:
    _ = problem_id
    page_path = FRONTEND_DIR / "problem.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="frontend/problem.html 不存在")
    return FileResponse(page_path)


if FRONTEND_DIR.exists():
    app.mount(
        "/frontend",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
