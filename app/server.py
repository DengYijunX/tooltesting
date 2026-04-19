from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ENABLED_SUBJECTS
from app.workflows.problem_generation_workflow import run_problem_generation_workflow


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class GenerateProblemRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=80)
    subject: str = Field("physics", min_length=1, max_length=32)
    mode: str = Field("verified", max_length=32)
    algorithm: str = Field("auto", max_length=64)
    notes: str = Field("", max_length=1000)


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

    return {
        "ok": not bool(error),
        "error": error or "",
        "result": result,
    }


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html 不存在")
    return FileResponse(index_path)


if FRONTEND_DIR.exists():
    app.mount(
        "/frontend",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
