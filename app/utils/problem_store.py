import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "storage" / "problem_bank.sqlite3"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS problems (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            subject TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            audit_file TEXT NOT NULL,
            problem_json TEXT NOT NULL,
            reference_solution_json TEXT NOT NULL,
            test_cases_json TEXT NOT NULL,
            task_model_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _row_to_record(row: sqlite3.Row, include_private: bool = False) -> Dict[str, Any]:
    problem = _json_loads(row["problem_json"], {})
    test_cases = _json_loads(row["test_cases_json"], [])
    record = {
        "id": row["id"],
        "topic": row["topic"],
        "subject": row["subject"],
        "title": row["title"],
        "created_at": row["created_at"],
        "audit_file": row["audit_file"],
        "problem_statement": problem,
        "task_model": _json_loads(row["task_model_json"], {}),
        "test_case_count": len(test_cases),
        "metadata": _json_loads(row["metadata_json"], {}),
    }
    if include_private:
        record["reference_solution"] = _json_loads(row["reference_solution_json"], {})
        record["test_cases"] = test_cases
    return record


def save_problem_from_workflow_result(result: Dict[str, Any]) -> Dict[str, Any]:
    final_result = result.get("final_result", {})
    if not isinstance(final_result, dict) or final_result.get("error"):
        raise ValueError("only successful workflow result can be saved")

    problem = final_result.get("problem_statement") or result.get("problem_statement") or {}
    if not problem.get("title"):
        raise ValueError("problem title is required")

    problem_id = uuid.uuid4().hex[:12]
    created_at = datetime.now().isoformat(timespec="seconds")
    topic = str(final_result.get("topic") or result.get("topic") or "").strip()
    subject = str(final_result.get("subject") or result.get("subject") or "").strip()
    title = str(problem.get("title", "")).strip()
    reference_solution = final_result.get("reference_solution") or {
        "language": result.get("language", "python"),
        "explanation": result.get("solution_explanation", ""),
        "code": result.get("reference_code", ""),
    }
    test_cases = final_result.get("test_cases") or result.get("test_cases") or []
    task_model = final_result.get("task_model") or result.get("task_model") or {}
    audit_file = str(final_result.get("audit_file") or result.get("audit_file") or "")
    metadata = {
        "requested_algorithm": final_result.get("requested_algorithm") or result.get("requested_algorithm", "auto"),
        "sandbox_passed": bool((final_result.get("sandbox_result") or result.get("sandbox_result") or {}).get("passed")),
        "knowledge_sufficiency": bool(
            (final_result.get("knowledge_check") or {}).get("knowledge_sufficiency")
            or result.get("knowledge_sufficiency")
        ),
    }

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO problems (
                id, topic, subject, title, created_at, audit_file,
                problem_json, reference_solution_json, test_cases_json,
                task_model_json, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                problem_id,
                topic,
                subject,
                title,
                created_at,
                audit_file,
                _json_dumps(problem),
                _json_dumps(reference_solution),
                _json_dumps(test_cases),
                _json_dumps(task_model),
                _json_dumps(metadata),
            ),
        )
        conn.commit()

    return get_problem(problem_id, include_private=False)


def list_problems(limit: int = 30) -> List[Dict[str, Any]]:
    limit = max(1, min(limit, 100))
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM problems ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_record(row, include_private=False) for row in rows]


def get_problem(problem_id: str, include_private: bool = False) -> Dict[str, Any]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM problems WHERE id = ?",
            (problem_id,),
        ).fetchone()
    if row is None:
        raise KeyError(problem_id)
    return _row_to_record(row, include_private=include_private)
