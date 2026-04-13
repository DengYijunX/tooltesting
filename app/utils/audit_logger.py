import json
import os
from datetime import datetime
from typing import Any, Dict


class AuditLogger:
    def __init__(self, topic: str, base_dir: str = "audit_logs"):
        os.makedirs(base_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in topic)[:50]
        self.filepath = os.path.join(base_dir, f"{ts}_{safe_topic}.json")

        self.data = {
            "topic": topic,
            "created_at": datetime.now().isoformat(),
            "steps": []
        }
        self._flush()

    def log_step(self, step_name: str, input_data: Dict[str, Any], output_data: Dict[str, Any], error: str | None = None):
        record = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "output": output_data,
            "error": error,
        }
        self.data["steps"].append(record)
        self._flush()

    def _flush(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)