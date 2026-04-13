import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

CHAT_MODEL = os.getenv("SF_CHAT_MODEL", "Qwen/Qwen3-8B")
EMBED_MODEL = os.getenv("SF_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("SF_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_DIR = os.getenv("INDEX_DIR", "storage/faiss_index")

RECALL_K = int(os.getenv("RECALL_K", "8"))
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "2"))

SUPPORTED_SUBJECTS = [
    "psychology",
    "physics",
    "math",
    "finance",
    "mixed",
    "all",
]

ENABLED_SUBJECTS = [
    "psychology",
    "physics",
    "mixed",
    "all",
]

def validate_config():
    if not SILICONFLOW_API_KEY:
        raise ValueError(f"缺少 SILICONFLOW_API_KEY，请检查 {ENV_PATH}")