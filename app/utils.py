"""工具函数"""
import os
import httpx
from app.config import settings


def check_ollama_service():
    """检查Ollama服务是否运行"""
    try:
        response = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
        return response.status_code == 200
    except Exception:
        return False


def ensure_directories():
    """确保必要的目录存在"""
    os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)