import requests
from typing import List, Dict

from app.config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    CHAT_MODEL,
    validate_config,
)


class SiliconFlowLLM:
    def __init__(self):
        validate_config()
        self.api_key = SILICONFLOW_API_KEY
        self.base_url = SILICONFLOW_BASE_URL.rstrip("/")
        self.model = CHAT_MODEL

    def chat(self, messages: List[Dict]) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]