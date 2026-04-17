import time
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

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        max_tokens: int = 600
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None
        for attempt in range(3):
            try:
                start = time.perf_counter()
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=180,
                )
                elapsed = time.perf_counter() - start
                print(f"LLM request elapsed: {elapsed:.3f}s")

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                last_error = e
                print(f"LLM request failed on attempt {attempt + 1}: {e}")
                time.sleep(2 * (attempt + 1))

        raise last_error