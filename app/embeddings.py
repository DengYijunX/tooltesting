import time
import requests
from typing import List
from langchain_core.embeddings import Embeddings

from app.config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    EMBED_MODEL,
    validate_config,
)


class SiliconFlowEmbeddings(Embeddings):
    def __init__(self, batch_size: int = 16):
        validate_config()
        self.api_key = SILICONFLOW_API_KEY
        self.base_url = SILICONFLOW_BASE_URL.rstrip("/")
        self.model = EMBED_MODEL
        self.endpoint = f"{self.base_url}/embeddings"
        self.batch_size = batch_size

    def _post_embedding(self, inputs: List[str]):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": inputs,
        }

        last_error = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=180,
                )
                if not resp.ok:
                    print("Embedding error:", resp.status_code, resp.text[:500])
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                last_error = e
                time.sleep(2 * (attempt + 1))

        raise last_error

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        clean_texts = [t if t and t.strip() else " " for t in texts]
        all_embeddings: List[List[float]] = []

        total = len(clean_texts)
        for i in range(0, total, self.batch_size):
            batch = clean_texts[i:i + self.batch_size]
            print(f"Embedding batch: {i + 1}-{min(i + self.batch_size, total)} / {total}")
            data = self._post_embedding(batch)
            all_embeddings.extend([item["embedding"] for item in data["data"]])

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        clean_text = text if text and text.strip() else " "
        data = self._post_embedding([clean_text])
        return data["data"][0]["embedding"]