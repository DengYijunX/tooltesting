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
    def __init__(self):
        validate_config()
        self.api_key = SILICONFLOW_API_KEY
        self.base_url = SILICONFLOW_BASE_URL.rstrip("/")
        self.model = EMBED_MODEL
        self.endpoint = f"{self.base_url}/embeddings"

    def _post_embedding(self, inputs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": inputs,
        }

        resp = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        clean_texts = [t if t and t.strip() else " " for t in texts]
        data = self._post_embedding(clean_texts)
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        clean_text = text if text and text.strip() else " "
        data = self._post_embedding([clean_text])
        return data["data"][0]["embedding"]