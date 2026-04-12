import os
from pathlib import Path
from dotenv import load_dotenv
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

api_key = os.getenv("SILICONFLOW_API_KEY")
base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

print("api_key exists:", bool(api_key))
print("api_key length:", len(api_key) if api_key else 0)
print("base_url:", base_url)

url = f"{base_url}/embeddings"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "BAAI/bge-m3",
    "input": ["你好，这是一个测试"]
}

resp = requests.post(url, headers=headers, json=payload, timeout=60)
print(resp.status_code)
print(resp.text)