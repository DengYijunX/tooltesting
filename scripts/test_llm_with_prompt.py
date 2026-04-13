import os
import sys
import time
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from app.config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, CHAT_MODEL


def main():
    url = f"{SILICONFLOW_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严谨、简洁、只依据给定要求回答的助手。不要展开，不要赘述。"
            },
            {
                "role": "user",
                "content": "请用两句话解释牛顿第二定律，并给出公式。"
            }
        ],
        "temperature": 0,
        "max_tokens": 180,
    }

    start = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    elapsed = time.perf_counter() - start

    print("status_code:", resp.status_code)
    print("elapsed_seconds:", round(elapsed, 3))

    resp.raise_for_status()
    data = resp.json()
    print("\n===== response =====\n")
    print(data["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()