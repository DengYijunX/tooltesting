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

    context = """
[片段1] 普通高中教科书_物理必修_第一册.pdf | page=98
牛顿第二定律指出，物体加速度的大小跟它受到的作用力成正比，跟它的质量成反比，
加速度的方向跟作用力的方向相同。表达式为 F = ma。

[片段2] 普通高中教科书_物理必修_第一册.pdf | page=99
在国际单位制中，力的单位是牛顿，质量的单位是千克，加速度的单位是米每二次方秒。
"""

    prompt = f"""请严格依据上下文回答，不要编造。

问题：
牛顿第二定律是什么？

上下文：
{context}

请输出：
1. 简明答案
2. 依据说明
"""

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个严格依据上下文回答的助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 300,
    }

    start = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    elapsed = time.perf_counter() - start

    print("status_code:", resp.status_code)
    print("elapsed_seconds:", round(elapsed, 3))
    print("prompt_length:", len(prompt))

    resp.raise_for_status()
    data = resp.json()
    print("\n===== response =====\n")
    print(data["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()