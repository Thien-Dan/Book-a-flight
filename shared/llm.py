"""Khởi tạo lõi LLM DeepSeek. Cấu hình đọc từ file .env ở thư mục 'Bài tập buổi 3'."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)


def get_llm(temperature: float | None = None) -> ChatDeepSeek:
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key or key.startswith("sk-your"):
        raise RuntimeError("Chưa cấu hình DEEPSEEK_API_KEY trong file .env (thư mục Bài tập buổi 3).")
    return ChatDeepSeek(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=key,
        api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0")) if temperature is None else temperature,
        max_retries=2,
        timeout=90,
    )
