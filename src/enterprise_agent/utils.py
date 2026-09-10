"""Utility functions shared by graph nodes."""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def get_message_text(message: BaseMessage) -> str:
    """Return plain text from a LangChain message."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(str(item.get("text", "")))
    return "".join(parts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a model and support OpenAI-compatible providers through environment config."""
    provider, model = fully_specified_name.split("/", maxsplit=1)
    model_kwargs: dict[str, Any] = {}

    if provider == "openai":
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        if base_url:
            model_kwargs["base_url"] = base_url

    return init_chat_model(model, model_provider=provider, **model_kwargs)
