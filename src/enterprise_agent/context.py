"""Runtime configuration for the enterprise knowledge agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from . import prompts


@dataclass(kw_only=True)
class Context:
    """Configuration that can be changed without rebuilding the graph."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={"description": "System instructions used by the agent."},
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/qwen-plus",
        metadata={
            "description": "Chat model in provider/model format. The default uses "
            "Qwen through its OpenAI-compatible endpoint."
        },
    )

    company_name: str = field(
        default="示例科技公司",
        metadata={"description": "Company name shown in the system prompt."},
    )

    def __post_init__(self) -> None:
        """Read matching environment variables when values were not supplied."""
        for item in fields(self):
            if not item.init:
                continue
            if getattr(self, item.name) == item.default:
                setattr(
                    self, item.name, os.environ.get(item.name.upper(), item.default)
                )
