"""Business tools exposed to the LangGraph agent."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from langchain_core.tools import tool

from enterprise_agent.knowledge_base import get_default_knowledge_base


@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Search enterprise policies and return citable evidence."""
    knowledge_base = get_default_knowledge_base()
    results = knowledge_base.search(query=query, top_k=top_k)
    if not results:
        return json.dumps(
            {
                "found": False,
                "message": "当前知识库没有找到依据，请联系对应负责人确认。",
                "evidence": [],
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "found": True,
            "message": "请仅依据以下证据回答，并在答案中保留 citation 字段。",
            "evidence": [result.as_dict() for result in results],
        },
        ensure_ascii=False,
    )


@tool
def create_task_draft(
    title: str,
    objective: str,
    owner: str = "待确认",
    deadline: str = "待确认",
    acceptance_criteria: str = "待确认",
) -> str:
    """Create a task draft without submitting it to an external system."""
    fingerprint = "|".join(
        [title, objective, owner, deadline, acceptance_criteria]
    ).encode("utf-8")
    draft_id = hashlib.sha256(fingerprint).hexdigest()[:8].upper()
    payload = {
        "draft_id": f"TASK-{draft_id}",
        "status": "draft",
        "title": title.strip(),
        "objective": objective.strip(),
        "owner": owner.strip() or "待确认",
        "deadline": deadline.strip() or "待确认",
        "acceptance_criteria": acceptance_criteria.strip() or "待确认",
        "steps": [
            "确认任务范围、输入资料和限制条件",
            "拆分执行步骤并确认负责人",
            "完成核心工作并记录过程",
            "按照验收标准进行自检",
            "整理结果、风险和待确认事项",
        ],
        "notice": "该结果仅为任务草稿，尚未提交或审批。",
    }
    return json.dumps(payload, ensure_ascii=False)


TOOLS: list[Callable[..., Any]] = [search_knowledge_base, create_task_draft]
