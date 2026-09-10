import json

from enterprise_agent.tools import create_task_draft, search_knowledge_base


def test_knowledge_tool_returns_citable_evidence() -> None:
    raw = search_knowledge_base.invoke({"query": "报销需要哪些材料", "top_k": 2})
    payload = json.loads(raw)

    assert payload["found"] is True
    assert payload["evidence"]
    assert payload["evidence"][0]["citation"].startswith("【来源：")


def test_task_tool_only_creates_a_draft() -> None:
    raw = create_task_draft.invoke(
        {
            "title": "整理客户反馈",
            "objective": "汇总本周高频问题",
            "owner": "陈仕涛",
        }
    )
    payload = json.loads(raw)

    assert payload["status"] == "draft"
    assert payload["draft_id"].startswith("TASK-")
    assert "尚未提交或审批" in payload["notice"]
