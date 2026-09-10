"""Demonstrate the deterministic tools without calling a paid model API."""

from __future__ import annotations

import json

from enterprise_agent.tools import create_task_draft, search_knowledge_base


def print_json(title: str, raw_payload: str) -> None:
    """Print a readable JSON tool result."""
    print(f"\n=== {title} ===")
    print(json.dumps(json.loads(raw_payload), ensure_ascii=False, indent=2))


def main() -> None:
    """Run one retrieval example and one task-draft example."""
    search_result = search_knowledge_base.invoke(
        {"query": "出差回来后报销需要哪些材料？", "top_k": 2}
    )
    task_result = create_task_draft.invoke(
        {
            "title": "整理客户高频问题",
            "objective": "汇总本周客户反馈并形成改进清单",
            "owner": "陈仕涛",
            "deadline": "本周五",
            "acceptance_criteria": "输出高频问题、出现次数和建议处理方案",
        }
    )
    print_json("知识库检索", search_result)
    print_json("任务草稿", task_result)


if __name__ == "__main__":
    main()
