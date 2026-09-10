from pathlib import Path

from enterprise_agent.knowledge_base import KnowledgeBase, tokenize


def test_chinese_tokenizer_keeps_useful_bigrams() -> None:
    tokens = tokenize("费用报销需要发票")
    assert "报销" in tokens
    assert "发票" in tokens


def test_search_returns_relevant_section(tmp_path: Path) -> None:
    (tmp_path / "policy.md").write_text(
        "# 费用制度\n\n## 报销时限\n费用发生后 30 天内提交报销。\n\n"
        "## 请假流程\n请假需要直属负责人审批。\n",
        encoding="utf-8",
    )
    knowledge_base = KnowledgeBase(tmp_path)
    results = knowledge_base.search("报销最晚什么时候提交", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.section == "报销时限"
    assert results[0].chunk.citation == "【来源：policy.md#报销时限】"


def test_empty_directory_returns_no_evidence(tmp_path: Path) -> None:
    assert KnowledgeBase(tmp_path).search("任意问题") == []
