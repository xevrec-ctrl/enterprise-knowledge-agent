from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from enterprise_agent.graph import (
    CITATION_GUARD_TAG,
    citation_guard,
    needs_citation_retry,
    route_citation_guard,
    route_model_output,
)
from enterprise_agent.state import State


def test_tool_call_is_routed_to_tool_node() -> None:
    state = State(
        messages=[
            HumanMessage(content="报销需要什么？"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"query": "报销材料"},
                        "id": "call-1",
                    }
                ],
            ),
        ]
    )
    assert route_model_output(state) == "tools"


def test_guard_retries_answer_without_citation_once() -> None:
    state = State(
        messages=[
            HumanMessage(content="报销需要什么材料？"),
            ToolMessage(
                content='{"evidence": [{"citation": "【来源：费用报销制度.md#所需材料】"}]}',
                tool_call_id="call-1",
                name="search_knowledge_base",
            ),
            AIMessage(content="需要发票和支付凭证。"),
        ]
    )

    assert needs_citation_retry(state) is True
    update = citation_guard(state)
    assert update["messages"][0].content.startswith(CITATION_GUARD_TAG)


def test_guard_accepts_answer_with_citation() -> None:
    state = State(
        messages=[
            HumanMessage(content="报销需要什么材料？"),
            ToolMessage(
                content=(
                    '{"evidence": [{"citation": "【来源：费用报销制度.md#所需材料】"}]}'
                ),
                tool_call_id="call-1",
                name="search_knowledge_base",
            ),
            AIMessage(content="需要发票。【来源：费用报销制度.md#所需材料】"),
        ]
    )

    assert needs_citation_retry(state) is False
    assert citation_guard(state) == {}
    assert route_citation_guard(state) == "__end__"


def test_guard_does_not_loop_after_retry_instruction() -> None:
    state = State(
        messages=[
            HumanMessage(content="报销需要什么材料？"),
            ToolMessage(
                content="evidence",
                tool_call_id="call-1",
                name="search_knowledge_base",
            ),
            AIMessage(content="需要发票。"),
            SystemMessage(content=f"{CITATION_GUARD_TAG} 请补充来源。"),
            AIMessage(content="仍然没有来源。"),
        ]
    )

    assert needs_citation_retry(state) is False
    assert citation_guard(state) == {}


def test_guard_retries_invented_citation_without_knowledge_search() -> None:
    state = State(
        messages=[
            HumanMessage(content="生成任务草稿。"),
            ToolMessage(
                content='{"status": "draft", "notice": "尚未提交或审批。"}',
                tool_call_id="call-1",
                name="create_task_draft",
            ),
            AIMessage(content="草稿已生成。【来源：不存在的制度.md#任务规则】"),
        ]
    )

    assert needs_citation_retry(state) is True


def test_guard_retries_citation_not_returned_by_search() -> None:
    state = State(
        messages=[
            HumanMessage(content="报销需要什么材料？"),
            ToolMessage(
                content=(
                    '{"evidence": [{"citation": "【来源：费用报销制度.md#所需材料】"}]}'
                ),
                tool_call_id="call-1",
                name="search_knowledge_base",
            ),
            AIMessage(content="需要发票。【来源：员工手册.md#报销】"),
        ]
    )

    assert needs_citation_retry(state) is True
