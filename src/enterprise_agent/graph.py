"""LangGraph workflow for the enterprise knowledge and task agent."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from enterprise_agent.context import Context
from enterprise_agent.state import InputState, State
from enterprise_agent.tools import TOOLS
from enterprise_agent.utils import get_message_text, load_chat_model

CITATION_GUARD_TAG = "[CITATION_GUARD]"
CITATION_PATTERN = re.compile(r"【来源：[^】]+】")


async def call_model(
    state: State, runtime: Runtime[Context]
) -> dict[str, list[AIMessage]]:
    """Ask the configured model to answer or select a tool."""
    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)
    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat(),
        company_name=runtime.context.company_name,
    )
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="当前步骤数已达到上限，请缩小问题范围后重试。",
                )
            ]
        }
    return {"messages": [response]}


def route_model_output(state: State) -> Literal["tools", "citation_guard"]:
    """Route tool calls to execution and final answers to evidence validation."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Expected AIMessage, got {type(last_message).__name__}")
    return "tools" if last_message.tool_calls else "citation_guard"


def _current_turn_messages(state: State) -> list:
    messages = list(state.messages)
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


def needs_citation_retry(state: State) -> bool:
    """Return True once when an answer omitted or invented a source marker."""
    turn_messages = _current_turn_messages(state)
    guard_already_used = any(
        isinstance(message, SystemMessage)
        and get_message_text(message).startswith(CITATION_GUARD_TAG)
        for message in turn_messages
    )
    last_answer = next(
        (
            message
            for message in reversed(turn_messages)
            if isinstance(message, AIMessage) and not message.tool_calls
        ),
        None,
    )
    if not last_answer or guard_already_used:
        return False

    allowed_citations: set[str] = set()
    used_knowledge_base = False
    for message in turn_messages:
        if (
            not isinstance(message, ToolMessage)
            or message.name != "search_knowledge_base"
        ):
            continue
        used_knowledge_base = True
        try:
            payload = json.loads(get_message_text(message))
        except (json.JSONDecodeError, TypeError):
            continue
        allowed_citations.update(
            evidence["citation"]
            for evidence in payload.get("evidence", [])
            if isinstance(evidence, dict) and evidence.get("citation")
        )

    answer_citations = set(CITATION_PATTERN.findall(get_message_text(last_answer)))
    if not used_knowledge_base:
        return bool(answer_citations)
    return not answer_citations or not answer_citations.issubset(allowed_citations)


def citation_guard(state: State) -> dict[str, list[SystemMessage]]:
    """Ask the model for one corrected answer when source markers are invalid."""
    if not needs_citation_retry(state):
        return {}
    return {
        "messages": [
            SystemMessage(
                content=(
                    f"{CITATION_GUARD_TAG} 上一条回答的来源标记不符合要求。请重新回答："
                    "只能原样使用本轮 search_knowledge_base 工具实际返回的 citation；"
                    "如果本轮没有调用该工具，就删除所有来源、制度名和章节号。"
                    "同时只陈述用户输入和工具结果中已有的信息，不要补充虚构人名或企业事实。"
                )
            )
        ]
    }


def route_citation_guard(state: State) -> Literal["call_model", "__end__"]:
    """Retry the answer once only when the guard appended a correction instruction."""
    last_message = state.messages[-1]
    if isinstance(last_message, SystemMessage) and get_message_text(
        last_message
    ).startswith(CITATION_GUARD_TAG):
        return "call_model"
    return "__end__"


builder = StateGraph(State, input_schema=InputState, context_schema=Context)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("citation_guard", citation_guard)

builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")
builder.add_conditional_edges(
    "citation_guard",
    route_citation_guard,
    {"call_model": "call_model", "__end__": END},
)

graph = builder.compile(name="Enterprise Knowledge Agent")
