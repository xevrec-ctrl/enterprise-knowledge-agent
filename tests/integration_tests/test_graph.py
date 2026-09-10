import os

import pytest

from enterprise_agent import graph
from enterprise_agent.context import Context

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "1",
        reason="Set RUN_INTEGRATION_TESTS=1 and configure a model API key.",
    ),
]


async def test_agent_answers_with_a_source() -> None:
    result = await graph.ainvoke(
        {"messages": [("user", "报销需要准备哪些材料？")]},
        context=Context(),
    )
    assert "【来源：" in str(result["messages"][-1].content)
