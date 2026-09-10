from enterprise_agent.context import Context


def test_context_accepts_explicit_model() -> None:
    context = Context(model="openai/deepseek-chat")
    assert context.model == "openai/deepseek-chat"


def test_context_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("MODEL", "openai/qwen-turbo")
    monkeypatch.setenv("COMPANY_NAME", "测试公司")
    context = Context()
    assert context.model == "openai/qwen-turbo"
    assert context.company_name == "测试公司"
