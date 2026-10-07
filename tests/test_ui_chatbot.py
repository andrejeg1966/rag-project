from rag_project.app import ui_chatbot


def test_streaming_web_chatbot_uses_resolved_model_settings(monkeypatch):
    calls = []

    class FakeLLM:
        def stream(self, history):
            calls.append(history)
            yield "Hello"
            yield " world"

    def fake_build_llm(model, **kwargs):
        assert model == "openai:gpt-4o-mini"
        assert kwargs["streaming"] is True
        assert kwargs["temperature"] == 0.35
        return FakeLLM()

    monkeypatch.setattr(ui_chatbot, "build_llm", fake_build_llm)

    bot = ui_chatbot.StreamingWebChatbot(model="openai:gpt-4o-mini", temperature=0.35)
    result = list(bot.respond("What?", [], "System", "openai:gpt-4o-mini", 0.35))

    assert result[:-1] == ["Hello", "Hello world"]
    assert "Model: GPT-4o mini" in result[-1]
    assert "Provider: openai" in result[-1]
    assert "Base URL: https://api.openai.com/v1" in result[-1]
    assert "Temperature: 0.35" in result[-1]
    assert len(calls) == 1


def test_to_langchain_messages_supports_gradio_message_dicts():
    history = [
        {"role": "user", "content": "was ist python list"},
        {"role": "assistant", "content": "Eine Liste ..."},
        {"role": "user", "content": "was ist ein python dict"},
    ]

    messages = ui_chatbot._to_langchain_messages(history)

    assert [message.type for message in messages] == [
        "human",
        "ai",
        "human",
    ]
    assert [message.content for message in messages] == [
        "was ist python list",
        "Eine Liste ...",
        "was ist ein python dict",
    ]


def test_model_choices_include_current_models():
    chosen = {item[1] for item in ui_chatbot.MODEL_CHOICES}
    assert chosen == {
        "openai:gpt-4o-mini",
        "openai:gpt-4o",
        "openai:GPT-5 Pro",
        "openai:GPT-5.6 Luna",
        "openrouter:deepseek/deepseek-v4-flash",
        "openrouter:gpt-5.6-luna",
    }
