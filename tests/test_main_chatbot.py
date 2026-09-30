from rag_project import main_chatbot


def test_chatbot_builds_streaming_client_on_init_and_system_reset(monkeypatch):
    calls = []

    def fake_build_llm(model, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(main_chatbot, "build_llm", fake_build_llm)

    bot = main_chatbot.ChatBot()
    bot.reset("A new system prompt")

    assert calls == [{"streaming": True}, {"streaming": True}]