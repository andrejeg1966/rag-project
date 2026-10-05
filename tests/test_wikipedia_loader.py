import wikipedia

from rag_project.pipeline import loading


def test_wikipedia_loader_sets_configured_user_agent(monkeypatch):
    observed = {}

    class FakeWikipediaLoader:
        def __init__(self, **kwargs):
            observed["loader_kwargs"] = kwargs

        def load(self):
            return []

    monkeypatch.setenv(
        "WIKIPEDIA_USER_AGENT",
        "rag-project-test/1.0 (contact: test@example.com)",
    )
    monkeypatch.setattr(
        wikipedia,
        "set_user_agent",
        lambda value: observed.update(user_agent=value),
    )
    monkeypatch.setattr(
        loading,
        "_import_or_die",
        lambda *_args: (FakeWikipediaLoader,),
    )

    assert loading._load_wikipedia("Künstliche Intelligenz") == []
    assert observed["user_agent"] == "rag-project-test/1.0 (contact: test@example.com)"
    assert observed["loader_kwargs"] == {
        "lang": "de",
        "load_max_docs": 1,
        "query": "Künstliche Intelligenz",
    }