from langchain_core.documents import Document

from rag_project.app import main_loading


def test_loading_writes_document_text(tmp_path, monkeypatch):
    target = tmp_path / "docs" / "wikipedia.txt"
    monkeypatch.setattr(main_loading, "_assert_remote_enabled", lambda *_args: True)
    monkeypatch.setattr(
        main_loading,
        "load_documents",
        lambda *_args, **_kwargs: [
            Document(page_content="First article"),
            Document(page_content="Second article"),
        ],
    )

    result = main_loading.main([
        "--wikipedia",
        "Artificial intelligence",
        "--wiki-lang",
        "en",
        "--wiki-max-docs",
        "2",
        "--allow-remote",
        "--write",
        str(target),
        "--quiet",
    ])

    assert result == 0
    assert target.read_text(encoding="utf-8") == "First article\n\nSecond article"