import pytest
from langchain_core.documents import Document

from rag_project.app import main_loading


def test_loading_writes_document_text(tmp_path, monkeypatch):
    target = tmp_path / "docs" / "page.txt"
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
        "--url",
        "https://example.com/page",
        "--allow-remote",
        "--write",
        str(target),
        "--quiet",
    ])

    assert result == 0
    assert target.read_text(encoding="utf-8") == "First article\n\nSecond article"


def test_loading_rejects_wikipedia_option():
    with pytest.raises(SystemExit):
        main_loading.build_parser().parse_args(["--wikipedia", "RAG"])