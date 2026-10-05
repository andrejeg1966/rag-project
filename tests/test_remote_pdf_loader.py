import pytest
from langchain_core.documents import Document

from rag_project.pipeline import loading


def test_load_document_downloads_pdf_url_and_sets_metadata(monkeypatch, tmp_path):
    url = "https://example.test/archive/scan.PDF?download=1"
    pdf_bytes = b"%PDF-1.7\nexample pdf bytes"
    observed = {}

    class FakeResponse:
        content = pdf_bytes

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            pass

    class FakePyPDFLoader:
        def __init__(self, file_path):
            observed["file_path"] = file_path

        def load(self):
            with open(observed["file_path"], "rb") as pdf_file:
                observed["downloaded_bytes"] = pdf_file.read()
            return [Document(page_content="raw OCR text", metadata={"page": 1})]

    def fake_get(requested_url, **kwargs):
        observed["request"] = requested_url, kwargs
        return FakeResponse()

    monkeypatch.setattr(loading.requests, "get", fake_get)
    monkeypatch.setattr(loading, "PyPDFLoader", FakePyPDFLoader)
    monkeypatch.setattr(loading, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("USER_AGENT", "rag-project-test/1.0")

    documents = loading.load_document(url)

    assert len(documents) == 1
    assert documents[0].page_content == "raw OCR text"
    assert documents[0].metadata == {
        "page": 1,
        "source": url,
        "file_name": "scan.PDF",
        "format": "pdf",
        "size_bytes": len(pdf_bytes),
        "page_index": 0,
    }
    assert observed["request"] == (
        url,
        {"headers": {"User-Agent": "rag-project-test/1.0"}, "timeout": 30},
    )
    assert observed["downloaded_bytes"] == pdf_bytes
    assert (tmp_path / "docs" / "scan.PDF").read_bytes() == pdf_bytes


def test_load_document_keeps_html_urls_on_web_loader(monkeypatch):
    url = "https://example.test/page"
    monkeypatch.setattr(
        loading,
        "_load_url",
        lambda source, _on_event: [Document(page_content=f"HTML from {source}")],
    )

    documents = loading.load_document(url)

    assert documents[0].page_content == f"HTML from {url}"
    assert documents[0].metadata["format"] == "web"


def test_load_document_rejects_non_pdf_content_from_pdf_url(monkeypatch):
    class FakeResponse:
        content = b"<html>not a PDF</html>"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            pass

    monkeypatch.setattr(loading.requests, "get", lambda *_args, **_kwargs: FakeResponse())

    with pytest.raises(ValueError, match="keine gueltige PDF-Datei"):
        loading.load_document("https://example.test/not-a-pdf.pdf")


def test_load_document_does_not_overwrite_existing_pdf(monkeypatch, tmp_path):
    url = "https://example.test/archive/scan.pdf"
    pdf_bytes = b"%PDF-1.7\nnew pdf bytes"
    docs_directory = tmp_path / "docs"
    docs_directory.mkdir()
    existing_pdf = docs_directory / "scan.pdf"
    existing_pdf.write_bytes(b"existing file")

    class FakeResponse:
        content = pdf_bytes

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            pass

    class FakePyPDFLoader:
        def __init__(self, _file_path):
            pass

        def load(self):
            return [Document(page_content="raw OCR text")]

    monkeypatch.setattr(loading.requests, "get", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(loading, "PyPDFLoader", FakePyPDFLoader)
    monkeypatch.setattr(loading, "PROJECT_ROOT", tmp_path)

    loading.load_document(url)

    assert existing_pdf.read_bytes() == b"existing file"
    assert (docs_directory / "scan_1.pdf").read_bytes() == pdf_bytes