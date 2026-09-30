from pydantic import SecretStr

from rag_project.config import Settings
from rag_project.paths import resolve_documents


def test_relative_filename_falls_back_to_configured_docs_dir(tmp_path, monkeypatch):
    working_dir = tmp_path / "working"
    docs_dir = tmp_path / "documents"
    working_dir.mkdir()
    docs_dir.mkdir()
    source = docs_dir / "handbuch.txt"
    source.write_text("Inhalt", encoding="utf-8")
    monkeypatch.chdir(working_dir)
    settings = Settings(
        openrouter_api_key=SecretStr("test-key"),
        docs_dir=docs_dir,
    )

    assert resolve_documents(["handbuch.txt"], settings) == [source]


def test_relative_filename_in_working_dir_takes_precedence(tmp_path, monkeypatch):
    working_dir = tmp_path / "working"
    docs_dir = tmp_path / "documents"
    working_dir.mkdir()
    docs_dir.mkdir()
    source = working_dir / "handbuch.txt"
    source.write_text("Inhalt", encoding="utf-8")
    (docs_dir / "handbuch.txt").write_text("Anderer Inhalt", encoding="utf-8")
    monkeypatch.chdir(working_dir)
    settings = Settings(
        openrouter_api_key=SecretStr("test-key"),
        docs_dir=docs_dir,
    )

    assert resolve_documents(["handbuch.txt"], settings) == [source]
