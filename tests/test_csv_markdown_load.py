from pathlib import Path

from rag_project.pipeline.csv_load import CSV_SUFFIXES, load_csv_file
from rag_project.pipeline.markdown_load import MARKDOWN_SUFFIXES, load_markdown_file


def test_load_csv_file_reads_rows(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    source.write_text("name,city\nAda,Berlin\nLin,Paris\n", encoding="utf-8")

    documents = load_csv_file(source, encoding="utf-8")

    assert [doc.page_content for doc in documents] == [
        "name: Ada\ncity: Berlin",
        "name: Lin\ncity: Paris",
    ]
    assert [doc.metadata["row_index"] for doc in documents] == [0, 1]


def test_load_markdown_file_reads_text(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Titel\n\nErster Absatz.", encoding="utf-8")

    documents = load_markdown_file(source)

    assert documents
    assert "Titel" in documents[0].page_content
    assert "Erster Absatz." in documents[0].page_content


def test_main_loads_markdown_file(tmp_path: Path, capsys) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Titel\n\nErster Absatz.", encoding="utf-8")

    exit_code = __import__("rag_project.pipeline.markdown_load", fromlist=["main"]).main(
        [str(source)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "documents=1" in captured.out
    assert "Titel" in captured.out
    assert "Erster Absatz." in captured.out


def test_main_loads_cleans_and_chunks_markdown_file(tmp_path: Path, capsys) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Titel\n\nErster Absatz.", encoding="utf-8")

    exit_code = __import__("rag_project.pipeline.markdown_load", fromlist=["main"]).main(
        [str(source), "--clean", "--chunk"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Chunking" in captured.out
    assert "Chunk 0" in captured.out
    assert "Titel" in captured.out
    assert "Erster Absatz." in captured.out


def test_main_loads_and_cleans_csv_file(tmp_path: Path, capsys) -> None:
    source = tmp_path / "data.csv"
    source.write_text(
        "name,notes\nAda,  Vorname  \nLin,\u200bzweiter Punkt\n",
        encoding="utf-8",
    )

    exit_code = __import__("rag_project.pipeline.csv_load", fromlist=["main"]).main(
        [str(source), "--clean"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "2 Dokument(e)" in captured.out
    assert "name: Ada\nnotes: Vorname" in captured.out
    assert "name: Lin\nnotes: zweiter Punkt" in captured.out


def test_main_loads_cleans_and_chunks_csv_file(tmp_path: Path, capsys) -> None:
    source = tmp_path / "data.csv"
    source.write_text(
        "name,notes\nAda,  Vorname  \nLin,\u200bzweiter Punkt\n",
        encoding="utf-8",
    )

    exit_code = __import__("rag_project.pipeline.csv_load", fromlist=["main"]).main(
        [str(source), "--clean", "--chunk"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "2 Dokument(e)" in captured.out
    assert "Chunking" in captured.out
    assert "Chunk 0" in captured.out
    assert "name: Ada\nnotes: Vorname" in captured.out
    assert "name: Lin\nnotes: zweiter Punkt" in captured.out


def test_csv_and_markdown_suffixes_are_exposed() -> None:
    assert ".csv" in CSV_SUFFIXES
    assert ".md" in MARKDOWN_SUFFIXES
    assert ".markdown" in MARKDOWN_SUFFIXES
