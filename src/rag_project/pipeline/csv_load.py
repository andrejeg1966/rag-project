"""CSV-Dateien und entfernte CSV-Quellen laden."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

CSV_SUFFIXES: frozenset[str] = frozenset({".csv"})


def _import_or_die(module: str, names: tuple[str, ...], package: str):
    """Importiert Namen und uebersetzt einen Importfehler in eine klare Meldung."""
    try:
        imported = __import__(module, fromlist=list(names))
    except ImportError as exc:
        raise RuntimeError(
            f"Fuer diese Quelle wird {package!r} gebraucht, ist aber nicht "
            f"installiert. Installieren mit: uv add {package} ({exc})"
        ) from exc
    return [getattr(imported, name) for name in names]


def load_csv_file(source: Path, encoding: str) -> list[Document]:
    """Laedt eine lokale CSV-Datei als LangChain-Dokumente."""
    try:
        (CSVLoader,) = _import_or_die(
            "langchain_community.document_loaders",
            ("CSVLoader",),
            "langchain-community",
        )
        documents = CSVLoader(str(source), encoding=encoding).load()
        for index, document in enumerate(documents):
            document.metadata.setdefault("row_index", index)
        return documents
    except (RuntimeError, TypeError):
        pass

    text = source.read_text(encoding=encoding, errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []

    documents: list[Document] = []
    for index, row in enumerate(reader):
        body = "\n".join(
            f"{name}: {(row.get(name) or '').strip()}" for name in fieldnames
        )
        documents.append(
            Document(
                page_content=body,
                metadata={"row_index": index},
            )
        )
    return documents


def load_remote_csv(
    url: str, on_event: Callable[[str], None] | None = None
) -> list[Document]:
    """Laedt eine entfernte CSV-Datei ueber HTTP."""
    (CSVLoader,) = _import_or_die(
        "langchain_community.document_loaders",
        ("CSVLoader",),
        "langchain-community",
    )
    if on_event is not None:
        on_event("[netz]  " + url)
    return CSVLoader(url).load()


def clean_csv_documents(
    documents: list[Document],
    *,
    drop_empty: bool = True,
    **options: object,
) -> tuple[list[Document], object]:
    """Bereinigt bereits geladene CSV-Dokumente."""
    from rag_project.pipeline.cleaning import clean_documents

    return clean_documents(documents, drop_empty=drop_empty, **options)


def chunk_csv_documents(
    documents: list[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> tuple[list[Document], object]:
    """Zerlegt bereits geladene Dokumente in Chunks."""
    from rag_project.pipeline.chunking import split_documents

    return split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def extract_write_option(arguments: list[str]) -> tuple[list[str], Path | None]:
    """Trennt ``--write DATEI`` von den uebrigen Argumenten."""
    rest = list(arguments)
    for flag in ("--write", "-write"):
        if flag in rest:
            index = rest.index(flag)
            if index + 1 >= len(rest):
                raise ValueError(f"{flag} braucht einen Dateipfad")
            target = Path(rest[index + 1]).expanduser()
            del rest[index : index + 2]
            return rest, target
    return rest, None


def write_documents(documents: list[Document], target: Path) -> None:
    """Schreibt die Dokumentinhalte als UTF-8-Datei.

    ``.txt`` u. a.: Leerzeilen als Trenner, Chunks mit Praefix ``Chunk N:``.
    ``.md``/``.markdown``: Chunks als ``### Chunk N`` getrennt durch ``---``.
    """
    markdown = target.suffix.lower() in {".md", ".markdown"}

    def render(document: Document) -> str:
        index = document.metadata.get("chunk_index")
        if index is None:
            return document.page_content
        if markdown:
            return f"### Chunk {index}\n\n{document.page_content}"
        return f"Chunk {index}: {document.page_content}"

    separator = "\n\n---\n\n" if markdown else "\n\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        separator.join(render(document) for document in documents) + "\n",
        encoding="utf-8",
    )
    print(f"[gespeichert] {target}")


def main(argv: list[str] | None = None) -> int:
    """Lädt CSV-Dokumente optional bereinigt und in Chunks aufgeteilt."""
    import sys

    usage = "Usage: csv-load <csv-file> [encoding] [--clean] [--chunk] [--write DATEI]"
    arguments = sys.argv[1:] if argv is None else argv
    try:
        arguments, write_target = extract_write_option(arguments)
    except ValueError as exc:
        print(f"{exc}\n{usage}", file=sys.stderr)
        return 2
    if not arguments:
        print(usage, file=sys.stderr)
        return 2

    source = Path(arguments[0]).expanduser() # Pfad expandieren, z.B. ~ -> /home/user
    clean = "--clean" in arguments
    chunk = "--chunk" in arguments
    encoding_arguments = [
        arg for arg in arguments[1:] if arg not in {"--clean", "--chunk"}
    ]

    if len(encoding_arguments) > 1:
        print(usage, file=sys.stderr)
        return 2

    encoding = encoding_arguments[0] if encoding_arguments else "utf-8"
    if arguments[0].lower().startswith(("http://", "https://")):
        documents = load_remote_csv(arguments[0], on_event=print)
    else:
        documents = load_csv_file(source, encoding)

    if clean:
        documents, cleaning_stats = clean_csv_documents(documents)
        print(cleaning_stats.summary())

    if chunk:
        documents, chunking_stats = chunk_csv_documents(documents)
        print(chunking_stats.summary())
        print("Chunking")
        for document in documents:
            print(f"Chunk {document.metadata.get('chunk_index')}: {document.page_content}")
    else:
        print(f"rows={len(documents)}")
        for document in documents:
            print(document.page_content)

    if write_target is not None:
        write_documents(documents, write_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CSV_SUFFIXES",
    "load_csv_file",
    "load_remote_csv",
    "clean_csv_documents",
    "main",
]
