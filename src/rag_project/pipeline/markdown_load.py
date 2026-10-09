"""Markdown-Dateien laden."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

MARKDOWN_SUFFIXES: frozenset[str] = frozenset({".md", ".markdown"})


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


def load_markdown_file(source: Path) -> list[Document]:
    """Laedt Markdown mit dem Unstructured-Loader oder TextLoader."""
    try:
        (MarkdownLoader,) = _import_or_die(
            "langchain_community.document_loaders",
            ("UnstructuredMarkdownLoader",),
            "unstructured",
        )
        return MarkdownLoader(str(source), mode="single").load()
    except (RuntimeError, ModuleNotFoundError):
        from langchain_community.document_loaders import TextLoader

        return TextLoader(
            str(source), encoding="utf-8", autodetect_encoding=True
        ).load()


def clean_markdown_documents(
    documents: list[Document],
    *,
    drop_empty: bool = True,
    **options: object,
) -> tuple[list[Document], object]:
    """Bereinigt bereits geladene Markdown-Dokumente."""
    from rag_project.pipeline.cleaning import clean_documents

    return clean_documents(documents, drop_empty=drop_empty, **options)


def chunk_markdown_documents(
    documents: list[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> tuple[list[Document], object]:
    """Zerlegt bereits geladene Markdown-Dokumente in Chunks."""
    from rag_project.pipeline.chunking import split_documents

    return split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def main(argv: list[str] | None = None) -> int:
    """Lädt eine Markdown-Datei optional bereinigt und in Chunks aufgeteilt."""
    import sys

    from rag_project.pipeline.csv_load import extract_write_option, write_documents

    usage = "Usage: markdown-load <markdown-file> [encoding] [--clean] [--chunk] [--write DATEI]"
    arguments = sys.argv[1:] if argv is None else argv
    try:
        arguments, write_target = extract_write_option(arguments)
    except ValueError as exc:
        print(f"{exc}\n{usage}", file=sys.stderr)
        return 2
    if not arguments:
        print(usage, file=sys.stderr)
        return 2

    source = Path(arguments[0]).expanduser()
    clean = "--clean" in arguments
    chunk = "--chunk" in arguments
    encoding_arguments = [
        arg for arg in arguments[1:] if arg not in {"--clean", "--chunk"}
    ]

    if len(encoding_arguments) > 1:
        print(usage, file=sys.stderr)
        return 2

    encoding = encoding_arguments[0] if encoding_arguments else "utf-8"
    documents = load_markdown_file(source)

    if encoding != "utf-8":
        source_text = source.read_text(encoding=encoding, errors="replace")
        documents = [Document(page_content=source_text, metadata={})]

    if clean:
        documents, cleaning_stats = clean_markdown_documents(documents)
        print(cleaning_stats.summary())

    if chunk:
        documents, chunking_stats = chunk_markdown_documents(documents)
        print(chunking_stats.summary())
        print("Chunking")
        for document in documents:
            print(f"Chunk {document.metadata.get('chunk_index')}: {document.page_content}")
    else:
        print(f"documents={len(documents)}")
        print(documents[0].page_content)

    if write_target is not None:
        write_documents(documents, write_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MARKDOWN_SUFFIXES", "load_markdown_file", "main"]
