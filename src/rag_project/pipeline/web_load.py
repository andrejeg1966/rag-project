"""Webseiten per URL mit dem WebBaseLoader laden."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

URL_PREFIXES: tuple[str, ...] = ("http://", "https://")

EventHook = Callable[[str], None]


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


def load_web_page(url: str, on_event: EventHook | None = None) -> list[Document]:
    """Laedt eine Webseite als Text.

    :raises ValueError: wenn die Adresse nicht mit http:// oder https:// beginnt.
    """
    url = url.strip()
    if not url.lower().startswith(URL_PREFIXES):
        raise ValueError(f"Keine gueltige URL (http:// oder https:// erwartet): {url}")

    (WebBaseLoader,) = _import_or_die(
        "langchain_community.document_loaders", ("WebBaseLoader",), "beautifulsoup4"
    )
    user_agent = os.getenv("USER_AGENT")
    headers = {"User-Agent": user_agent} if user_agent else None
    if on_event is not None:
        on_event(f"[netz]  {url}")

    documents = WebBaseLoader(url, header_template=headers).load()
    for index, document in enumerate(documents):
        document.metadata.update(
            {
                "source": url,
                "file_name": url,
                "format": "web",
                "page_index": index,
            }
        )
    return documents


def clean_web_documents(
    documents: list[Document],
    *,
    drop_empty: bool = True,
    **options: object,
) -> tuple[list[Document], object]:
    """Bereinigt bereits geladene Webseiten-Dokumente."""
    from rag_project.pipeline.cleaning import clean_documents

    return clean_documents(documents, drop_empty=drop_empty, **options)


def chunk_web_documents(
    documents: list[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> tuple[list[Document], object]:
    """Zerlegt bereits geladene Webseiten-Dokumente in Chunks."""
    from rag_project.pipeline.chunking import split_documents

    return split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def main(argv: list[str] | None = None) -> int:
    """Laedt eine Webseite optional bereinigt und in Chunks aufgeteilt."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="web-load",
        description="Webseite per URL laden, optional bereinigen und chunken.",
    )
    parser.add_argument("url", help="Adresse der Webseite (http:// oder https://)")
    parser.add_argument("--clean", action="store_true", help="Dokumente bereinigen")
    parser.add_argument("--chunk", action="store_true", help="Dokumente in Chunks teilen")
    parser.add_argument(
        "--write", "-write", metavar="DATEI", type=Path,
        help="Ergebnis speichern (.md: Markdown, sonst Text; Chunks nummeriert)",
    )
    args = parser.parse_args(argv)

    try:
        documents = load_web_page(args.url, on_event=print)
    except (ValueError, RuntimeError) as exc:
        print(f"Fehler: {exc}")
        return 2
    if not documents:
        print("Keine Dokumente gefunden.")
        return 1

    if args.clean:
        documents, cleaning_stats = clean_web_documents(documents)
        print(cleaning_stats.summary())

    if args.chunk:
        documents, chunking_stats = chunk_web_documents(documents)
        print(chunking_stats.summary())
        print("Chunking")
        for document in documents:
            print(f"Chunk {document.metadata.get('chunk_index')}: {document.page_content}")
    else:
        print(f"documents={len(documents)}")
        for document in documents:
            print(document.page_content)

    if args.write is not None:
        from rag_project.pipeline.csv_load import write_documents

        write_documents(documents, args.write.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["URL_PREFIXES", "load_web_page", "clean_web_documents", "chunk_web_documents", "main"]
