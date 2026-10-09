"""Wikipedia-Artikel laden."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import requests
from langchain_core.documents import Document

WIKIPEDIA_PREFIX = "wikipedia:"

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


def _user_agent() -> str:
    return os.getenv("WIKIPEDIA_USER_AGENT", "rag-project/0.1.0 (Wikipedia loader)")


def _wikipedia_cache_path(topic: str, language: str) -> Path:
    """Pfad zur lokalen Wikipedia-Cache-Datei fuer den Suchbegriff."""
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in topic)
    return Path.cwd() / ".cache" / "wikipedia" / language / f"{slug}.json"


def _write_cache(topic: str, language: str, documents: list[Document]) -> None:
    cache_path = _wikipedia_cache_path(topic, language)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in documents
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _load_wikipedia_from_cache(topic: str, language: str) -> list[Document] | None:
    """Laedt ein bereits gecachtes Wikipedia-Ergebnis, falls vorhanden."""
    cache_path = _wikipedia_cache_path(topic, language)
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    documents = [
        Document(
            page_content=item["page_content"],
            metadata=item.get("metadata", {}),
        )
        for item in payload
    ]
    return documents or None


def load_wikipedia_direct(
    topic: str,
    *,
    language: str = "de",
    max_docs: int = 1,
) -> list[Document]:
    """Laedt Wikipedia-Artikel direkt ueber die API als robusten Fallback.

    Wiederholt bei HTTP 429 und Netzfehlern mit Backoff und cached Erfolge.
    """
    cached = _load_wikipedia_from_cache(topic, language)
    if cached is not None:
        return cached

    headers = {"User-Agent": _user_agent()}
    base = f"https://{language}.wikipedia.org/w/api.php"

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            search_response = requests.get(
                base,
                params={
                    "action": "query",
                    "list": "search",
                    "format": "json",
                    "srsearch": topic,
                    "srlimit": max_docs,
                    "srnamespace": 0,
                    "utf8": 1,
                },
                headers=headers,
                timeout=30,
            )
            search_response.raise_for_status()
            titles = [
                hit["title"]
                for hit in search_response.json().get("query", {}).get("search", [])
            ]
            if not titles:
                titles = [topic]

            extract_response = requests.get(
                base,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "format": "json",
                    "redirects": 1,
                    "titles": "|".join(quote(title, safe="") for title in titles),
                    "utf8": 1,
                },
                headers=headers,
                timeout=30,
            )
            extract_response.raise_for_status()
            pages = extract_response.json().get("query", {}).get("pages", {})

            documents: list[Document] = []
            for index, page in enumerate(pages.values()):
                if page.get("missing"):
                    continue
                content = (page.get("extract") or "").strip()
                if not content:
                    continue
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": f"wikipedia:{topic}",
                            "file_name": f"wikipedia:{topic}",
                            "title": page.get("title", topic),
                            "format": "wiki",
                            "page_index": index,
                            "size_bytes": 0,
                        },
                    )
                )

            if not documents:
                raise ValueError(f"Keine Wikipedia-Ergebnisse fuer '{topic}' gefunden.")

            _write_cache(topic, language, documents)
            return documents
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            last_error = exc
            if status == 429 and attempt < 2:
                time.sleep(2**attempt)
                continue
            raise
        except (requests.exceptions.RequestException, ValueError, TypeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Wikipedia-Ladung fuer '{topic}' fehlgeschlagen.")


def load_wikipedia(
    topic: str,
    *,
    language: str = "de",
    query: str | None = None,
    max_docs: int = 1,
    on_event: EventHook | None = None,
) -> list[Document]:
    """Laedt Artikel aus Wikipedia (LangChain-Loader, Fallback: direkte API)."""
    (WikipediaLoader,) = _import_or_die(
        "langchain_community.document_loaders", ("WikipediaLoader",), "wikipedia"
    )
    import wikipedia

    wikipedia.set_user_agent(_user_agent())
    if on_event is not None:
        on_event(f"[wikipedia] {topic} (Sprache {language}, max {max_docs})")
    kwargs: dict[str, object] = {
        "lang": language,
        "load_max_docs": max_docs,
        "query": query or topic,
    }
    try:
        documents = WikipediaLoader(**kwargs).load()
        _write_cache(topic, language, documents)
        return documents
    except Exception as exc:
        if isinstance(exc, (requests.exceptions.JSONDecodeError, ValueError)):
            return load_wikipedia_direct(topic, language=language, max_docs=max_docs)
        raise


def clean_wikipedia_documents(
    documents: list[Document],
    *,
    drop_empty: bool = True,
    **options: object,
) -> tuple[list[Document], object]:
    """Bereinigt bereits geladene Wikipedia-Dokumente."""
    from rag_project.pipeline.cleaning import clean_documents

    return clean_documents(documents, drop_empty=drop_empty, **options)


def chunk_wikipedia_documents(
    documents: list[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> tuple[list[Document], object]:
    """Zerlegt bereits geladene Wikipedia-Dokumente in Chunks."""
    from rag_project.pipeline.chunking import split_documents

    return split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def main(argv: list[str] | None = None) -> int:
    """Laedt Wikipedia-Artikel optional bereinigt und in Chunks aufgeteilt."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="wikipedia-load",
        description="Wikipedia-Artikel laden, optional bereinigen und chunken.",
    )
    parser.add_argument("topic", help="Suchbegriff / Artikelthema")
    parser.add_argument("--lang", default="de", help="Sprachversion (Standard: de)")
    parser.add_argument(
        "--max-docs", type=int, default=1, help="Zahl der Artikel (Standard: 1)"
    )
    parser.add_argument("--clean", action="store_true", help="Dokumente bereinigen")
    parser.add_argument("--chunk", action="store_true", help="Dokumente in Chunks teilen")
    parser.add_argument(
        "--write", "-write", metavar="DATEI", type=Path,
        help="Ergebnis speichern (.md: Markdown, sonst Text; Chunks nummeriert)",
    )
    args = parser.parse_args(argv)

    documents = load_wikipedia(
        args.topic, language=args.lang, max_docs=args.max_docs, on_event=print
    )
    if not documents:
        print("Keine Dokumente gefunden.")
        return 1

    if args.clean:
        documents, cleaning_stats = clean_wikipedia_documents(documents)
        print(cleaning_stats.summary())

    if args.chunk:
        documents, chunking_stats = chunk_wikipedia_documents(documents)
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


__all__ = [
    "WIKIPEDIA_PREFIX",
    "load_wikipedia",
    "load_wikipedia_direct",
    "main",
]
