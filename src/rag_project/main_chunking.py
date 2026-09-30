"""CLI-Anwendung fuer das Chunking.

Getrennt von :mod:`rag_project.chunking` -- dort liegt nur die Bibliothek,
hier nur die Bedienung (``argparse``, Ausgabe, Dateischreiben, Exit-Codes).

Die Pfadaufloesung liegt in :mod:`rag_project.paths`: ohne Argument werden die
Pfade aus der ``.env`` gelesen (``DOCS_DIR``, ``DEFAULT_DOCUMENT``,
``DOCUMENT_FILES``), sonst der Fallback ``docs/handbuch.txt``.

Faehrt die vollstaendige Vorverarbeitung: laden -> bereinigen -> chunken.

Aufruf:
    uv run python -m rag_project.main_chunking --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from rag_project.chunking import (
    HANDBOOK_SEPARATORS,
    STRATEGIES,
    chunks_to_markdown,
    compare_strategies,
    format_chunk_preview,
    format_chunking_stats,
    format_strategy_comparison,
    split_documents,
)
from rag_project.cleaning import clean_documents, format_cleaning_stats
from rag_project.loading import UnsupportedFormatError, load_documents
from rag_project.paths import (
    DocumentPathError,
    describe_paths,
    resolve_documents,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-project.main_chunking",
        description="Dokumente chunken und Kennzahlen ausgeben.",
    )
    parser.add_argument(
        "paths", nargs="*",
        help="Quelldateien. Ohne Angabe: DOCS_DIR/DEFAULT_DOCUMENT aus der .env",
    )
    parser.add_argument(
        "--strategy", default="recursive", choices=STRATEGIES,
        help="Splitter-Strategie (Standard: recursive)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=800, help="Zielgroesse (Standard: 800)"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=100, help="Ueberlappung (Standard: 100)"
    )
    parser.add_argument(
        "--handbook", action="store_true",
        help="Trennzeichen-Reihenfolge fuer Handbuecher verwenden",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Alle Strategien bei gleichem Zielwert vergleichen",
    )
    parser.add_argument(
        "--show", type=int, default=3, help="Erste N Chunks anzeigen (0 = keine)"
    )
    parser.add_argument(
        "--show-paths", action="store_true",
        help="Aufgeloeste Pfade anzeigen, bevor geladen wird",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="Fortschritt je Stufe anzeigen (nach stderr)",
    )
    parser.add_argument(
        "--write", metavar="DATEI", help="Chunks als Textdatei speichern"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        paths = resolve_documents(args.paths)
    except DocumentPathError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    if args.show_paths:
        print(describe_paths(paths))
        print()

    # Fortschritt geht nach stderr, damit stdout sauber bleibt.
    hook = (lambda line: print(line, file=sys.stderr)) if args.progress else None

    try:
        raw_documents = load_documents(paths, on_event=hook)
    except UnsupportedFormatError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    documents, cleaning_stats = clean_documents(raw_documents, on_event=hook)

    print("Pipeline")
    print("=" * 62)
    print(f"Geladen    : {len(raw_documents)} Dokument(e)")
    print(f"Bereinigt  : {cleaning_stats.summary()}")
    print()

    separators = HANDBOOK_SEPARATORS if args.handbook else None

    try:
        if args.compare:
            results = compare_strategies(
                documents,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                on_event=hook,
            )
            print(format_strategy_comparison(results))
            print()
            return 0

        chunks, stats = split_documents(
            documents,
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            separators=separators,
            on_event=hook,
        )
    except ValueError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    print(format_chunking_stats(stats))

    if args.show > 0 and chunks:
        print()
        print(format_chunk_preview(chunks, limit=args.show))

    if args.write:
        target = Path(args.write)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(chunks_to_markdown(chunks), encoding="utf-8")
        print(f"\nGeschrieben: {target} ({target.stat().st_size} Bytes)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
