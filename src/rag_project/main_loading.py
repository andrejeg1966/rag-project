"""CLI-Anwendung fuer das Dokumentenladen.

Getrennt von :mod:`rag_project.loading` -- dort liegt nur die Bibliothek,
hier nur die Bedienung (``argparse``, Ausgabe, Exit-Codes).

Die Pfadaufloesung liegt in :mod:`rag_project.paths`: ohne Argument werden die
Pfade aus der ``.env`` gelesen (``DOCS_DIR``, ``DEFAULT_DOCUMENT``,
``DOCUMENT_FILES``), sonst der Fallback ``docs/handbuch.txt``.

Aufruf:
    uv run python -m rag_project.main_loading --help
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from rag_project.loading import (
    UnsupportedFormatError,
    describe,
    format_document_metadata,
    format_load_report,
    load_documents,
)
from rag_project.paths import (
    DocumentPathError,
    describe_paths,
    resolve_documents,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-project.main_loading",
        description="Dokumente laden und als LangChain-Document ausgeben.",
    )
    parser.add_argument(
        "paths", nargs="*",
        help="Quelldateien. Ohne Angabe: DOCS_DIR/DEFAULT_DOCUMENT aus der .env",
    )
    parser.add_argument(
        "--preview", type=int, default=200,
        help="Zeichen der Vorschau pro Datei (0 = keine Vorschau)",
    )
    parser.add_argument(
        "--encoding", default="utf-8", help="Kodierung fuer Textdateien"
    )
    parser.add_argument(
        "--show-paths", action="store_true",
        help="Aufgeloeste Pfade anzeigen, bevor geladen wird",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="Fortschritt je Datei anzeigen (nach stderr)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Nur die Zusammenfassung, keine Vorschau und keine Metadaten",
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
        documents = load_documents(paths, encoding=args.encoding, on_event=hook)
    except UnsupportedFormatError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    print(format_load_report(describe(documents)))

    if args.quiet or args.preview <= 0 or not documents:
        return 0

    first = documents[0]
    print("\nVorschau (erstes Dokument):")
    print("-" * 62)
    print(first.page_content[: args.preview].strip())
    print("-" * 62)
    print(format_document_metadata(first))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
