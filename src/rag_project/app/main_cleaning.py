"""CLI-Anwendung fuer die Textbereinigung.

Getrennt von :mod:`rag_project.cleaning` -- dort liegt nur die Bibliothek,
hier nur die Bedienung (``argparse``, Ausgabe, Dateischreiben, Exit-Codes).

Die Pfadaufloesung liegt in :mod:`rag_project.paths`: ohne Argument werden die
Pfade aus der ``.env`` gelesen (``DOCS_DIR``, ``DEFAULT_DOCUMENT``,
``DOCUMENT_FILES``), sonst der Fallback ``docs/handbuch.txt``.

Aufruf:
    uv run python -m rag_project.app.main_cleaning --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from rag_project.pipeline.cleaning import (
    clean_documents,
    format_cleaning_report,
    format_cleaning_stats,
    join_cleaned_text,
)
from rag_project.pipeline.loading import UnsupportedFormatError, load_documents
from rag_project.core.paths import (
    DocumentPathError,
    describe_paths,
    resolve_documents,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-project.main_cleaning",
        description="Text bereinigen (Steuerzeichen, Leerraum, Silbentrennung).",
    )
    parser.add_argument(
        "paths", nargs="*",
        help="Quelldateien. Ohne Angabe: DOCS_DIR/DEFAULT_DOCUMENT aus der .env",
    )
    parser.add_argument(
        "--no-unicode", action="store_true",
        help="Unicode-Normalisierung (NFKC) abschalten",
    )
    parser.add_argument(
        "--no-hyphens", action="store_true",
        help="Silbentrennung am Zeilenende nicht aufloesen",
    )
    parser.add_argument(
        "--no-bullets", action="store_true",
        help="Aufzaehlungszeichen nicht vereinheitlichen",
    )
    parser.add_argument(
        "--keep-empty", action="store_true",
        help="Leere Dokumente behalten statt verwerfen",
    )
    parser.add_argument(
        "--preview", type=int, default=300,
        help="Zeichen der Vorschau (0 = keine)",
    )
    parser.add_argument(
        "--show-paths", action="store_true",
        help="Aufgeloeste Pfade anzeigen, bevor geladen wird",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="Fortschritt je Dokument anzeigen (nach stderr)",
    )
    parser.add_argument(
        "--steps", action="store_true",
        help="Zeigen, welcher Bereinigungsschritt wie viel bewirkt hat",
    )
    parser.add_argument(
        "--write", metavar="DATEI",
        help="Bereinigten Text in dieser Datei speichern",
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

    try:
        documents = load_documents(paths)
    except UnsupportedFormatError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    options: dict[str, object] = {}
    if args.no_unicode:
        options["normalize_unicode_text"] = False
    if args.no_hyphens:
        options["join_hyphens"] = False
    if args.no_bullets:
        options["unify_bullets"] = False

    # Fortschritt geht nach stderr, damit stdout sauber bleibt.
    hook = (lambda line: print(line, file=sys.stderr)) if args.progress else None

    cleaned, stats = clean_documents(
        documents,
        drop_empty=not args.keep_empty,
        on_event=hook,
        **options,
    )

    print(format_cleaning_stats(stats))

    if args.steps and documents:
        print()
        print(format_cleaning_report(
            documents[0].page_content,
            cleaned[0].page_content if cleaned else "",
        ))

    if args.preview > 0 and cleaned:
        first = cleaned[0]
        print("\nVorschau (erstes Dokument):")
        print("-" * 62)
        print(first.page_content[: args.preview].strip())
        print("-" * 62)
        print(
            f"Metadaten: chars_before={first.metadata.get('chars_before')}, "
            f"chars_after={first.metadata.get('chars_after')}"
        )

    if args.write:
        target = Path(args.write)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(join_cleaned_text(cleaned), encoding="utf-8")
        print(f"\nGeschrieben: {target} ({target.stat().st_size} Bytes)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
