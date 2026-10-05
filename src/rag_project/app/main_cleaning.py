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

#: Praefix, das ein Argument als netzbasierte Quelle kennzeichnet.
SOURCE_PREFIX = "src:"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-project.main_cleaning",
        description="Text bereinigen (Steuerzeichen, Leerraum, Silbentrennung).",
    )
    parser.add_argument(
        "paths", nargs="*",
        help="Quelldateien oder Quellen mit src:-Praefix. Ohne Angabe: "
             "DOCS_DIR/DEFAULT_DOCUMENT aus der .env",
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

    remote = parser.add_argument_group(
        "netzbasierte Quellen",
        "Optionen fuer src:http..., src:wikipedia:... und src:csv:...",
    )
    remote.add_argument(
        "--url", action="append", default=[], metavar="ADRESSE",
        help="Webseite laden; mehrfach angebbar",
    )
    remote.add_argument(
        "--wikipedia", action="append", default=[], metavar="THEMA",
        help="Wikipedia-Artikel zum Thema laden; mehrfach angebbar",
    )
    remote.add_argument(
        "--csv-url", action="append", default=[], metavar="ADRESSE",
        help="entfernte CSV-Datei laden; mehrfach angebbar",
    )
    remote.add_argument(
        "--wiki-lang", default="de",
        help="Sprachversion fuer Wikipedia (Standard: de)",
    )
    remote.add_argument(
        "--wiki-max-docs", type=int, default=1,
        help="Zahl der Wikipedia-Artikel je Thema (Standard: 1)",
    )
    remote.add_argument(
        "--allow-remote", action="store_true",
        help="netzbasierte Quellen ohne Rueckfrage laden",
    )
    return parser


def _split_sources(arguments: Sequence[str]) -> tuple[list[str], list[str]]:
    local: list[str] = []
    remote: list[str] = []

    for argument in arguments:
        if argument.startswith(SOURCE_PREFIX):
            remote.append(argument[len(SOURCE_PREFIX):].strip())
        else:
            local.append(argument)

    return local, remote


def _assert_remote_enabled(remote: Sequence[str], allowed: bool) -> bool:
    if not remote or allowed:
        return True

    print(
        f"Diese Quelle(n) werden ueber das Netz geladen ({len(remote)}):",
        file=sys.stderr,
    )
    for value in remote:
        print(f"  {value}", file=sys.stderr)
    try:
        answer = input("Fortfahren? [j/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAbgebrochen.", file=sys.stderr)
        return False
    return answer in {"j", "ja", "y", "yes"}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    local_arguments, prefixed_remote = _split_sources(args.paths)
    remote_sources = list(prefixed_remote)
    remote_sources.extend(args.url)
    remote_sources.extend(f"wikipedia:{topic}" for topic in args.wikipedia)
    remote_sources.extend(f"csv:{url}" for url in args.csv_url)

    if not _assert_remote_enabled(remote_sources, args.allow_remote):
        return 1

    paths: list[str] = []
    if local_arguments or not remote_sources:
        try:
            paths = [str(path) for path in resolve_documents(local_arguments or None)]
        except DocumentPathError as exc:
            print(f"Fehler: {exc}", file=sys.stderr)
            return 1

    if args.show_paths and paths:
        print(describe_paths([Path(p) for p in paths]))
        print()

    try:
        documents = load_documents(
            [str(path) for path in paths] + list(remote_sources),
            on_event=None,
            wikipedia_language=args.wiki_lang,
            wikipedia_max_docs=args.wiki_max_docs,
        )
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
