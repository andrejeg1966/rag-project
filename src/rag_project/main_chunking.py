"""CLI-Anwendung fuer das Chunking.

Getrennt von :mod:`rag_project.chunking` -- dort liegt nur die Bibliothek,
hier nur die Bedienung (``argparse``, Ausgabe, Dateischreiben, Exit-Codes).

Die Pfadaufloesung liegt in :mod:`rag_project.paths`: ohne Argument werden die
Pfade aus der ``.env`` gelesen (``DOCS_DIR``, ``DEFAULT_DOCUMENT``,
``DOCUMENT_FILES``), sonst der Fallback ``docs/handbuch.txt``.

Faehrt die vollstaendige Vorverarbeitung: laden -> bereinigen -> chunken.

Neben Dateien nimmt die Anwendung netzbasierte Quellen an. Sie werden am
Praefix ``src:`` erkannt und nicht ueber :mod:`rag_project.paths` aufgeloest
-- eine URL ist kein Pfad:

    src:https://example.com/handbuch    Webseite als Text
    src:wikipedia:RAG                   Wikipedia-Artikel
    src:csv:https://example.com/d.csv   entfernte CSV-Datei

Weil netzbasierte Inhalte nicht reproduzierbar sind, empfiehlt sich der
Schalter ``--load-jsonl``: Er legt den Ladesatz ab, bevor bereinigt und
gechunkt wird. Ein zweiter Lauf kann denselben Satz dann wiederverwenden.

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
from rag_project.loading import (
    LoaderDependencyError,
    UnsupportedFormatError,
    from_jsonl,
    load_documents,
    to_jsonl,
)
from rag_project.paths import (
    DocumentPathError,
    describe_paths,
    resolve_documents,
)

#: Praefix, das ein Argument als netzbasierte Quelle kennzeichnet.
SOURCE_PREFIX = "src:"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-project.main_chunking",
        description="Dokumente chunken und Kennzahlen ausgeben.",
    )
    parser.add_argument(
        "paths", nargs="*",
        help="Quelldateien oder Quellen mit src:-Praefix. Ohne Angabe: "
             "DOCS_DIR/DEFAULT_DOCUMENT aus der .env",
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

    # --- netzbasierte Quellen ------------------------------------------------
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
        "--wiki-lang", default="de", help="Sprachversion fuer Wikipedia (Standard: de)"
    )
    remote.add_argument(
        "--wiki-max-docs", type=int, default=1,
        help="Zahl der Wikipedia-Artikel je Thema (Standard: 1)",
    )
    remote.add_argument(
        "--allow-remote", action="store_true",
        help="netzbasierte Quellen ohne Rueckfrage laden",
    )

    # --- Ladesatz ------------------------------------------------------------
    parser.add_argument(
        "--load-jsonl", metavar="DATEI",
        help="Ladesatz speichern -- Pflicht bei netzbasierten Quellen, wenn "
             "der Lauf reproduzierbar sein soll",
    )
    parser.add_argument(
        "--from-jsonl", metavar="DATEI",
        help="Ladesatz statt Quellen verwenden; ueberspringt Laden und Netz",
    )
    return parser


def _split_sources(arguments: Sequence[str]) -> tuple[list[str], list[str]]:
    """Trennt Argumente in lokale Pfade und netzbasierte Quellen.

    Ein Argument mit ``src:``-Praefix ist eine Quelle; alles andere ist ein
    Pfad. Nur so bekommt der Pfadaufloeser das, was er pruefen kann.
    """
    local: list[str] = []
    remote: list[str] = []

    for argument in arguments:
        if argument.startswith(SOURCE_PREFIX):
            remote.append(argument[len(SOURCE_PREFIX):].strip())
        else:
            local.append(argument)

    return local, remote


def _assert_remote_enabled(remote: Sequence[str], allowed: bool) -> bool:
    """Bestaetigt netzbasierte Quellen, wenn kein Schalter gesetzt ist."""
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


def _collect_remote_sources(args: argparse.Namespace, positional: Sequence[str]) -> list[str]:
    """Sammelt alle netzbasierten Quellen aus Praefixen und Optionen."""
    remote = list(positional)
    remote.extend(args.url)
    remote.extend(f"wikipedia:{topic}" for topic in args.wikipedia)
    remote.extend(f"csv:{url}" for url in args.csv_url)
    return remote


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # --- Ladesatz statt Quellen ---------------------------------------------
    if args.from_jsonl:
        # Ein Ladesatz umgeht Laden und Netz vollstaendig. Quellen anzugeben
        # waere widerspruechlich -- die Datei entscheidet den Inhalt.
        if args.paths or args.url or args.wikipedia or args.csv_url:
            print(
                "Fehler: --from-jsonl und Quellenangaben schliessen sich aus.",
                file=sys.stderr,
            )
            return 1
        try:
            raw_documents = from_jsonl(args.from_jsonl)
        except FileNotFoundError:
            print(f"Fehler: Ladesatz nicht gefunden: {args.from_jsonl}", file=sys.stderr)
            return 1
        except (ValueError, KeyError) as exc:
            print(f"Fehler: Ladesatz nicht lesbar: {exc}", file=sys.stderr)
            return 1

        print("Pipeline (aus Ladesatz)")
        print("=" * 62)
        print(f"Geladen    : {len(raw_documents)} Dokument(e) aus {args.from_jsonl}")
        print()

    else:
        local_arguments, prefixed_remote = _split_sources(args.paths)
        remote_sources = _collect_remote_sources(args, prefixed_remote)

        if not _assert_remote_enabled(remote_sources, args.allow_remote):
            return 1

        paths: list[str] = []
        if local_arguments or not remote_sources:
            try:
                paths = [str(p) for p in resolve_documents(local_arguments or None)]
            except DocumentPathError as exc:
                print(f"Fehler: {exc}", file=sys.stderr)
                return 1

        if args.show_paths and paths:
            print(describe_paths([Path(p) for p in paths]))
            print()

        # Fortschritt geht nach stderr, damit stdout sauber bleibt.
        hook = (lambda line: print(line, file=sys.stderr)) if args.progress else None

        sources = paths + list(remote_sources)
        try:
            raw_documents = load_documents(
                sources,
                on_event=hook,
                wikipedia_language=args.wiki_lang,
                wikipedia_max_docs=args.wiki_max_docs,
            )
        except UnsupportedFormatError as exc:
            print(f"Fehler: {exc}", file=sys.stderr)
            return 1
        except LoaderDependencyError as exc:
            print(f"Fehlende Abhaengigkeit: {exc}", file=sys.stderr)
            return 2

        # Ein netzbasierter Lauf ohne abgelegten Ladesatz ist nicht
        # wiederholbar. Der Hinweis kommt nur, wenn wirklich Netz im Spiel war.
        if remote_sources and not args.load_jsonl:
            print(
                "Hinweis: netzbasierte Quellen ohne --load-jsonl -- dieser Lauf "
                "ist nicht reproduzierbar.",
                file=sys.stderr,
            )

        if args.load_jsonl:
            written = to_jsonl(raw_documents, args.load_jsonl)
            print(
                f"Ladesatz   : {written} "
                f"({written.stat().st_size} Bytes, {len(raw_documents)} Dokument(e))",
                file=sys.stderr,
            )

        print("Pipeline")
        print("=" * 62)
        print(f"Geladen    : {len(raw_documents)} Dokument(e)")

    # --- bereinigen und chunken ---------------------------------------------
    hook = (lambda line: print(line, file=sys.stderr)) if args.progress else None

    documents, cleaning_stats = clean_documents(raw_documents, on_event=hook)
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
