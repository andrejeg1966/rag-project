"""CLI-Anwendung fuer das Dokumentenladen.

Getrennt von :mod:`rag_project.loading` -- dort liegt nur die Bibliothek,
hier nur die Bedienung (``argparse``, Ausgabe, Exit-Codes).

Die Pfadaufloesung liegt in :mod:`rag_project.paths`: ohne Argument werden die
Pfade aus der ``.env`` gelesen (``DOCS_DIR``, ``DEFAULT_DOCUMENT``,
``DOCUMENT_FILES``), sonst der Fallback ``docs/handbuch.txt``.

Neben Dateien nimmt die Anwendung netzbasierte Quellen an. Sie werden am
Praefix erkannt und *nicht* ueber :func:`rag_project.paths.resolve_documents`
aufgeloest -- eine URL ist kein Pfad und darf nicht gegen das Arbeits- oder
Dokumentenverzeichnis geprueft werden:

    src:https://example.com/handbuch    Webseite als Text
    src:wikipedia:RAG                   Wikipedia-Artikel
    src:csv:https://example.com/d.csv   entfernte CSV-Datei

Das Praefix ``src:`` ist noetig, damit die Anwendung Quellen von Pfaden
unterscheiden kann. Ohne Praefix gilt ein Argument als Dateipfad.

Aufruf:
    uv run python -m rag_project.main_loading --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from rag_project.loading import (
    LoaderDependencyError,
    UnsupportedFormatError,
    describe,
    detect_source_kind,
    format_document_metadata,
    format_load_report,
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

#: Praefixe innerhalb der Quelle, die die Bibliothek kennt.
REMOTE_PREFIXES = ("http://", "https://", "wikipedia:", "csv:")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-project.main_loading",
        description="Dokumente laden und als LangChain-Document ausgeben.",
    )
    parser.add_argument(
        "paths", nargs="*",
        help="Quelldateien oder Quellen mit src:-Praefix. Ohne Angabe: "
             "DOCS_DIR/DEFAULT_DOCUMENT aus der .env",
    )
    parser.add_argument(
        "--preview", type=int, default=200,
        help="Zeichen der Vorschau pro Datei (0 = keine Vorschau)",
    )
    parser.add_argument(
        "--encoding", default="utf-8", help="Kodierung fuer Text- und CSV-Dateien"
    )
    parser.add_argument(
        "--show-paths", action="store_true",
        help="Aufgeloeste Pfade anzeigen, bevor geladen wird",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help="Fortschritt je Quelle anzeigen (nach stderr)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Nur die Zusammenfassung, keine Vorschau und keine Metadaten",
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

    # --- Zwischenablage -----------------------------------------------------
    parser.add_argument(
        "--write-jsonl", metavar="DATEI",
        help="geladene Dokumente als JSONL speichern (reproduzierbarer Index)",
    )
    return parser


def _split_sources(arguments: Sequence[str]) -> tuple[list[str], list[str]]:
    """Trennt Argumente in lokale Pfade und netzbasierte Quellen.

    Ein Argument mit ``src:``-Praefix ist eine Quelle; alles andere ist ein
    Pfad. Die Rueckgabe sind zwei Listen, damit der Pfadaufloeser nur das
    bekommt, was er auch pruefen kann.
    """
    local: list[str] = []
    remote: list[str] = []

    for argument in arguments:
        if argument.startswith(SOURCE_PREFIX):
            value = argument[len(SOURCE_PREFIX):].strip()
            # Ein zweites Praefix ist erlaubt, damit beide Schreibweisen
            # funktionieren: src:https://... und src:wikipedia:...
            remote.append(value)
        else:
            local.append(argument)

    return local, remote


def _assert_remote_enabled(remote: Sequence[str], allowed: bool) -> bool:
    """Bestaetigt netzbasierte Quellen, wenn kein Schalter gesetzt ist.

    Ein Lauf, der ungefragt das Netz anfasst, ist bei RAG unerwartet: Die
    Quelle ist nicht reproduzierbar, und der Aufruf kann lange dauern. Die
    Rueckfrage ist die Voreinstellung; ``--allow-remote`` ueberspringt sie.
    """
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

    local_arguments, remote_sources = _split_sources(args.paths)

    # Optionen wie --url ergaenzen die Quellen aus den Positionsargumenten.
    remote_sources.extend(args.url)
    remote_sources.extend(f"wikipedia:{topic}" for topic in args.wikipedia)
    remote_sources.extend(f"csv:{url}" for url in args.csv_url)

    if not _assert_remote_enabled(remote_sources, args.allow_remote):
        return 1

    # --- lokale Pfade aufloesen ---------------------------------------------
    paths = []
    if local_arguments or not remote_sources:
        # Ohne Positionsargumente und ohne Quelle greifen die Settings.
        try:
            paths = resolve_documents(local_arguments or None)
        except DocumentPathError as exc:
            print(f"Fehler: {exc}", file=sys.stderr)
            return 1

    if args.show_paths and paths:
        print(describe_paths(paths))
        print()

    # Fortschritt geht nach stderr, damit stdout sauber bleibt.
    hook = (lambda line: print(line, file=sys.stderr)) if args.progress else None

    # --- laden ---------------------------------------------------------------
    sources: list[str] = [str(path) for path in paths] + list(remote_sources)

    try:
        documents = load_documents(
            sources,
            encoding=args.encoding,
            on_event=hook,
            wikipedia_language=args.wiki_lang,
            wikipedia_max_docs=args.wiki_max_docs,
        )
    except UnsupportedFormatError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    except LoaderDependencyError as exc:
        # Fehlendes Paket -- eine andere Ursache als ein unbekanntes Format
        # und deshalb auch eine eigene Meldung.
        print(f"Fehlende Abhaengigkeit: {exc}", file=sys.stderr)
        return 2

    print(format_load_report(describe(documents)))

    if args.write_jsonl:
        target = Path(args.write_jsonl)
        written = to_jsonl(documents, target)
        print(
            f"\nGeschrieben: {written} "
            f"({written.stat().st_size} Bytes, {len(documents)} Dokument(e))"
        )

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
