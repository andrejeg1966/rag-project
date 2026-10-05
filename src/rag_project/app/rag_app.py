"""CLI fuer die RAG-Anwendung -- Stufen aus dem Projekt, Kette in rag_lib.

Der Ablauf ist der des Projekts:

    laden      rag_project.pipeline.loading   load_documents
    bereinigen rag_project.pipeline.cleaning  clean_documents
    chunken    rag_project.pipeline.chunking  split_documents
    indexieren rag_lib.build_vectorstore
    antworten  rag_lib.build_rag_chain

Fortschritt und Fehler gehen nach stderr, Ergebnisse nach stdout.

Sprache: ``--language de|en``, Standard Deutsch.

Aufrufe:
    python rag_app.py                              Uebersicht und Beispiele
    python rag_app.py --config                     aktive Einstellungen
    python rag_app.py --queries                    die Testfragen zeigen
    python rag_app.py --load                       Stufe 1: Ladebericht
    python rag_app.py --clean                      Stufe 2: Reinigungsbilanz
    python rag_app.py --chunks                     Stufe 3: Chunking-Kennzahlen
    python rag_app.py --preview                    erste Chunks mit Metadaten
    python rag_app.py --write DATEI.md             Chunks als Markdown ablegen
    python rag_app.py --retrieve "Frage"           Treffer ohne LLM-Aufruf
    python rag_app.py --ask "Frage" [--show-hits]  eine Frage beantworten
    python rag_app.py --compare                    beide Suchmodi, alle Testfragen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from langchain_qdrant import RetrievalMode

from rag_project.app import rag_lib
from rag_project.app.rag_lib import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_K,
    DEFAULT_LANGUAGE,
    EMBEDDING_MODELS,
    GENERATION_MODEL,
    LANGUAGES,
    LANGUAGE_LABELS,
    OPENROUTER_BASE_URL,
    PROJECT_ROOT,
    REJECTED_EMBEDDING_MODELS,
    SOURCE_FILES,
    SPARSE_MODEL,
    IndexBuildError,
    MissingCredentialsError,
    RagError,
)

SEPARATOR = "=" * 68
COLUMN = 62


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def emit(line: str = "") -> None:
    """Eine Ergebniszeile nach stdout."""
    print(line)


def progress(line: str) -> None:
    """Eine Fortschrittszeile nach stderr -- stdout bleibt sauber."""
    print(line, file=sys.stderr)


def header(title: str) -> None:
    emit(SEPARATOR)
    emit(title)
    emit(SEPARATOR)


# ---------------------------------------------------------------------------
# Ansichten
# ---------------------------------------------------------------------------

def show_config(args, language: str) -> None:
    """Aktive Einstellungen -- ohne Klartext-Keys."""
    header("Konfiguration")
    emit(f"Projektwurzel  : {PROJECT_ROOT}")
    emit(f"Sprache        : {LANGUAGE_LABELS[language]} ({language})"
         f"  [Standard: {DEFAULT_LANGUAGE}]")
    for key in LANGUAGES:
        emit(f"  {LANGUAGE_LABELS[key]:<8}: docs/{SOURCE_FILES[key]['pdf']}"
             f"  |  docs/{SOURCE_FILES[key]['txt']}")
    emit(f"Quellformat    : {args.format}")
    emit()
    emit(f"Stufen (aus dem Projekt)")
    emit(f"  laden        : rag_project.pipeline.loading.load_documents")
    emit(f"  bereinigen   : rag_project.pipeline.cleaning.clean_documents")
    emit(f"  chunken      : rag_project.pipeline.chunking.split_documents")
    emit()
    emit(f"Index          : Qdrant, location=':memory:'")
    emit(f"Embedding      : {args.embedding_model}")
    emit(f"  weitere      : {', '.join(EMBEDDING_MODELS)}")
    emit(f"  abgelehnt    : {', '.join(REJECTED_EMBEDDING_MODELS)} (Chat-Modell)")
    emit(f"Sparse-Modell  : {SPARSE_MODEL}")
    emit(f"LLM            : {GENERATION_MODEL} (reasoning aus)")
    emit(f"Base-URL       : {OPENROUTER_BASE_URL}")
    emit(f"chunk_size     : {args.chunk_size}  chunk_overlap: {args.chunk_overlap}")
    emit(f"Strategie      : {args.strategy}")
    emit(f"Trefferzahl k  : {args.k}")
    emit()


def show_queries(language: str) -> None:
    """Die Testfragen einer Sprache, nummeriert."""
    header(f"Testfragen ({LANGUAGE_LABELS[language]})")
    questions = rag_lib.TEST_QUERIES[language]
    for number, question in enumerate(questions, start=1):
        emit(f"  {number}. {question}")
    emit()
    emit(f"Die ersten {len(questions) - 1} sind aus dem Handbuch beantwortbar;")
    emit("die letzte liegt bewusst ausserhalb des Korpus.")
    emit()


def show_load(pipeline, language: str) -> None:
    """Stufe 1: der Ladebericht aus dem loading-Modul."""
    header(f"Stufe 1 -- laden ({LANGUAGE_LABELS[language]})")
    emit(pipeline.load_report)
    emit()
    first = pipeline.loaded[0]
    emit("Erste Seite, Metadaten:")
    for key, value in sorted(first.metadata.items()):
        emit(f"  {key:<12}: {value}")
    emit(f"\nSeiten geladen : {len(pipeline.loaded)}")
    emit(f"Zeichen        : {sum(len(d.page_content) for d in pipeline.loaded)}")
    emit()


def show_clean(pipeline, language: str) -> None:
    """Stufe 2: die Reinigungsbilanz aus dem cleaning-Modul."""
    header(f"Stufe 2 -- bereinigen ({LANGUAGE_LABELS[language]})")
    emit(pipeline.cleaning_report)
    emit()
    emit("Metadaten der ersten Seite nach der Bereinigung:")
    for key, value in sorted(pipeline.cleaned[0].metadata.items()):
        emit(f"  {key:<12}: {value}")
    emit()


def show_chunks(pipeline, language: str) -> None:
    """Stufe 3: die Chunking-Kennzahlen aus dem chunking-Modul."""
    stats = pipeline.chunking_stats
    header(f"Stufe 3 -- chunken ({LANGUAGE_LABELS[language]})")
    emit(f"{'Quelldokumente':<18}: {stats.source_documents}")
    emit(f"{'Chunks':<18}: {stats.chunks}")
    emit(f"{'Zeichen gesamt':<18}: {stats.total_chars}")
    emit(f"{'je Chunk':<18}: min {stats.min_chars} / median "
         f"{stats.median_chars:.0f} / mean {stats.mean_chars:.0f} / max "
         f"{stats.max_chars}")
    emit(f"{'uebergross':<18}: {stats.oversized}")
    emit(f"{'Strategie':<18}: {stats.strategy} (chunk_size "
         f"{stats.chunk_size}, overlap {stats.overlap})")
    emit()


def show_preview(pipeline) -> None:
    """Die ersten Chunks mit ihrer Herkunft."""
    limit = 5
    chunks = pipeline.chunks[:limit]
    emit(f"Erste {len(chunks)} von {len(pipeline.chunks)} Chunks")
    emit("-" * COLUMN)
    for chunk in chunks:
        meta = chunk.metadata
        emit(f"[{meta.get('chunk_index')}] Kapitel {meta.get('chapter')}, "
             f"Abschnitt {meta.get('section')} -- {len(chunk.page_content)} Zeichen")
        emit(f"Titel: {meta.get('title')}")
        body = " ".join(chunk.page_content.split())[:200]
        emit(f"  {body}...")
        emit()


def show_hits(system, question: str) -> None:
    """Treffer zu einer Frage -- Rang, Kapitel, Auszug."""
    hits = system.retrieve(question)
    emit(f"Treffer ({system.mode.value}): {len(hits)}")
    for rank, chunk in enumerate(hits, start=1):
        meta = chunk.metadata
        excerpt = " ".join(chunk.page_content.split())[:150]
        emit(f"  {rank}. Kapitel {meta.get('chapter')}, "
             f"Abschnitt {meta.get('section')} -- {meta.get('title')}")
        emit(f"     {excerpt}...")
    emit()


def show_answer(system, question: str) -> str:
    """Eine Frage beantworten."""
    emit(f"Suchmodus: {system.label}")
    emit(f"Frage    : {question}")
    emit()
    answer = system.answer(question)
    rag_lib.print_wrapped(answer, width=84)
    emit()
    if system.is_honest(answer):
        emit("-> Grenze des Korpus benannt.")
        emit()
    return answer


def show_comparison(results: list[dict], language: str) -> None:
    """Beide Suchmodi je Frage nebeneinander -- ohne Bewertung."""
    header(f"Vergleich der Suchmodi -- {LANGUAGE_LABELS[language]}")
    for entry in results:
        emit(f"Frage: {entry['question']}")
        emit("-" * COLUMN)
        for mode in entry["modes"]:
            emit(f"[{mode['label']}]")
            emit(f"  Kapitel : {mode['chapters']}")
            emit("  Antwort :")
            rag_lib.print_wrapped(mode["answer"], width=80, indent=4)
            emit()
        emit()

    emit(SEPARATOR)
    emit("Uebersicht")
    emit(SEPARATOR)
    emit(f"{'Modus':<30}{'Treffer o.':>12}{'Grenze':>9}")
    emit("-" * COLUMN)
    for label in sorted({m["label"] for e in results for m in e["modes"]}):
        rows = [m for e in results for m in e["modes"] if m["label"] == label]
        average = sum(m["hits"] for m in rows) / len(rows)
        honest = sum(1 for m in rows if m["honest"])
        emit(f"{label:<30}{average:>12.1f}{honest:>6}/{len(rows)}")
    emit()
    emit("'Grenze' zaehlt Antworten, die den Korpus als Quelle nennen --")
    emit("eine Beobachtung, keine Bewertung der Antworten.")
    emit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_app",
        description="RAG auf dem Benutzerhandbuch: laden, bereinigen, chunken, "
                    "indexieren, antworten.",
    )

    action = parser.add_mutually_exclusive_group()
    action.add_argument("--config", action="store_true",
                        help="aktive Einstellungen anzeigen")
    action.add_argument("--queries", action="store_true",
                        help="die Testfragen anzeigen")
    action.add_argument("--load", action="store_true",
                        help="Stufe 1: Ladebericht zeigen")
    action.add_argument("--clean", action="store_true",
                        help="Stufe 2: Reinigungsbilanz zeigen")
    action.add_argument("--chunks", action="store_true",
                        help="Stufe 3: Chunking-Kennzahlen zeigen")
    action.add_argument("--preview", action="store_true",
                        help="die ersten Chunks mit Metadaten zeigen")
    action.add_argument("--write", metavar="DATEI",
                        help="Chunks als Markdown ablegen und beenden")
    action.add_argument("--retrieve", metavar="FRAGE",
                        help="Treffer zeigen, ohne das LLM zu rufen")
    action.add_argument("--ask", metavar="FRAGE",
                        help="eine Frage beantworten")
    action.add_argument("--compare", action="store_true",
                        help="alle Testfragen in beiden Suchmodi beantworten")

    parser.add_argument("--language", choices=list(LANGUAGES),
                        default=DEFAULT_LANGUAGE,
                        help="Sprache der Wissensbasis (Standard: %(default)s)")
    parser.add_argument("--format", choices=["pdf", "txt"], default="pdf",
                        help="Quellformat des Handbuchs (Standard: %(default)s)")
    parser.add_argument("--mode", choices=[m.value for m in RetrievalMode],
                        default="dense",
                        help="Suchmodus fuer --retrieve und --ask (Standard: %(default)s)")
    parser.add_argument("--embedding-model", choices=EMBEDDING_MODELS,
                        default=EMBEDDING_MODELS[0],
                        help="Embedding-Modell (Standard: %(default)s)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Zielgroesse eines Chunks (Standard: %(default)s)")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help="Ueberlappung (Standard: %(default)s)")
    parser.add_argument("--strategy", default="recursive",
                        choices=["recursive", "character", "token"],
                        help="Splitter-Strategie (Standard: %(default)s)")
    parser.add_argument("--no-handbook", action="store_true",
                        help="die Standard-Trennzeichen statt der Handbuch-"
                             "Reihenfolge verwenden")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Trefferzahl (Standard: %(default)s)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM-Temperatur (Standard: %(default)s)")
    parser.add_argument("--show-hits", action="store_true",
                        help="bei --ask die Treffer vor der Antwort zeigen")
    parser.add_argument("--no-progress", action="store_true",
                        help="die Fortschrittszeilen der Projektstufen unterdruecken")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    language = rag_lib.check_language(args.language)
    hook = None if args.no_progress else progress

    if args.config:
        show_config(args, language)
        return 0
    if args.queries and not args.ask and not args.retrieve:
        show_queries(language)
        return 0

    # Die drei Projektstufen -- laden, bereinigen, chunken.
    try:
        pipeline = rag_lib.run_pipeline(
            language=language,
            fmt=args.format,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            strategy=args.strategy,
            handbook=not args.no_handbook,
            on_event=hook,
        )
    except rag_lib.DocumentPathError as exc:
        progress(f"Fehler: {exc}")
        return 1
    except rag_lib.UnsupportedFormatError as exc:
        progress(f"Fehler: {exc}")
        return 1
    except rag_lib.LoaderDependencyError as exc:
        progress(f"Fehlende Abhaengigkeit: {exc}")
        return 2
    except ValueError as exc:
        progress(f"Fehler: {exc}")
        return 1

    if args.load:
        show_load(pipeline, language)
        return 0
    if args.clean:
        show_clean(pipeline, language)
        return 0
    if args.chunks:
        show_chunks(pipeline, language)
        return 0
    if args.preview:
        header(f"Chunks -- {LANGUAGE_LABELS[language]}")
        show_preview(pipeline)
        return 0
    if args.write:
        from rag_project.pipeline.chunking import chunks_to_markdown

        target = Path(args.write)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(chunks_to_markdown(pipeline.chunks), encoding="utf-8")
        emit(f"{target} ({target.stat().st_size} Bytes, "
             f"{len(pipeline.chunks)} Chunks)")
        return 0

    # Ab hier braucht es Embeddings -- also einen Key und das Netz.
    try:
        key = rag_lib.openrouter_api_key()

        if args.compare:
            results: list[dict] = []
            systems = {}
            for mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID):
                progress(f"[index] {LANGUAGE_LABELS[language]} {mode.value}")
                systems[mode] = rag_lib.create_rag_system(
                    language=language,
                    mode=mode,
                    embedding_model=args.embedding_model,
                    pipeline=pipeline,
                    api_key=key,
                    k=args.k,
                )

            for question in rag_lib.iter_test_queries(language):
                progress(f"[frage] {question}")
                entry = {"question": question, "modes": []}
                for system in systems.values():
                    hits = system.retrieve(question)
                    answer = system.answer(question)
                    entry["modes"].append(
                        {
                            "label": system.label,
                            "hits": len(hits),
                            "chapters": ", ".join(
                                f"K{h.metadata.get('chapter')}"
                                f"{'.' + str(h.metadata.get('section')) if h.metadata.get('section') else ''}"
                                for h in hits
                            ),
                            "answer": answer,
                            "honest": system.is_honest(answer),
                        }
                    )
                results.append(entry)

            show_comparison(results, language)
            return 0

        mode = RetrievalMode(args.mode)
        progress(f"[index] {LANGUAGE_LABELS[language]} {mode.value}")
        system = rag_lib.create_rag_system(
            language=language,
            mode=mode,
            embedding_model=args.embedding_model,
            pipeline=pipeline,
            api_key=key,
            k=args.k,
        )

        if args.retrieve:
            header(f"Treffer -- {LANGUAGE_LABELS[language]}")
            emit(f"Frage    : {args.retrieve}")
            emit()
            show_hits(system, args.retrieve)
            return 0

        if args.ask:
            header(f"Frage und Antwort -- {LANGUAGE_LABELS[language]}")
            if args.show_hits:
                emit(f"Frage    : {args.ask}")
                emit()
                show_hits(system, args.ask)
            show_answer(system, args.ask)
            return 0

    except MissingCredentialsError as exc:
        progress(f"Fehler: {exc}")
        return 1
    except IndexBuildError as exc:
        progress(f"Fehler: {exc}")
        return 2
    except RagError as exc:
        progress(f"Fehler: {exc}")
        return 1

    # Kein Aktionsschalter: die Pipeline zeigen.
    show_load(pipeline, language)
    show_clean(pipeline, language)
    show_chunks(pipeline, language)
    show_preview(pipeline)
    emit("Beispielaufrufe:")
    emit('  python rag_app.py --retrieve "Wie lange gilt der Link zum Zurücksetzen?"')
    emit('  python rag_app.py --ask "Welche Dateigröße ist erlaubt?" --show-hits')
    emit("  python rag_app.py --compare")
    emit("  python rag_app.py --language en --compare")
    emit('  python rag_app.py --retrieve "..." --mode hybrid')
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
