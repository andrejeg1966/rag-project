"""Chunking fuer die RAG-Pipeline.

Reine Bibliothek -- keine CLI, kein ``argparse``. Die Ausfuehrung liegt in
:mod:`rag_project.main_chunking`.

Verantwortung:
    - Dokumente in ueberlappende Textstuecke aufteilen
    - Splitter-Strategien waehlbar machen (Recursive, Character, Token)
    - Metadaten anreichern: Kapitel, Abschnitt, Chunk-Index
    - Qualitaetskennzahlen je Strategie liefern

Ausgabe:
    Die Funktionen geben selbst nichts auf stdout aus. Fortschritt laeuft ueber
    einen ``on_event``-Callback, fertige Ausgabetexte liefern
    :func:`format_chunking_stats`, :func:`format_strategy_comparison` und
    :func:`format_chunk_preview`.

Der Splitter wird auf die *bereinigten* Dokumente aus
:mod:`rag_project.cleaning` angewendet. Die Reihenfolge ist wichtig: erst
bereinigen, dann teilen -- sonst wandern Steuerzeichen in die Embeddings.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

#: Trennzeichen-Reihenfolge fuer den Recursive-Splitter.
#:
#: Die Voreinstellung von LangChain zerlegt zuerst an "\\n\\n" und arbeitet
#: sich dann herunter. Fuer Handbuecher ist die Reihenfolge bewusst anders:
#: Eine Kapitelueberschrift ("4. Rechnungen") ist eine natuerliche Grenze und
#: sollte vor einem beliebigen Absatzumbruch greifen. Sonst zerreisst ein
#: Chunk ein Thema mitten durch.
HANDBOOK_SEPARATORS: tuple[str, ...] = (
    "\n\n",      # Absatz
    "\n- ",      # Aufzaehlungspunkt
    "\n",        # Zeilenumbruch
    ". ",        # Satzende
    " ",
    "",
)

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

#: Ueberschrift eines Kapitels, z. B. "4. Rechnungen".
CHAPTER_PATTERN = re.compile(r"^(\d{1,2})\.\s+\S.+$")

#: Ueberschrift eines Unterabschnitts, z. B. "4.1 Finde ich...".
SECTION_PATTERN = re.compile(r"^(\d{1,2}\.\d)\s+(.+)$")

#: Titel wird auf diese Laenge gekuerzt, damit Metadaten kompakt bleiben.
TITLE_MAX_CHARS = 120

#: Erlaubte Splitter-Strategien.
STRATEGIES: tuple[str, ...] = ("recursive", "character", "token")

#: Signatur eines Ausgabe-Callbacks: bekommt eine fertige Textzeile.
EventHook = Callable[[str], None]


class UnknownStrategyError(ValueError):
    """Die angeforderte Splitter-Strategie existiert nicht."""


def _emit(hook: EventHook | None, line: str) -> None:
    """Ruft den Callback auf, wenn einer uebergeben wurde."""
    if hook is not None:
        hook(line)


# ---------------------------------------------------------------------------
# Strategien
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Grobe Token-Schaetzung: rund vier Zeichen je Token.

    Reicht, um Groessenordnungen zu planen. Eine echte Zaehlung braucht den
    Tokenizer des Zielmodells -- der gehoert spaeter in die Indexierung.
    """
    return max(1, len(text) // 4)


def build_splitter(
    strategy: str = "recursive",
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: Sequence[str] | None = None,
    length_function: str = "chars",
):
    """Erzeugt einen LangChain-Splitter fuer die gewaehlte Strategie.

    :param strategy: ``recursive`` (Standard), ``character`` oder ``token``.
    :param chunk_size: Zielgroesse eines Chunks.
    :param chunk_overlap: Ueberlappung zwischen zwei Chunks.
    :param separators: Nur fuer ``recursive`` -- eigene Trennzeichen-Reihenfolge.
    :param length_function: ``chars`` zaehlt Zeichen, ``tokens`` schaetzt Tokens.

    :raises UnknownStrategyError: bei unbekannter Strategie.

    Zu den Strategien:

    * ``recursive`` -- zerlegt an einer Hierarchie von Trennzeichen und faellt
      weiter herunter, wenn ein Stueck noch zu gross ist. Fuer Fliesstext die
      beste Wahl, weil sie Absaetze erhaelt.
    * ``character`` -- schneidet hart an einer festen Zeichenzahl. Simpel und
      vorhersagbar, zerlegt aber Saetze mitten durch.
    * ``token`` -- rechnet in Tokens statt Zeichen. Nuetzlich, wenn du das
      Kontextfenster des Modells exakt ausreizen willst.
    """
    if strategy not in STRATEGIES:
        raise UnknownStrategyError(
            f"Unbekannte Strategie: {strategy!r}. Erlaubt: {', '.join(STRATEGIES)}"
        )

    length_fn = len if length_function == "chars" else _estimate_tokens

    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(separators or DEFAULT_SEPARATORS),
            length_function=length_fn,
            add_start_index=True,
        )
    if strategy == "character":
        return CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_fn,
            add_start_index=True,
        )
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


# ---------------------------------------------------------------------------
# Metadaten
# ---------------------------------------------------------------------------

def detect_heading(text: str) -> tuple[str | None, str | None]:
    """Sucht eine Kapitel- oder Abschnittsueberschrift in den ersten Zeilen.

    :return: ``(kapitel, abschnitt)`` -- beide ``None``, wenn nichts passt.
        Eine Abschnittsueberschrift setzt beide Werte, eine Kapitelueberschrift
        setzt nur das Kapitel und loescht den Abschnitt.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:3]:
        section_match = SECTION_PATTERN.match(line)
        if section_match:
            number = section_match.group(1)
            return number.split(".")[0], number
        chapter_match = CHAPTER_PATTERN.match(line)
        if chapter_match:
            return chapter_match.group(1), None

    return None, None


def extract_title(text: str) -> str:
    """Zieht eine lesbare Ueberschrift aus dem Chunk-Anfang.

    Aus "4. Rechnungen" wird "4. Rechnungen", aus "4.1 Finde ich aeltere
    Rechnungen noch?" wird "Finde ich aeltere Rechnungen noch?".
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    title = lines[0]
    section_match = SECTION_PATTERN.match(title)
    if section_match:
        title = section_match.group(2).strip()
    else:
        chapter_match = CHAPTER_PATTERN.match(title)
        if chapter_match:
            number = chapter_match.group(1)
            title = f"{number}. {title.split('.', 1)[1].strip()}"

    return title[:TITLE_MAX_CHARS]


def enrich_chunk_metadata(chunks: Sequence[Document]) -> list[Document]:
    """Ergaenzt jeden Chunk um Chunk-Index, Kapitel und Abschnitt.

    Die Kapitelerkennung arbeitet auf dem Chunk-Text: Beginnt ein Chunk mit
    einer Ueberschrift, wird die Nummer uebernommen; sonst erbt er die des
    Vorgaengers. Das ist die Grundlage fuer Zitate wie "[4.1]".
    """
    current_chapter: str | None = None
    current_section: str | None = None

    for index, chunk in enumerate(chunks):
        chapter, section = detect_heading(chunk.page_content)
        if chapter is not None:
            current_chapter = chapter
            current_section = section

        chunk.metadata.update(
            {
                "chunk_index": index,
                "chunk_chars": len(chunk.page_content),
                "chapter": current_chapter,
                "section": current_section,
                "title": extract_title(chunk.page_content),
            }
        )
    return list(chunks)


# ---------------------------------------------------------------------------
# Aufteilen
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ChunkingStats:
    """Kennzahlen eines Chunking-Durchlaufs.

    ``oversized`` zaehlt Chunks, die groesser als ``chunk_size`` sind. Beim
    Recursive-Splitter passiert das, wenn ein einzelnes Wort oder eine lange
    Zeile ohne Trennzeichen den Zielwert ueberschreitet -- solche Chunks
    solltest du dir ansehen.
    """

    source_documents: int = 0
    chunks: int = 0
    total_chars: int = 0
    min_chars: int = 0
    max_chars: int = 0
    mean_chars: float = 0.0
    median_chars: float = 0.0
    oversized: int = 0
    chunk_size: int = 0
    overlap: int = 0
    strategy: str = "recursive"

    def summary_lines(self) -> list[str]:
        return [
            f"Strategie        : {self.strategy}",
            f"chunk_size       : {self.chunk_size}",
            f"chunk_overlap    : {self.overlap}",
            f"Quelldokumente   : {self.source_documents}",
            f"Chunks           : {self.chunks}",
            f"Zeichen gesamt   : {self.total_chars}",
            f"Zeichen je Chunk : min {self.min_chars} / median {self.median_chars:.0f} "
            f"/ mean {self.mean_chars:.0f} / max {self.max_chars}",
            f"ueber chunk_size : {self.oversized}",
        ]

    def summary(self) -> str:
        """Einzeiler fuer Tabellen und Logs."""
        return (
            f"{self.strategy:<10} chunks={self.chunks:<4} "
            f"median={self.median_chars:>6.0f} max={self.max_chars:>4} "
            f"uebergross={self.oversized}"
        )


def format_chunking_stats(stats: ChunkingStats) -> str:
    """Formatiert die Kennzahlen als mehrzeiligen Text."""
    return "\n".join(["Chunking", "=" * 62, *stats.summary_lines()])


def format_strategy_comparison(results: Sequence[ChunkingStats]) -> str:
    """Formatiert den Strategievergleich als Tabelle."""
    header = (
        f"{'STRATEGIE':<12} {'CHUNKS':>7} {'MIN':>6} {'MEDIAN':>8} "
        f"{'MAX':>6} {'UEBERGROSS':>11}"
    )
    lines = ["Strategievergleich", "=" * 62, header, "-" * 62]
    for stats in results:
        lines.append(
            f"{stats.strategy:<12} {stats.chunks:>7} {stats.min_chars:>6} "
            f"{stats.median_chars:>8.0f} {stats.max_chars:>6} {stats.oversized:>11}"
        )
    return "\n".join(lines)


def format_chunk_preview(
    chunks: Sequence[Document],
    limit: int = 3,
    body_chars: int = 220,
) -> str:
    """Formatiert die ersten ``limit`` Chunks mit Metadaten und Textanfang."""
    if not chunks:
        return "Keine Chunks vorhanden."

    shown = min(limit, len(chunks))
    lines = [f"Erste {shown} von {len(chunks)} Chunk(s):"]
    for chunk in chunks[:shown]:
        meta = chunk.metadata
        lines.append("-" * 62)
        lines.append(
            f"[{meta.get('chunk_index')}] {len(chunk.page_content)} Zeichen | "
            f"Kapitel {meta.get('chapter')} | Abschnitt {meta.get('section')}"
        )
        lines.append(f"Titel: {meta.get('title')}")
        body = chunk.page_content[:body_chars].replace("\n", " ")
        suffix = "..." if len(chunk.page_content) > body_chars else ""
        lines.append(f"{body}{suffix}")
    return "\n".join(lines)


def split_documents(
    documents: Iterable[Document],
    *,
    strategy: str = "recursive",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: Sequence[str] | None = None,
    on_event: EventHook | None = None,
) -> tuple[list[Document], ChunkingStats]:
    """Teilt Dokumente in Chunks und liefert Kennzahlen dazu.

    :param on_event: optionaler Callback, bekommt Fortschrittszeilen.

    :raises ValueError: wenn ``chunk_overlap`` >= ``chunk_size`` ist. Der
        Splitter wuerde sonst nicht terminieren oder Endlos-Chunks erzeugen.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) muss kleiner als "
            f"chunk_size ({chunk_size}) sein"
        )

    source_docs = list(documents)
    _emit(
        on_event,
        f"[start] {len(source_docs)} Dokument(e), Strategie {strategy}, "
        f"chunk_size {chunk_size}, overlap {chunk_overlap}",
    )

    splitter = build_splitter(
        strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    chunks = splitter.split_documents(source_docs)
    _emit(on_event, f"[split] {len(chunks)} Chunk(s) erzeugt")

    chunks = enrich_chunk_metadata(chunks)
    _emit(on_event, "[meta]  Kapitel und Abschnitt ergaenzt")

    sizes = [len(c.page_content) for c in chunks] or [0]
    stats = ChunkingStats(
        source_documents=len(source_docs),
        chunks=len(chunks),
        total_chars=sum(sizes),
        min_chars=min(sizes),
        max_chars=max(sizes),
        mean_chars=statistics.fmean(sizes),
        median_chars=statistics.median(sizes),
        oversized=sum(1 for size in sizes if size > chunk_size),
        chunk_size=chunk_size,
        overlap=chunk_overlap,
        strategy=strategy,
    )
    _emit(on_event, f"[fertig] {stats.summary()}")
    return chunks, stats


def compare_strategies(
    documents: Sequence[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    on_event: EventHook | None = None,
) -> list[ChunkingStats]:
    """Vergleicht alle Strategien bei gleichem Zielwert.

    Hilfreich, um vor dem Indexieren zu entscheiden, welche Strategie den
    Korpus am gleichmaessigsten zerlegt.
    """
    results: list[ChunkingStats] = []
    for strategy in STRATEGIES:
        _emit(on_event, f"[vergleich] {strategy}")
        _, stats = split_documents(
            documents,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        results.append(stats)
    return results


def chunks_to_markdown(chunks: Iterable[Document]) -> str:
    """Formatiert Chunks als Markdown-Block -- fuer die Dateiausgabe."""
    blocks: list[str] = []
    for chunk in chunks:
        meta = chunk.metadata
        blocks.append(
            f"### Chunk {meta.get('chunk_index')} "
            f"(Kapitel {meta.get('chapter')}, Abschnitt {meta.get('section')}, "
            f"{len(chunk.page_content)} Zeichen)\n{chunk.page_content}"
        )
    return "\n\n".join(blocks)


__all__ = [
    "HANDBOOK_SEPARATORS",
    "DEFAULT_SEPARATORS",
    "CHAPTER_PATTERN",
    "SECTION_PATTERN",
    "STRATEGIES",
    "TITLE_MAX_CHARS",
    "EventHook",
    "UnknownStrategyError",
    "build_splitter",
    "detect_heading",
    "extract_title",
    "enrich_chunk_metadata",
    "ChunkingStats",
    "format_chunking_stats",
    "format_strategy_comparison",
    "format_chunk_preview",
    "split_documents",
    "compare_strategies",
    "chunks_to_markdown",
]
