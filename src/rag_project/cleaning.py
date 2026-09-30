"""Textbereinigung fuer die RAG-Vorverarbeitung.

Reine Bibliothek -- keine CLI, kein ``argparse``. Die Ausfuehrung liegt in
:mod:`rag_project.main_cleaning`.

Verantwortung:
    - Steuerzeichen und unsichtbare Zeichen entfernen
    - Zeilenumbrueche und Leerzeichen normalisieren
    - Silbentrennung aufloesen, Aufzaehlungszeichen vereinheitlichen
    - Metadaten mitfuehren (Zeichen vor/nach der Bereinigung)

Ausgabe:
    Die Funktionen geben selbst nichts auf stdout aus. Fortschritt laeuft ueber
    einen ``on_event``-Callback, fertige Ausgabetexte liefern
    :func:`format_cleaning_stats` und :func:`format_cleaning_report`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Regex-Muster
# ---------------------------------------------------------------------------

#: Steuerzeichen ausser \t \n \r (die behandeln wir getrennt).
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: Zero-Width- und BOM-Zeichen, die in kopierten Texten haeufig stecken.
INVISIBLE_CHARS = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")

#: Weiches Trennzeichen und geschuetztes Leerzeichen.
SOFT_HYPHEN = "\u00ad"
NBSP = "\u00a0"

#: Drei oder mehr Leerzeilen -> zwei.
EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

#: Mehrfache Leerzeichen/Tabs -> eines.
EXCESS_SPACES = re.compile(r"[ \t]{2,}")

#: Leerzeichen am Zeilenanfang/ende.
TRAILING_SPACES = re.compile(r"[ \t]+\n")

#: Silbentrennung am Zeilenende: "Wort-\nfortsetzung" -> "Wortfortsetzung".
HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")

#: Aufzaehlungszeichen, die LangChain-Splitter nicht als Trenner kennen.
BULLET_NORMALIZE = re.compile(r"^\s*[•▪◦‣·]\s+", re.MULTILINE)

#: Signatur eines Ausgabe-Callbacks: bekommt eine fertige Textzeile.
EventHook = Callable[[str], None]


def _emit(hook: EventHook | None, line: str) -> None:
    """Ruft den Callback auf, wenn einer uebergeben wurde."""
    if hook is not None:
        hook(line)


# ---------------------------------------------------------------------------
# Bausteine
# ---------------------------------------------------------------------------

def strip_control_characters(text: str) -> str:
    """Entfernt Steuerzeichen und unsichtbare Zeichen."""
    text = CONTROL_CHARS.sub("", text)
    text = INVISIBLE_CHARS.sub("", text)
    return text.replace(SOFT_HYPHEN, "")


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Unicode-Normalisierung.

    ``NFKC`` fasst zusammengesetzte Zeichen zusammen und wandelt typografische
    Varianten in ihre Grundform -- gut fuer Volltextsuche. Ligaturen wie "fi"
    werden dabei aufgeloest.
    """
    return unicodedata.normalize(form, text)


def normalize_whitespace(text: str) -> str:
    """Vereinheitlicht Leerzeichen, Tabs und Leerzeilen."""
    text = text.replace(NBSP, " ")
    text = TRAILING_SPACES.sub("\n", text)
    text = EXCESS_SPACES.sub(" ", text)
    text = EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def join_hyphenated_linebreaks(text: str) -> str:
    """Fuegt am Zeilenende getrennte Woerter wieder zusammen.

    "Anmelde-\\nversuche" wird zu "Anmeldeversuche". Ohne diesen Schritt
    zerbricht die Silbentrennung die Suche nach dem Wort.
    """
    return HYPHEN_LINEBREAK.sub(r"\1\2", text)


def normalize_bullets(text: str, marker: str = "- ") -> str:
    """Vereinheitlicht Aufzaehlungszeichen auf ein Format."""
    return BULLET_NORMALIZE.sub(marker, text)


def collapse_to_single_line(text: str) -> str:
    """Macht aus einem Absatz eine Zeile -- fuer Metadaten und Titel."""
    return " ".join(text.split())


def count_step_changes(text: str) -> dict[str, int]:
    """Zaehlt, wie viele Treffer jeder Schritt im Text findet.

    Nur fuer Ausgaben gedacht -- hilfreich, um zu sehen, welcher Schritt
    tatsaechlich etwas bewirkt hat.
    """
    return {
        "control": len(CONTROL_CHARS.findall(text)) + len(INVISIBLE_CHARS.findall(text)),
        "soft_hyphen": text.count(SOFT_HYPHEN),
        "hyphen_break": len(HYPHEN_LINEBREAK.findall(text)),
        "bullets": len(BULLET_NORMALIZE.findall(text)),
        "nbsp": text.count(NBSP),
        "blank_runs": len(EXCESS_BLANK_LINES.findall(text)),
        "space_runs": len(EXCESS_SPACES.findall(text)),
    }


def clean_text(
    text: str,
    *,
    unicode_form: str | None = "NFKC",
    normalize_unicode_text: bool = True,
    join_hyphens: bool = True,
    unify_bullets: bool = True,
) -> str:
    """Vollstaendige Bereinigung eines Textes.

    Die Schritte laufen in dieser Reihenfolge ab, weil spaetere Schritte auf
    den Ergebnissen der frueheren aufbauen:

        1. Steuerzeichen entfernen
        2. Unicode normalisieren
        3. Silbentrennung zusammenfuegen
        4. Aufzaehlungszeichen vereinheitlichen
        5. Leerraum normalisieren

    :param text: Rohtext.
    :param unicode_form: NFKC (Standard), NFC oder None fuer keine Aenderung.
    :param normalize_unicode_text: Unicode-Normalisierung an/aus.
    :param join_hyphens: Silbentrennung aufloesen.
    :param unify_bullets: Aufzaehlungszeichen vereinheitlichen.
    """
    result = strip_control_characters(text)
    if normalize_unicode_text and unicode_form:
        result = normalize_unicode(result, unicode_form)
    if join_hyphens:
        result = join_hyphenated_linebreaks(result)
    if unify_bullets:
        result = normalize_bullets(result)
    return normalize_whitespace(result)


# ---------------------------------------------------------------------------
# Document-Ebene
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CleaningStats:
    """Bilanz eines Bereinigungsdurchlaufs."""

    documents: int = 0
    characters_before: int = 0
    characters_after: int = 0
    empty_after: int = 0

    @property
    def removed(self) -> int:
        return self.characters_before - self.characters_after

    @property
    def removed_percent(self) -> float:
        if not self.characters_before:
            return 0.0
        return 100.0 * self.removed / self.characters_before

    def summary(self) -> str:
        return (
            f"{self.documents} Dokument(e) | "
            f"{self.characters_before} -> {self.characters_after} Zeichen | "
            f"-{self.removed} ({self.removed_percent:.1f}%) | "
            f"{self.empty_after} leer"
        )

    def summary_lines(self) -> list[str]:
        """Mehrzeilige Ausgabe fuer die CLI."""
        return [
            f"Dokumente          : {self.documents}",
            f"Zeichen vorher     : {self.characters_before}",
            f"Zeichen nachher    : {self.characters_after}",
            f"entfernt           : {self.removed} ({self.removed_percent:.1f}%)",
            f"leer nach Reinigung: {self.empty_after}",
        ]


def format_cleaning_stats(stats: CleaningStats) -> str:
    """Formatiert die Bilanz als mehrzeiligen Text."""
    lines = ["Textbereinigung", "=" * 62, *stats.summary_lines()]
    return "\n".join(lines)


def format_cleaning_report(
    before: str,
    after: str,
    indent: str = "  ",
) -> str:
    """Zeigt, welche Schritte wie viel bewirkt haben.

    Vergleicht den Rohtext mit dem bereinigten Text und listet je Schritt die
    Zahl der Treffer sowie die Netto-Aenderung.
    """
    counts = count_step_changes(before)
    lines = [
        "Schritte (Treffer im Rohtext)",
        "-" * 62,
        f"{indent}Steuerzeichen/unsichtbar : {counts['control']}",
        f"{indent}weiches Trennzeichen     : {counts['soft_hyphen']}",
        f"{indent}Silbentrennung Zeilenende: {counts['hyphen_break']}",
        f"{indent}Aufzaehlungszeichen      : {counts['bullets']}",
        f"{indent}geschuetztes Leerzeichen : {counts['nbsp']}",
        f"{indent}Leerzeilen-Blöcke (3+)   : {counts['blank_runs']}",
        f"{indent}Mehrfach-Leerzeichen     : {counts['space_runs']}",
        "-" * 62,
        f"{indent}Zeichen {len(before)} -> {len(after)} "
        f"({len(before) - len(after):+d})",
    ]
    return "\n".join(lines)


def clean_document(
    document: Document,
    *,
    drop_empty: bool = True,
    **options: Any,
) -> Document | None:
    """Bereinigt ein einzelnes ``Document`` und ergaenzt Metadaten.

    Die Originalgroesse wandert als ``chars_before`` in die Metadaten -- so
    bleibt nachvollziehbar, wie stark eingegriffen wurde.

    :param drop_empty: ``True`` gibt ``None`` zurueck, wenn der Text danach leer
        ist. Leere Chunks sind in einem Vektorindex reines Rauschen.
    """
    before = len(document.page_content)
    cleaned = clean_text(document.page_content, **options)

    if drop_empty and not cleaned.strip():
        return None

    metadata = dict(document.metadata)
    metadata["chars_before"] = before
    metadata["chars_after"] = len(cleaned)
    metadata["cleaned"] = True

    return Document(page_content=cleaned, metadata=metadata)


def clean_documents(
    documents: Iterable[Document],
    *,
    drop_empty: bool = True,
    on_event: EventHook | None = None,
    **options: Any,
) -> tuple[list[Document], CleaningStats]:
    """Bereinigt eine Dokumentliste und liefert Bilanz plus Ergebnis.

    :param on_event: optionaler Callback, bekommt je Dokument eine Zeile.
    """
    document_list = list(documents)
    _emit(on_event, f"[start] {len(document_list)} Dokument(e) bereinigen")

    stats = CleaningStats()
    cleaned_documents: list[Document] = []

    for position, doc in enumerate(document_list, start=1):
        stats.documents += 1
        stats.characters_before += len(doc.page_content)

        result = clean_document(doc, drop_empty=drop_empty, **options)
        name = doc.metadata.get("file_name", f"Dokument {position}")

        if result is None:
            stats.empty_after += 1
            _emit(on_event, f"[{position}/{len(document_list)}] {name}: leer, verworfen")
            continue

        stats.characters_after += len(result.page_content)
        cleaned_documents.append(result)
        delta = len(result.page_content) - len(doc.page_content)
        _emit(
            on_event,
            f"[{position}/{len(document_list)}] {name}: "
            f"{len(doc.page_content)} -> {len(result.page_content)} Zeichen ({delta:+d})",
        )

    _emit(on_event, f"[fertig] {stats.summary()}")
    return cleaned_documents, stats


def join_cleaned_text(documents: Iterable[Document], separator: str = "\n\n") -> str:
    """Fuegt bereinigte Dokumente zu einem Text zusammen -- fuer Dateiausgabe."""
    return separator.join(doc.page_content for doc in documents)


__all__ = [
    "CONTROL_CHARS",
    "INVISIBLE_CHARS",
    "SOFT_HYPHEN",
    "NBSP",
    "EventHook",
    "strip_control_characters",
    "normalize_unicode",
    "normalize_whitespace",
    "join_hyphenated_linebreaks",
    "normalize_bullets",
    "collapse_to_single_line",
    "count_step_changes",
    "clean_text",
    "CleaningStats",
    "format_cleaning_stats",
    "format_cleaning_report",
    "clean_document",
    "clean_documents",
    "join_cleaned_text",
]
