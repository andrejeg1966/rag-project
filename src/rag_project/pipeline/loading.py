"""Dokumentenladen fuer die RAG-Pipeline.

Reine Bibliothek -- keine CLI, kein ``argparse``. Die Ausfuehrung liegt in
:mod:`rag_project.app.main_loading`.

Verantwortung:
    - Quelldateien in LangChain-``Document``-Objekte ueberfuehren
    - Format anhand der Endung oder explizit waehlen
    - Metadaten anreichern (Quellpfad, Format, Groesse)

Ausgabe:
    Die oeffentlichen Funktionen geben selbst nichts auf stdout aus. Wer
    Fortschritt sehen will, uebergibt einen ``on_event``-Callback oder nutzt
    :func:`format_load_report`, das einen fertigen Ausgabetext liefert.

Unterstuetzte Quellen:

    Dateibasiert (Endung entscheidet):
        .txt .rst        TextLoader
        .md .markdown    UnstructuredMarkdownLoader
        .csv             CSVLoader
        .pdf             PyPDFLoader

    Netzbasiert (explizit anzugeben, keine Endung):
        http:// https:// WebBaseLoader
        wikipedia:...    WikipediaLoader
        csv:...          CSVLoader fuer entfernte Dateien

Die netzbasierten Loader brauchen ``langchain-community`` und ziehen je nach
Ziel weitere Pakete (etwa ``beautifulsoup4`` fuer HTML). Ein fehlendes Paket
wird als :class:`LoaderDependencyError` gemeldet, nicht als Importfehler.

Wichtig: Netzbasierte Loader sind nicht deterministisch. Derselbe Aufruf kann
zu zwei Zeitpunkten verschiedene Inhalte liefern. Fuer einen reproduzierbaren
Index sollte das Ergebnis einmal geladen und als Datei abgelegt werden.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from rag_project.core.config import load_environment

load_environment()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Formatregister
# ---------------------------------------------------------------------------

TEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".rst"})
PDF_SUFFIXES: frozenset[str] = frozenset({".pdf"})
MARKDOWN_SUFFIXES: frozenset[str] = frozenset({".md", ".markdown"})
CSV_SUFFIXES: frozenset[str] = frozenset({".csv"})

#: Alle Formate, die sich an der Endung erkennen lassen.
FILE_SUFFIXES: frozenset[str] = (
    TEXT_SUFFIXES | PDF_SUFFIXES | MARKDOWN_SUFFIXES | CSV_SUFFIXES
)

#: Praefixe fuer netzbasierte Quellen. Sie werden *vor* der Endungspruefung
#: ausgewertet, weil eine URL mit .csv-Endung sonst als lokale Datei gaelte.
URL_PREFIXES: tuple[str, ...] = ("http://", "https://")
WIKIPEDIA_PREFIX = "wikipedia:"
REMOTE_CSV_PREFIX = "csv:"

#: Signatur eines Ausgabe-Callbacks: bekommt eine fertige Textzeile.
EventHook = Callable[[str], None]


class UnsupportedFormatError(ValueError):
    """Die Dateiendung oder das Quellpraefix wird nicht unterstuetzt."""


class SourceNotFoundError(FileNotFoundError):
    """Die Quelldatei existiert nicht."""


class LoaderDependencyError(RuntimeError):
    """Ein fuer den Loader noetiges Paket fehlt.

    Getrennt von :class:`UnsupportedFormatError`: das Format ist bekannt und
    unterstuetzt, es fehlt nur die Installation. Die Meldung nennt das Paket.
    """


@dataclass(frozen=True, slots=True)
class LoadReport:
    """Ergebnis eines Ladevorgangs -- fuer CLI-Ausgaben und Logs."""

    path: Path
    format: str
    documents: int
    characters: int

    def summary(self) -> str:
        return (
            f"{self.path.name:<28} {self.format:<6} "
            f"{self.documents:>3} Dok. {self.characters:>7} Zeichen"
        )


# ---------------------------------------------------------------------------
# Ausgabe-Helfer
# ---------------------------------------------------------------------------

def _emit(hook: EventHook | None, line: str) -> None:
    """Ruft den Callback auf, wenn einer uebergeben wurde.

    Ohne Callback passiert nichts -- so bleibt die Bibliothek still und die
    CLI entscheidet selbst, ob und was sie zeigt.
    """
    if hook is not None:
        hook(line)


def format_load_report(reports: Sequence[LoadReport]) -> str:
    """Baut die mehrzeilige Zusammenfassung eines Ladevorgangs.

    Liefert den fertigen Text zurueck, statt ihn auszugeben -- damit ist die
    Formatierung testbar und die CLI entscheidet ueber das Ziel.
    """
    if not reports:
        return "Keine Dokumente geladen."

    width = max(len(r.path.name) for r in reports)
    lines = [
        "Geladene Dokumente",
        "=" * 62,
        f"{'DATEI':<{width}}  FORMAT  DOK.   ZEICHEN",
        "-" * 62,
    ]
    for report in reports:
        lines.append(
            f"{report.path.name:<{width}}  {report.format:<6}  "
            f"{report.documents:>4}  {report.characters:>8}"
        )
    total_docs = sum(r.documents for r in reports)
    total_chars = sum(r.characters for r in reports)
    lines.append("-" * 62)
    lines.append(
        f"{'SUMME':<{width}}  {'':<6}  {total_docs:>4}  {total_chars:>8}"
    )
    return "\n".join(lines)


def format_document_metadata(document: Document, indent: str = "  ") -> str:
    """Formatiert die Metadaten eines Dokuments als mehrzeiligen Text."""
    lines = ["Metadaten:"]
    for key, value in sorted(document.metadata.items()):
        lines.append(f"{indent}{key:<12}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Format- und Quellenerkennung
# ---------------------------------------------------------------------------

def detect_source_kind(source: str | Path) -> str:
    """Bestimmt die Art der Quelle.

    :return: ``file``, ``url``, ``wikipedia`` oder ``remote_csv``.

    Die netzbasierten Praefixe werden zuerst geprueft. Eine entfernte
    CSV-Datei wuerde sonst wegen ihrer Endung als lokale Datei behandelt.
    """
    if isinstance(source, Path):
        return "file"

    text = str(source).strip()
    lowered = text.lower()

    if lowered.startswith(WIKIPEDIA_PREFIX):
        return "wikipedia"
    if lowered.startswith(REMOTE_CSV_PREFIX) and not lowered.startswith(URL_PREFIXES):
        return "remote_csv"
    if lowered.startswith(URL_PREFIXES):
        return "url"
    return "file"


def detect_format(path: Path) -> str:
    """Bestimmt das Format anhand der Endung.

    :raises UnsupportedFormatError: bei unbekannter oder fehlender Endung.
    """
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in CSV_SUFFIXES:
        return "csv"
    if suffix in PDF_SUFFIXES:
        return "pdf"
    raise UnsupportedFormatError(
        f"Unsupported file format: {suffix or '<keine Endung>'!r} "
        f"({path.name}). Unterstuetzt: "
        f"{', '.join(sorted(FILE_SUFFIXES))}"
    )


def _import_or_die(module: str, names: Sequence[str], package: str):
    """Importiert Namen und uebersetzt einen Importfehler in eine klare Meldung.

    Ohne diesen Helfer ergaebe ein fehlendes Paket einen nackten
    ``ImportError`` irgendwo im Loader -- fuer den Anwender nicht zu deuten.
    """
    try:
        imported = __import__(module, fromlist=list(names))
    except ImportError as exc:
        raise LoaderDependencyError(
            f"Fuer diese Quelle wird {package!r} gebraucht, ist aber nicht "
            f"installiert. Installieren mit: uv add {package} ({exc})"
        ) from exc
    return [getattr(imported, name) for name in names]


# ---------------------------------------------------------------------------
# Einzelne Quelle laden
# ---------------------------------------------------------------------------

def _load_text_file(source: Path, encoding: str) -> list[Document]:
    loader = TextLoader(
        str(source), encoding=encoding, autodetect_encoding=True
    )
    return loader.load()


def _load_markdown_file(source: Path) -> list[Document]:
    """Laedt Markdown mit dem Unstructured-Loader.

    Der Loader setzt ``unstructured`` voraus; fehlt es, faellt die Funktion auf
    den reinen Textloader zurueck. Markdown ist Text -- der Rueckfall kostet
    nur die Absatztrennung, die Unstructured zusaetzlich liefert.
    """
    try:
        (MarkdownLoader,) = _import_or_die(
            "langchain_community.document_loaders",
            ["UnstructuredMarkdownLoader"],
            "unstructured",
        )
    except LoaderDependencyError:
        return _load_text_file(source, encoding="utf-8")

    return MarkdownLoader(str(source), mode="single").load()


def _load_csv_file(source: Path, encoding: str) -> list[Document]:
    """Laedt CSV mit dem LangChain-Loader, sonst mit eigenem Rueckfall.

    Der Rueckfall erzeugt je Zeile ein Dokument mit einer Kopfzeile
    ``spalte: wert`` -- dasselbe Ausgabeformat wie der Standardloader, nur
    ohne dessen optionale Abhaengigkeiten.
    """
    try:
        (CSVLoader,) = _import_or_die(
            "langchain_community.document_loaders", ["CSVLoader"], "langchain-community"
        )
        return CSVLoader(str(source), encoding=encoding).load()
    except (LoaderDependencyError, TypeError):
        pass

    text = source.read_text(encoding=encoding, errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []

    documents: list[Document] = []
    for index, row in enumerate(reader):
        body = "\n".join(
            f"{name}: {(row.get(name) or '').strip()}" for name in fieldnames
        )
        documents.append(
            Document(
                page_content=body,
                metadata={"row_index": index},
            )
        )
    return documents


def _load_url(source: str, on_event: EventHook | None) -> list[Document]:
    """Laedt eine Webseite als Text."""
    (WebBaseLoader,) = _import_or_die(
        "langchain_community.document_loaders", ["WebBaseLoader"], "beautifulsoup4"
    )
    user_agent = os.getenv("USER_AGENT")
    headers = {"User-Agent": user_agent} if user_agent else None
    _emit(on_event, f"[netz]  {source}")
    return WebBaseLoader(source, header_template=headers).load()


def _load_wikipedia(
    topic: str,
    *,
    language: str = "de",
    query: str | None = None,
    max_docs: int = 1,
    on_event: EventHook | None = None,
) -> list[Document]:
    """Laedt Artikel aus Wikipedia.

    :param topic: Suchbegriff, also der Teil hinter dem Praefix.
    :param query: Freitext-Suche statt direkter Artikelabruf.
    :param max_docs: Zahl der zurueckzugebenden Artikel.
    """
    (WikipediaLoader,) = _import_or_die(
        "langchain_community.document_loaders", ["WikipediaLoader"], "wikipedia"
    )
    _emit(on_event, f"[wikipedia] {topic} (Sprache {language}, max {max_docs})")
    kwargs: dict[str, object] = {"lang": language, "load_max_docs": max_docs}
    if query:
        kwargs["query"] = query
    else:
        kwargs["query"] = topic
    return WikipediaLoader(**kwargs).load()


def _load_remote_csv(url: str, on_event: EventHook | None) -> list[Document]:
    """Laedt eine entfernte CSV-Datei ueber HTTP.

    Der Inhalt wird zuerst geholt und dann wie eine lokale Datei gelesen --
    so bleibt genau eine CSV-Auswertung im Modul.
    """
    (CSVLoader,) = _import_or_die(
        "langchain_community.document_loaders", ["CSVLoader"], "langchain-community"
    )
    _emit(on_event, f"[netz]  {url}")
    return CSVLoader(url).load()


def load_document(
    path: str | Path,
    encoding: str = "utf-8",
    on_event: EventHook | None = None,
    *,
    wikipedia_language: str = "de",
    wikipedia_max_docs: int = 1,
) -> list[Document]:
    """Laedt eine einzelne Quelle als Liste von ``Document``-Objekten.

    Die Art der Quelle entscheidet den Loader:

        Datei      Endung bestimmt das Format (siehe :func:`detect_format`)
        URL        Webseite als Text
        wikipedia: Artikel zu einem Suchbegriff
        csv:       entfernte CSV-Datei ueber einen URL-Parameter

    :param path: Pfad, URL oder Quellpraefix.
    :param encoding: Kodierung fuer Text- und CSV-Dateien.
    :param on_event: optionaler Callback, bekommt Fortschrittszeilen.

    :raises SourceNotFoundError: wenn eine lokale Datei fehlt.
    :raises UnsupportedFormatError: wenn Format oder Praefix unbekannt sind.
    :raises LoaderDependencyError: wenn ein noetiges Paket fehlt.
    """
    kind = detect_source_kind(path)

    if kind == "wikipedia":
        topic = str(path).strip()[len(WIKIPEDIA_PREFIX):].strip()
        if not topic:
            raise UnsupportedFormatError(
                "Wikipedia-Quelle ohne Suchbegriff: 'wikipedia:<Thema>' erwartet."
            )
        documents = _load_wikipedia(
            topic,
            language=wikipedia_language,
            max_docs=wikipedia_max_docs,
            on_event=on_event,
        )
        source_label = f"wikipedia:{topic}"
        fmt = "wiki"
        size_bytes = 0

    elif kind == "remote_csv":
        url = str(path).strip()[len(REMOTE_CSV_PREFIX):].strip()
        if not url:
            raise UnsupportedFormatError(
                "CSV-Quelle ohne Adresse: 'csv:<url>' erwartet."
            )
        documents = _load_remote_csv(url, on_event)
        source_label = url
        fmt = "csv"
        size_bytes = 0

    elif kind == "url":
        url = str(path).strip()
        documents = _load_url(url, on_event)
        source_label = url
        fmt = "web"
        size_bytes = 0

    else:
        source = Path(path).expanduser()
        if not source.is_file():
            raise SourceNotFoundError(f"Source not found: {source}")

        fmt = detect_format(source)
        size_bytes = source.stat().st_size
        _emit(on_event, f"[laden] {source.name} ({size_bytes} Bytes, {fmt})")

        if fmt == "pdf":
            documents = PyPDFLoader(str(source)).load()
        elif fmt == "markdown":
            documents = _load_markdown_file(source)
        elif fmt == "csv":
            documents = _load_csv_file(source, encoding)
        else:
            # autodetect_encoding faengt Dateien ab, die nicht UTF-8 sind --
            # ein echtes Problem bei Handbuechern aus Windows-Programmen.
            documents = _load_text_file(source, encoding)

        source_label = str(source.resolve())

    for index, doc in enumerate(documents):
        doc.metadata.update(
            {
                "source": source_label,
                "file_name": Path(source_label).name
                if fmt not in {"web", "wiki", "csv"}
                else source_label,
                "format": fmt,
                "size_bytes": size_bytes,
                "page_index": index,
            }
        )

    chars = sum(len(d.page_content) for d in documents)
    _emit(on_event, f"[ok]    {len(documents)} Dokument(e), {chars} Zeichen")
    return documents


def load_documents(
    paths: Iterable[str | Path],
    encoding: str = "utf-8",
    on_event: EventHook | None = None,
    **kwargs: object,
) -> list[Document]:
    """Laedt mehrere Quellen nacheinander.

    Ein Fehler bei einer Quelle bricht den Durchlauf ab -- bewusst: eine
    stumm uebersprungene Quelle ist bei RAG schlimmer als ein Abbruch.

    :param on_event: optionaler Callback, bekommt je Quelle eine Zeile.
    :param kwargs: weitere Optionen fuer :func:`load_document`.
    """
    path_list = list(paths)
    _emit(on_event, f"[start] {len(path_list)} Quelle(n) zu laden")

    collected: list[Document] = []
    for position, path in enumerate(path_list, start=1):
        label = Path(str(path)).name if detect_source_kind(path) == "file" else str(path)
        _emit(on_event, f"[{position}/{len(path_list)}] {label}")
        collected.extend(load_document(path, encoding=encoding, on_event=on_event, **kwargs))

    _emit(on_event, f"[fertig] {len(collected)} Dokument(e) geladen")
    return collected


def describe(documents: Sequence[Document]) -> list[LoadReport]:
    """Fasst geladene Dokumente je Quelle zusammen."""
    grouped: dict[str, LoadReport] = {}
    for doc in documents:
        key = str(doc.metadata.get("source", "<unbekannt>"))
        chars = len(doc.page_content)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = LoadReport(
                path=Path(key),
                format=str(doc.metadata.get("format", "?")),
                documents=1,
                characters=chars,
            )
        else:
            grouped[key] = LoadReport(
                path=existing.path,
                format=existing.format,
                documents=existing.documents + 1,
                characters=existing.characters + chars,
            )
    return list(grouped.values())


def to_jsonl(documents: Sequence[Document], target: str | Path) -> Path:
    """Schreibt geladene Dokumente als JSONL-Zeilen -- fuer die Zwischenablage.

    Eine netzbasierte Quelle ist nicht reproduzierbar. Wer den Index stabil
    halten will, laedt einmal und legt das Ergebnis hier ab.

    :return: Pfad der geschriebenen Datei.
    """
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(
                json.dumps(
                    {"page_content": doc.page_content, "metadata": doc.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return destination


def from_jsonl(source: str | Path) -> list[Document]:
    """Liest eine mit :func:`to_jsonl` geschriebene Datei zurueck."""
    path = Path(source)
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            documents.append(
                Document(
                    page_content=record["page_content"],
                    metadata=record.get("metadata", {}),
                )
            )
    return documents


__all__ = [
    "TEXT_SUFFIXES",
    "PDF_SUFFIXES",
    "MARKDOWN_SUFFIXES",
    "CSV_SUFFIXES",
    "FILE_SUFFIXES",
    "URL_PREFIXES",
    "WIKIPEDIA_PREFIX",
    "REMOTE_CSV_PREFIX",
    "EventHook",
    "UnsupportedFormatError",
    "SourceNotFoundError",
    "LoaderDependencyError",
    "LoadReport",
    "detect_source_kind",
    "detect_format",
    "load_document",
    "load_documents",
    "describe",
    "format_load_report",
    "format_document_metadata",
    "to_jsonl",
    "from_jsonl",
]
