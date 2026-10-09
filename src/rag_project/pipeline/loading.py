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
        .pdf             PyPDFLoader

    Netzbasiert (explizit anzugeben, keine Endung):
        http:// https:// WebBaseLoader (HTML) oder PyPDFLoader (PDF)

CSV, Markdown und Wikipedia werden hier nicht geladen; dafuer gibt es die
Module ``csv_load``, ``markdown_load`` und ``wikipedia_load`` mit eigenen CLIs.

Die netzbasierten Loader brauchen ``langchain-community`` und ziehen je nach
Ziel weitere Pakete (etwa ``beautifulsoup4`` fuer HTML). Ein fehlendes Paket
wird als :class:`LoaderDependencyError` gemeldet, nicht als Importfehler.

Wichtig: Netzbasierte Loader sind nicht deterministisch. Derselbe Aufruf kann
zu zwei Zeitpunkten verschiedene Inhalte liefern. Fuer einen reproduzierbaren
Index sollte das Ergebnis einmal geladen und als Datei abgelegt werden.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.parse import unquote, urlsplit

import requests

from rag_project.core.config import DEFAULT_DOCS_DIR, PROJECT_ROOT, load_environment

load_environment()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Formatregister
# ---------------------------------------------------------------------------

TEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".rst"})
PDF_SUFFIXES: frozenset[str] = frozenset({".pdf"})

#: Alle Formate, die sich an der Endung erkennen lassen.
FILE_SUFFIXES: frozenset[str] = TEXT_SUFFIXES | PDF_SUFFIXES

#: Praefixe fuer netzbasierte Quellen. Sie werden *vor* der Endungspruefung
#: ausgewertet, weil eine URL mit .pdf-Endung sonst als lokale Datei gaelte.
URL_PREFIXES: tuple[str, ...] = ("http://", "https://")

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

    :return: ``file`` oder ``url``.

    Das URL-Praefix wird zuerst geprueft. Eine entfernte PDF-Datei
    wuerde sonst wegen ihrer Endung als lokale Datei behandelt.
    """
    if isinstance(source, Path):
        return "file"

    lowered = str(source).strip().lower()
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
    if suffix in PDF_SUFFIXES:
        return "pdf"
    raise UnsupportedFormatError(
        f"Unsupported file format: {suffix or '<keine Endung>'!r} "
        f"({path.name}). Unterstuetzt: "
        f"{', '.join(sorted(FILE_SUFFIXES))}. CSV, Markdown und Wikipedia "
        f"laden csv-load, markdown-load und wikipedia-load."
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


def _load_url(source: str, on_event: EventHook | None) -> list[Document]:
    """Laedt eine Webseite als Text."""
    (WebBaseLoader,) = _import_or_die(
        "langchain_community.document_loaders", ["WebBaseLoader"], "beautifulsoup4"
    )
    user_agent = os.getenv("USER_AGENT")
    headers = {"User-Agent": user_agent} if user_agent else None
    _emit(on_event, f"[netz]  {source}")
    return WebBaseLoader(source, header_template=headers).load()


def _is_pdf_url(source: str) -> bool:
    """Erkennt PDF-URLs anhand des URL-Pfads, auch mit Query-Parametern."""
    return Path(urlsplit(source).path).suffix.lower() == ".pdf"


def _load_remote_pdf(
    source: str, on_event: EventHook | None
) -> tuple[list[Document], int]:
    """Laedt eine PDF-URL temporaer herunter und liest sie mit PyPDFLoader."""
    user_agent = os.getenv("USER_AGENT")
    headers = {"User-Agent": user_agent} if user_agent else None
    _emit(on_event, f"[netz]  {source}")

    with requests.get(source, headers=headers, timeout=30) as response:
        response.raise_for_status()
        content = response.content

    if b"%PDF-" not in content[:1024]:
        raise ValueError(f"URL liefert keine gueltige PDF-Datei: {source}")

    with tempfile.TemporaryDirectory() as directory:
        pdf_path = Path(directory) / "download.pdf"
        pdf_path.write_bytes(content)
        documents = PyPDFLoader(str(pdf_path)).load()

    docs_directory = PROJECT_ROOT / DEFAULT_DOCS_DIR
    docs_directory.mkdir(parents=True, exist_ok=True)
    original_name = Path(unquote(urlsplit(source).path)).name or "download.pdf"
    destination = docs_directory / original_name
    counter = 1
    while destination.exists():
        if destination.read_bytes() == content:
            break
        destination = docs_directory / f"{Path(original_name).stem}_{counter}{Path(original_name).suffix}"
        counter += 1
    else:
        destination.write_bytes(content)

    _emit(on_event, f"[gespeichert] {destination}")
    return documents, len(content)


def load_document(
    path: str | Path,
    encoding: str = "utf-8",
    on_event: EventHook | None = None,
) -> list[Document]:
    """Laedt eine einzelne Quelle als Liste von ``Document``-Objekten.

    Die Art der Quelle entscheidet den Loader:

        Datei      Endung bestimmt das Format (siehe :func:`detect_format`)
        URL        Webseite als Text oder PDF-Dokument

    :param path: Pfad oder URL.
    :param encoding: Kodierung fuer Textdateien.
    :param on_event: optionaler Callback, bekommt Fortschrittszeilen.

    :raises SourceNotFoundError: wenn eine lokale Datei fehlt.
    :raises UnsupportedFormatError: wenn das Format unbekannt ist.
    :raises LoaderDependencyError: wenn ein noetiges Paket fehlt.
    """
    kind = detect_source_kind(path)

    if kind == "url":
        url = str(path).strip()
        source_label = url
        if _is_pdf_url(url):
            documents, size_bytes = _load_remote_pdf(url, on_event)
            fmt = "pdf"
        else:
            documents = _load_url(url, on_event)
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
        else:
            # autodetect_encoding faengt Dateien ab, die nicht UTF-8 sind --
            # ein echtes Problem bei Handbuechern aus Windows-Programmen.
            documents = _load_text_file(source, encoding)

        source_label = str(source.resolve())

    if kind == "url" and fmt == "pdf":
        file_name = Path(urlsplit(source_label).path).name
    elif fmt != "web":
        file_name = Path(source_label).name
    else:
        file_name = source_label

    for index, doc in enumerate(documents):
        doc.metadata.update(
            {
                "source": source_label,
                "file_name": file_name,
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
    "FILE_SUFFIXES",
    "URL_PREFIXES",
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
