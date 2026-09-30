"""Dokumentenladen fuer die RAG-Pipeline.

Reine Bibliothek -- keine CLI, kein ``argparse``. Die Ausfuehrung liegt in
:mod:`rag_project.main_loading`.

Verantwortung:
    - Quelldateien in LangChain-``Document``-Objekte ueberfuehren
    - Format anhand der Endung oder explizit waehlen
    - Metadaten anreichern (Quellpfad, Format, Groesse)

Ausgabe:
    Die oeffentlichen Funktionen geben selbst nichts auf stdout aus. Wer
    Fortschritt sehen will, uebergibt einen ``on_event``-Callback oder nutzt
    :func:`format_load_report`, das einen fertigen Ausgabetext liefert.

Unterstuetzte Formate: .txt, .md, .markdown, .rst (text), .pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Formatregister
# ---------------------------------------------------------------------------

TEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".md", ".markdown", ".rst"})
PDF_SUFFIXES: frozenset[str] = frozenset({".pdf"})

#: Signatur eines Ausgabe-Callbacks: bekommt eine fertige Textzeile.
EventHook = Callable[[str], None]


class UnsupportedFormatError(ValueError):
    """Die Dateiendung wird nicht unterstuetzt."""


class SourceNotFoundError(FileNotFoundError):
    """Die Quelldatei existiert nicht."""


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
# Laden
# ---------------------------------------------------------------------------

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
        f"{', '.join(sorted(TEXT_SUFFIXES | PDF_SUFFIXES))}"
    )


def load_document(
    path: str | Path,
    encoding: str = "utf-8",
    on_event: EventHook | None = None,
) -> list[Document]:
    """Laedt eine einzelne Datei als Liste von ``Document``-Objekten.

    Eine PDF liefert pro Seite ein Dokument, eine Textdatei genau eines.

    :param path: Pfad zur Quelldatei.
    :param encoding: Kodierung fuer Textdateien (PDF ignoriert das).
    :param on_event: optionaler Callback, bekommt Fortschrittszeilen.

    :raises SourceNotFoundError: wenn die Datei fehlt.
    :raises UnsupportedFormatError: wenn das Format nicht unterstuetzt wird.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise SourceNotFoundError(f"Source not found: {source}")

    fmt = detect_format(source)
    size_bytes = source.stat().st_size
    _emit(on_event, f"[laden] {source.name} ({size_bytes} Bytes, {fmt})")

    if fmt == "pdf":
        loader = PyPDFLoader(str(source))
    else:
        # autodetect_encoding faengt Dateien ab, die nicht UTF-8 sind --
        # ein echtes Problem bei Handbuechern aus Windows-Programmen.
        loader = TextLoader(str(source), encoding=encoding, autodetect_encoding=True)

    documents = loader.load()

    for index, doc in enumerate(documents):
        doc.metadata.update(
            {
                "source": str(source.resolve()),
                "file_name": source.name,
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
) -> list[Document]:
    """Laedt mehrere Dateien nacheinander.

    Ein Fehler bei einer Datei bricht den Durchlauf ab -- bewusst: eine
    stumm uebersprungene Quelle ist bei RAG schlimmer als ein Abbruch.

    :param on_event: optionaler Callback, bekommt je Datei eine Zeile.
    """
    path_list = list(paths)
    _emit(on_event, f"[start] {len(path_list)} Datei(en) zu laden")

    collected: list[Document] = []
    for position, path in enumerate(path_list, start=1):
        _emit(on_event, f"[{position}/{len(path_list)}] {Path(path).name}")
        collected.extend(load_document(path, encoding=encoding))

    _emit(on_event, f"[fertig] {len(collected)} Dokument(e) geladen")
    return collected


def describe(documents: Sequence[Document]) -> list[LoadReport]:
    """Fasst geladene Dokumente je Quelldatei zusammen."""
    grouped: dict[str, LoadReport] = {}
    for doc in documents:
        key = doc.metadata.get("source", "<unbekannt>")
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


__all__ = [
    "TEXT_SUFFIXES",
    "PDF_SUFFIXES",
    "EventHook",
    "UnsupportedFormatError",
    "SourceNotFoundError",
    "LoadReport",
    "detect_format",
    "load_document",
    "load_documents",
    "describe",
    "format_load_report",
    "format_document_metadata",
]
