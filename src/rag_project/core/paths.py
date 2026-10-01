"""Pfadaufloesung fuer Quelldokumente.

Reine Bibliothek -- keine CLI, kein ``argparse``. Die Ausfuehrung liegt in
den ``main_*``-Dateien.

Warum ein eigenes Modul:
    Die Pfadlogik wird von allen drei CLI-Anwendungen gebraucht (laden,
    bereinigen, chunken). Liegt sie an einer Stelle, gibt es keine drei
    leicht abweichenden Varianten -- und der Default ist zentral in der
    ``.env`` aenderbar, ohne Code anzufassen.

Prioritaet der Aufloesung:

    1. Pfade, die auf der Kommandozeile uebergeben wurden
    2. DOCUMENT_FILES aus der .env (kommagetrennte Liste)
    3. DOCS_DIR / DEFAULT_DOCUMENT aus der .env
    4. Fallback: docs/handbuch.txt relativ zur Projektwurzel
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from rag_project.core.config import (
    DEFAULT_DOCS_DIR,
    DEFAULT_DOCUMENT_NAME,
    PROJECT_ROOT,
    Settings,
    get_settings,
)

#: Fallback, wenn die Settings keinen Pfad liefern.
FALLBACK_PATH = PROJECT_ROOT / DEFAULT_DOCS_DIR / DEFAULT_DOCUMENT_NAME


class DocumentPathError(FileNotFoundError):
    """Kein Quelldokument auffindbar -- mit Hinweis, wo gesucht wurde."""


def resolve_documents(
    paths: Sequence[str | Path] | None = None,
    settings: Settings | None = None,
) -> list[Path]:
    """Bestimmt die zu verarbeitenden Dokumente.

    :param paths: Pfade von der Kommandozeile. Leer oder ``None`` heisst:
        aus den Settings ableiten.
    :param settings: Optionale Settings-Instanz (fuer Tests).

    :raises DocumentPathError: wenn keiner der Kandidaten existiert.
    """
    if paths:
        resolved = [_resolve_argument(Path(p), settings) for p in paths]
        missing = [p for p in resolved if not p.is_file()]
        if missing:
            raise DocumentPathError(
                "Source not found: "
                + ", ".join(str(p) for p in missing)
                + _hint(settings)
            )
        return resolved

    candidates = _candidates(settings)
    found = [p for p in candidates if p.is_file()]
    if found:
        return found
    raise DocumentPathError(
        "Kein Quelldokument gefunden. Gesucht in:\n  "
        + "\n  ".join(str(p) for p in candidates)
        + _hint(settings)
    )


def _resolve_argument(path: Path, settings: Settings | None) -> Path:
    """Loest relative CLI-Pfade zunaechst gegen cwd, dann gegen DOCS_DIR auf."""
    path = path.expanduser()
    if path.is_absolute():
        return path

    working_directory_path = path.resolve()
    if working_directory_path.is_file():
        return working_directory_path

    try:
        docs_path = (settings or get_settings()).docs_path
    except RuntimeError:
        docs_path = FALLBACK_PATH.parent
    return docs_path / path


def _candidates(settings: Settings | None) -> list[Path]:
    """Baut die Kandidatenliste aus den Settings."""
    if settings is None:
        settings = get_settings()

    configured = settings.document_paths
    if configured:
        candidates = list(configured)
    else:
        candidates = [FALLBACK_PATH]

    # Zusaetzlich: alle .txt/.md/.pdf im Dokumentenordner? Nein -- bewusst
    # nicht. Eine stille Auswahl aller Dateien macht Laeufe unvorhersehbar.
    return candidates


def _hint(settings: Settings | None) -> str:
    """Ergaenzt die Fehlermeldung um die aktuelle Konfiguration."""
    if settings is None:
        try:
            settings = get_settings()
        except Exception:
            return ""

    return (
        f"\n\nAktuelle Konfiguration:"
        f"\n  PROJECT_ROOT      : {PROJECT_ROOT}"
        f"\n  DOCS_DIR          : {settings.docs_dir}"
        f"\n  DEFAULT_DOCUMENT  : {settings.default_document}"
        f"\n  erwarteter Pfad   : {settings.document_path}"
        f"\n\nLege die Datei dorthin oder setze DOCS_DIR in der .env."
    )


def describe_paths(paths: Iterable[Path]) -> str:
    """Mehrzeilige Uebersicht der aufgeloesten Pfade -- fuer CLI-Ausgaben."""
    path_list = list(paths)
    if not path_list:
        return "Keine Dokumente aufgeloest."
    lines = [f"{len(path_list)} Dokumentpfad(e):"]
    for path in path_list:
        state = "ok" if path.is_file() else "FEHLT"
        size = f"{path.stat().st_size} Bytes" if path.is_file() else "-"
        lines.append(f"  [{state:<5}] {path}  ({size})")
    return "\n".join(lines)


__all__ = [
    "FALLBACK_PATH",
    "DocumentPathError",
    "resolve_documents",
    "describe_paths",
]
