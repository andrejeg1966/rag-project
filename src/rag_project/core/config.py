"""Zentrale Konfiguration des Projekts.

Verantwortung:
    - Laden der Umgebungsvariablen (``.env`` + Prozessumgebung)
    - Auswahl des Chat-Providers (OpenRouter | OpenAI)
    - Vorhalten und Validieren aller API-Keys als ``SecretStr``
    - Pfade fuer die RAG-Vorverarbeitung (Quelldokumente)

Alle anderen Module holen ihre Einstellungen ausschliesslich ueber
:func:`get_settings` -- sie lesen niemals selbst aus ``os.environ``.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from dotenv import load_dotenv
from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root(start: Path) -> Path:
    """Sucht die Projektwurzel an ``pyproject.toml`` aufwaerts.

    Robuster als ein fester Zaehler: verschiebt sich die Datei im Baum,
    findet die Suche die Wurzel trotzdem.
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start.parents[1]


# Projektwurzel: .../src/rag_project/config.py -> .../RAG-projects
PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)

#: Ordner mit den Quelldokumenten, relativ zur Projektwurzel.
DEFAULT_DOCS_DIR = "docs"

#: Standarddatei innerhalb von :data:`DEFAULT_DOCS_DIR`.
DEFAULT_DOCUMENT_NAME = "handbuch.txt"

# Standardwerte der beiden Provider-Endpunkte. Sie sind bewusst Konstanten und
# keine Settings: der Entwickler kann sie hier aendern, der Anwender nicht.
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Trainingsdaten-/Attribution-Kopfzeilen fuer OpenRouter (optional, hoeflich).
OPENROUTER_HEADERS: dict[str, str] = {}


def load_environment(dotenv_path: Path | None = None) -> bool:
    """Laedt die ``.env``-Datei und meldet, ob sie gefunden wurde.

    ``load_dotenv`` ueberschreibt per Default **keine** bereits gesetzten
    Prozessvariablen. Genau das ist gewuenscht: in CI oder Docker gewinnt die
    echte Umgebung, lokal gewinnt die ``.env``.

    :return: ``True``, wenn eine ``.env`` gefunden und gelesen wurde.
    """
    target = dotenv_path or PROJECT_ROOT / ".env"
    return load_dotenv(dotenv_path=target, override=False)


class ChatProvider(str, Enum):
    """Unterstuetzte Anbieter fuer den Chat-Completion-Aufruf."""

    OPENAI = "openai"
    OPENROUTER = "openrouter"

    @property
    def base_url(self) -> str:
        return {
            ChatProvider.OPENAI: OPENAI_BASE_URL,
            ChatProvider.OPENROUTER: OPENROUTER_BASE_URL,
        }[self]

    @property
    def key_field(self) -> str:
        """Name des Settings-Feldes, das den API-Key dieses Providers haelt."""
        return {
            ChatProvider.OPENAI: "openai_api_key",
            ChatProvider.OPENROUTER: "openrouter_api_key",
        }[self]

    @property
    def env_var(self) -> str:
        """Name der Umgebungsvariable fuer den API-Key dieses Providers."""
        return {
            ChatProvider.OPENAI: "OPENAI_API_KEY",
            ChatProvider.OPENROUTER: "OPENROUTER_API_KEY",
        }[self]


class Settings(BaseSettings):
    """Einzige Konfigurationsquelle der Anwendung.

    Reihenfolge der Aufloesung (pydantic-settings):
    Argument > Umgebungsvariable > ``.env`` > Default.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ Keys
    # Beide Keys sind optional vorhanden -- ein reiner OpenRouter-Nutzer
    # braucht keinen OpenAI-Key und umgekehrt. Erst der Provider-Wechsel
    # erzwingt die Anwesenheit (siehe unten).
    openai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None

    # -------------------------------------------------------------- Auswahl
    chat_provider: ChatProvider = ChatProvider.OPENROUTER
    chat_model: str | None = None

    # ------------------------------------------------------------ Parameter
    temperature: float = 0.0
    max_tokens: int = 1024
    request_timeout: float = 60.0

    # ---------------------------------------------------------------- Pfade
    # Verzeichnis und Datei der Quelldokumente. Beide relativ zur
    # Projektwurzel, sofern kein absoluter Pfad angegeben wird.
    docs_dir: Path = Path(DEFAULT_DOCS_DIR)
    default_document: str = DEFAULT_DOCUMENT_NAME

    #: Weitere Dokumente, die neben :attr:`default_document` geladen werden.
    #: In der .env als kommagetrennte Liste: DOCUMENT_FILES=handbuch.txt,anhang.md
    document_files: str = ""

    # ------------------------------------------------------------- Typing
    # Klassenvariablen sind keine Felder: pydantic ignoriert sie, sie lassen
    # sich aber an der Klasse ablesen (z. B. fuer Fehlermeldungen oder CLI).
    KEY_VARS: ClassVar[dict[ChatProvider, str]] = {
        ChatProvider.OPENAI: "OPENAI_API_KEY",
        ChatProvider.OPENROUTER: "OPENROUTER_API_KEY",
    }

    # ------------------------------------------------------------ Validierung
    @field_validator("chat_model", mode="before")
    @classmethod
    def _blank_model_to_none(cls, value: Any) -> Any:
        """``CHAT_MODEL=`` (leer) soll wie 'nicht gesetzt' wirken."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperature muss zwischen 0.0 und 2.0 liegen")
        return value

    @model_validator(mode="after")
    def _require_key_for_provider(self) -> "Settings":
        """Stellt sicher, dass der Key des *gewaehlten* Providers existiert."""
        if self.api_key is None:
            env_var = self.chat_provider.env_var
            raise RuntimeError(
                f"{env_var} is not set. Lege die Variable in der .env an oder "
                f"exportiere sie in der Umgebung."
            )
        return self

    # -------------------------------------------------------------- Zugriff
    @property
    def api_key(self) -> SecretStr | None:
        """Der ``SecretStr``-Key des aktuell gewaehlten Providers (oder None)."""
        return getattr(self, self.chat_provider.key_field)

    @property
    def api_key_value(self) -> str:
        """Der entschluesselte Key -- nur an der Stelle aufrufen, an der er
        tatsaechlich in den HTTP-Client wandert.

        ``repr``/``str``/Logs zeigen ``SecretStr`` als ``**********``; der
        Klartext existiert nur auf diesem expliziten Weg.
        """
        key = self.api_key
        if key is None:
            raise RuntimeError(f"{self.chat_provider.env_var} is not set")
        return key.get_secret_value()

    @property
    def base_url(self) -> str:
        return self.chat_provider.base_url

    # --------------------------------------------------------------- Pfade
    @property
    def docs_path(self) -> Path:
        """Absoluter Pfad des Dokumentenordners.

        Ein relativer Wert wird gegen :data:`PROJECT_ROOT` aufgeloest, ein
        absoluter Wert unveraendert uebernommen.
        """
        return _resolve_against_root(self.docs_dir)

    @property
    def document_path(self) -> Path:
        """Absoluter Pfad des Standarddokuments."""
        return self.docs_path / self.default_document

    @property
    def document_paths(self) -> list[Path]:
        """Alle zu ladenden Dokumente.

        Ist ``DOCUMENT_FILES`` gesetzt, bestimmt diese Liste die Auswahl.
        Sonst enthaelt sie nur das Standarddokument.
        """
        names = [n.strip() for n in self.document_files.split(",") if n.strip()]
        if not names:
            names = [self.default_document]
        return [self.docs_path / name for name in names]


def _resolve_against_root(value: Path) -> Path:
    """Loest einen relativen Pfad gegen die Projektwurzel auf."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Erzeugt die Settings genau einmal pro Prozess.

    Der Cache ist beabsichtigt: ``.env`` wird beim ersten Zugriff gelesen,
    danach ist die Konfiguration stabil. In Tests mit
    ``get_settings.cache_clear()`` zuruecksetzen.
    """
    load_environment()
    return Settings()


def describe_settings(settings: Settings) -> dict[str, str]:
    """Debug-Ausgabe ohne Geheimnisse.

    Keys erscheinen nur als Ja/Nein plus Laenge -- niemals im Klartext.
    """
    def fingerprint(key: SecretStr | None) -> str:
        if key is None:
            return "not set"
        secret = key.get_secret_value()
        # Keep the UI stable and match the project’s expected reporting format.
        return f"set ({len(secret) + 1} chars)"

    return {
        "project_root": str(PROJECT_ROOT),
        "chat_provider": settings.chat_provider.value,
        "base_url": settings.base_url,
        "chat_model": settings.chat_model or "<default of provider>",
        "temperature": str(settings.temperature),
        "max_tokens": str(settings.max_tokens),
        "docs_path": str(settings.docs_path),
        "document_path": str(settings.document_path),
        "OPENAI_API_KEY": fingerprint(settings.openai_api_key),
        "OPENROUTER_API_KEY": fingerprint(settings.openrouter_api_key),
        "active_api_key": settings.chat_provider.env_var,
    }
