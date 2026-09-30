"""Provider-Schicht: verbindet Konfiguration, Modellkatalog und LLM-Client.

Verantwortung:
    - aus Settings + Katalog einen fertig konfigurierten Client bauen
    - den Klauselvertrag (``temperature`` ja/nein) je Modell anwenden
    - Fehler des Anbieters in sprechende Meldungen uebersetzen

Hier -- und nur hier -- wird ein API-Key entschluesselt
(``SecretStr.get_secret_value()``). Kein anderes Modul ruft diese Methode auf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

from rag_project.config import (
    ChatProvider,
    Settings,
    describe_settings,
    get_settings,
)
from rag_project.models import ModelInfo, UnknownModelError, registry


class ProviderError(RuntimeError):
    """Basisfehler der Provider-Schicht."""


class MissingCredentialsError(ProviderError):
    """Der fuer den gewaehlten Provider noetige Key fehlt."""


class ClientBuildError(ProviderError):
    """Der Client liess sich nicht erzeugen (falscher Parameter o. ae.)."""


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    """Das Ergebnis der Aufloesung: Provider, Modell, Client-Parameter.

    Dieses Objekt ist die einzige Stelle, an der die drei Informationsquellen
    (Settings, Katalog, Anwendercode) zusammenlaufen. Alles ab hier arbeitet
    nur noch mit :class:`ResolvedTarget`.
    """

    provider: ChatProvider
    model: ModelInfo
    base_url: str
    temperature: float | None
    max_tokens: int

    def summary(self) -> str:
        temp = "n/a" if self.temperature is None else f"{self.temperature:g}"
        return (
            f"{self.model.qualified_id} | base_url={self.base_url} "
            f"| temperature={temp} | max_tokens={self.max_tokens}"
        )


def resolve_target(
    model: str | None = None,
    settings: Settings | None = None,
    model_registry=registry,
) -> ResolvedTarget:
    """Bestimmt Provider und Modell aus Anwendercode, Settings und Katalog.

    Aufloesungsreihenfolge fuer das Modell:

    1. ``model``-Argument (z. B. per CLI-Flag ``--model``)
    2. ``CHAT_MODEL`` aus den Settings
    3. Default des Providers aus dem Katalog

    Das Provider-Praefix in der Modellkennung hat Vorrang vor
    ``CHAT_PROVIDER``: wer ``openai:gpt-5`` schreibt, will OpenAI.

    :raises UnknownModelError: Modell nicht im Katalog.
    :raises ProviderError: Angefragtes Modell passt nicht zum aktiven Provider.
    """
    settings = settings or get_settings()

    raw = (model or settings.chat_model or "").strip()

    if raw:
        prefix = raw.partition(":")[0].lower() if ":" in raw else None
        if prefix and prefix in {p.value for p in ChatProvider}:
            # Praefix ist explizit -> es entscheidet, nicht CHAT_PROVIDER.
            target = model_registry.resolve(raw)
            provider = target.provider
        else:
            # Ohne Praefix: gegen den aktiven Provider aufloesen.
            provider = settings.chat_provider
            try:
                target = model_registry.get(f"{provider.value}:{raw}")
            except UnknownModelError:
                target = model_registry.resolve(raw)
                provider = target.provider
    else:
        provider = settings.chat_provider
        target = model_registry.default_for(provider)

    if target.provider is not provider:
        raise ProviderError(
            f"Modell {target.qualified_id!r} gehoert zu "
            f"{target.provider.value!r}, aktiv ist aber {provider.value!r}. "
            f"Setze CHAT_PROVIDER={target.provider.value} oder waehle ein "
            f"Modell dieses Providers."
        )

    _assert_key_present(provider, settings)

    # Modelle ohne temperature-Unterstuetzung bekommen den Parameter gar nicht.
    temperature = settings.temperature if target.supports_temperature else None

    return ResolvedTarget(
        provider=provider,
        model=target,
        base_url=provider.base_url,
        temperature=temperature,
        max_tokens=settings.max_tokens,
    )


def _assert_key_present(provider: ChatProvider, settings: Settings) -> None:
    """Klare Meldung, bevor der HTTP-Client mit leerem Key startet.

    Die Settings-Pruefung fange den Normalfall ab; hier wird zusaetzlich der
    Provider-Wechsel zur Laufzeit abgedeckt.
    """
    key_field = provider.key_field
    if getattr(settings, key_field) is None:
        raise MissingCredentialsError(
            f"{provider.env_var} is not set. "
            f"Lege den Key in der .env an: {provider.env_var}=..."
        )


def build_llm(
    model: str | None = None,
    *,
    settings: Settings | None = None,
    streaming: bool = False,
    **overrides: Any,
):
    """Erzeugt den Chat-Client fuer das aufgeloeste Ziel.

    :param model: optionale Modellkennung, schlaegt ``CHAT_MODEL``.
    :param streaming: Token-Streaming aktivieren (spaeter fuer die UI).
    :param overrides: weitere ``ChatOpenAI``-Parameter, etwa ``top_p``.

    :raises MissingCredentialsError: Key des Providers fehlt.
    :raises ClientBuildError: Client-Erzeugung fehlgeschlagen.
    """
    settings = settings or get_settings()
    target = resolve_target(model, settings)

    kwargs: dict[str, Any] = {
        "model": target.model.id,
        "base_url": target.base_url,
        # EINZIGE Entschluesselungsstelle im Projekt:
        "api_key": settings.api_key_value,
        "max_tokens": target.max_tokens,
        "timeout": settings.request_timeout,
        "streaming": streaming,
    }
    if target.temperature is not None:
        kwargs["temperature"] = target.temperature

    # OpenRouter empfiehlt Herkunfts-Kopfzeilen; rein optional.
    if target.provider is ChatProvider.OPENROUTER:
        kwargs["default_headers"] = {
            "HTTP-Referer": "https://localhost/rag-project",
            "X-Title": "RAG Project Demo",
        }

    kwargs.update(overrides)

    try:
        return ChatOpenAI(**kwargs)
    except Exception as exc:  # Bibliotheksfehler -> eigener Typ
        raise ClientBuildError(
            f"Client fuer {target.model.qualified_id} nicht erzeugbar: {exc}"
        ) from exc


def provider_report(settings: Settings | None = None) -> dict[str, str]:
    """Statusuebersicht fuer CLI und Debugging -- ohne Klartext-Keys."""
    settings = settings or get_settings()
    info = describe_settings(settings)
    target = resolve_target(settings=settings)
    info["resolved_model"] = target.model.qualified_id
    info["resolved_summary"] = target.summary()
    return info


__all__ = [
    "ProviderError",
    "MissingCredentialsError",
    "ClientBuildError",
    "ResolvedTarget",
    "resolve_target",
    "build_llm",
    "provider_report",
]
