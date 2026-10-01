"""Modellkatalog: welche Chat-Modelle stehen je Provider zur Verfuegung.

Diese Datei ist reine Daten plus duenne Zugriffs-Helfer. Sie kennt weder
HTTP-Clients noch API-Keys -- die Trennung haelt :mod:`rag_project.providers`
frei von Modellwissen und laesst den Katalog unabhaengig erweitern.

Neue Modelle: einfach einen Eintrag in :data:`RAW_MODELS` ergaenzen. Die ID
wird dabei bewusst *ohne* Provider-Praefix gepflegt (``ibm-granite/granite-4.2-8b``),
weil der Provider spaeter aus dem Praefix des Anwendercodes abgeleitet wird.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from rag_project.core.config import ChatProvider


class ModelKind(str, Enum):
    """Grobe Einordnung -- steuert Kosten, Latenz und Eignung."""

    FLAGSHIP = "flagship"   # staerkste Qualitaet, hoechste Kosten
    BALANCED = "balanced"   # Standardwahl fuer RAG
    FAST = "fast"           # klein und guenstig, fuer Massen-Operationen
    REASONING = "reasoning" # explizites Nachdenken, langsam aber gruendlich
    OPEN_WEIGHT = "open_weight"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Ein einzelnes Modell im Katalog.

    :param id: Modell-ID beim Provider, ohne Provider-Praefix.
    :param provider: Zu welchem Anbieter das Modell gehoert.
    :param label: Sprechender Name fuer Ausgaben und Logs.
    :param kind: Einordnung (siehe :class:`ModelKind`).
    :param context_window: Token-Fenster laut Anbieter, falls bekannt.
    :param supports_temperature: Manche Reasoning-Modelle lehnen ``temperature``
        ab bzw. ignorieren den Wert. Steuert, ob wir ihn mitschicken.
    :param default: Ist dies die Standardwahl des Providers?
    :param note: Kurzer Hinweis fuer CLI-Ausgaben.
    """

    id: str
    provider: ChatProvider
    label: str
    kind: ModelKind
    context_window: int | None = None
    supports_temperature: bool = True
    default: bool = False
    note: str = ""

    @property
    def qualified_id(self) -> str:
        """Vollstaendige Kennung ``<provider>:<id>``.

        Genau dieses Format nimmt :func:`resolve_model` wieder entgegen, und
        genau dieses Format liest man in ``.env`` und Logs.
        """
        return f"{self.provider.value}:{self.id}"

    def __str__(self) -> str:  # pragma: no cover - reine Ausgabe
        marker = " *" if self.default else ""
        return f"{self.qualified_id:<52} {self.kind.value:<13}{marker}"


# ---------------------------------------------------------------------------
# Katalog
# ---------------------------------------------------------------------------

OPENROUTER_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="gpt-5.6-luna",
        provider=ChatProvider.OPENROUTER,
        label="GPT-5.6 Luna (via OpenRouter)",
        kind=ModelKind.BALANCED,
        context_window=200_000,
        default=True,
        note="Empfehlung fuer RAG: gute Balance aus Qualitaet und Kosten",
    ),
    ModelInfo(
        id="anthropic/claude-sonnet-5",
        provider=ChatProvider.OPENROUTER,
        label="Claude Sonnet 5 (via OpenRouter)",
        kind=ModelKind.FLAGSHIP,
        context_window=200_000,
        note="Stark bei langen Kontexten und Zitat-Treue",
    ),
    ModelInfo(
        id="deepseek/deepseek-v4-flash",
        provider=ChatProvider.OPENROUTER,
        label="DeepSeek V4 Flash",
        kind=ModelKind.FAST,
        context_window=128_000,
        note="Kostenguenstig fuer Indexierung und Massen-Aufrufe",
    ),
    ModelInfo(
        id="z-ai/glm-5.3",
        provider=ChatProvider.OPENROUTER,
        label="GLM 5.3",
        kind=ModelKind.BALANCED,
        context_window=128_000,
    ),
    ModelInfo(
        id="ibm-granite/granite-4.2-8b",
        provider=ChatProvider.OPENROUTER,
        label="IBM Granite 4.2 8B",
        kind=ModelKind.OPEN_WEIGHT,
        context_window=128_000,
        note="Offenes Modell, Sinnvoll fuer lokale Nachvollziehbarkeit",
    ),
)

OPENAI_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="gpt-4o-mini",
        provider=ChatProvider.OPENAI,
        label="GPT-4o mini",
        kind=ModelKind.FAST,
        context_window=128_000,
        default=True,
        note="Solide, guenstige Standardwahl fuer Einsteiger",
    ),
    ModelInfo(
        id="gpt-4",
        provider=ChatProvider.OPENAI,
        label="GPT-4",
        kind=ModelKind.OPEN_WEIGHT if False else ModelKind.FLAGSHIP,
        context_window=8_192,
        note="Aelterer Klassiker, kleines Kontextfenster",
    ),
    ModelInfo(
        id="gpt-4o",
        provider=ChatProvider.OPENAI,
        label="GPT-4o",
        kind=ModelKind.BALANCED,
        context_window=128_000,
    ),
    ModelInfo(
        id="gpt-5",
        provider=ChatProvider.OPENAI,
        label="GPT-5",
        kind=ModelKind.FLAGSHIP,
        context_window=200_000,
        note="Reasoning aktiv -- langsam, dafuer gruendlich",
    ),
    ModelInfo(
        id="gpt-5-mini",
        provider=ChatProvider.OPENAI,
        label="GPT-5 mini",
        kind=ModelKind.FAST,
        context_window=200_000,
    ),
    ModelInfo(
        id="gpt-5-pro",
        provider=ChatProvider.OPENAI,
        label="GPT-5 Pro",
        kind=ModelKind.REASONING,
        context_window=200_000,
        supports_temperature=False,
        note="Lehnt temperature ab",
    ),
    ModelInfo(
        id="gpt-5.4-mini",
        provider=ChatProvider.OPENAI,
        label="GPT-5.4 mini",
        kind=ModelKind.FAST,
        context_window=200_000,
    ),
    ModelInfo(
        id="gpt-5.6-sol",
        provider=ChatProvider.OPENAI,
        label="GPT-5.6 Sol",
        kind=ModelKind.FLAGSHIP,
        context_window=200_000,
    ),
    ModelInfo(
        id="gpt-5.6-luna",
        provider=ChatProvider.OPENAI,
        label="GPT-5.6 Luna",
        kind=ModelKind.BALANCED,
        context_window=200_000,
        note="Gleiche Modellfamilie wie der OpenRouter-Eintrag",
    ),
)

RAW_MODELS: tuple[ModelInfo, ...] = OPENROUTER_MODELS + OPENAI_MODELS


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Nachschlagewerk ueber alle bekannten Modelle.

    Bewusst ein normales Objekt statt einer Klasse mit Klassenvariablen: so
    laesst sich das Register in Tests mit einem eigenen Katalog instanziieren.
    """

    def __init__(self, models: Iterable[ModelInfo] = RAW_MODELS) -> None:
        self._by_qualified: dict[str, ModelInfo] = {}
        self._by_id: dict[str, list[ModelInfo]] = {}

        for model in models:
            key = model.qualified_id.lower()
            if key in self._by_qualified:
                raise ValueError(f"Doppelte Modell-ID im Katalog: {key}")
            self._by_qualified[key] = model
            self._by_id.setdefault(model.id.lower(), []).append(model)

    # ---------------------------------------------------------- Abfragen
    def all(self) -> tuple[ModelInfo, ...]:
        return tuple(self._by_qualified.values())

    def for_provider(self, provider: ChatProvider) -> tuple[ModelInfo, ...]:
        return tuple(m for m in self.all() if m.provider is provider)

    def default_for(self, provider: ChatProvider) -> ModelInfo:
        for model in self.for_provider(provider):
            if model.default:
                return model
        # Kein explizites Default markiert -> erster Eintrag des Providers.
        candidates = self.for_provider(provider)
        if not candidates:
            raise LookupError(f"Kein Modell fuer Provider {provider.value} im Katalog")
        return candidates[0]

    def get(self, qualified_id: str) -> ModelInfo:
        """Sucht ``provider:id`` -- mit klarer Fehlermeldung bei Tippfehlern.

        :raises UnknownModelError: wenn die Kennung nicht im Katalog steht.
        """
        found = self._by_qualified.get(qualified_id.strip().lower())
        if found is None:
            raise UnknownModelError(qualified_id, self.suggest(qualified_id))
        return found

    def suggest(self, qualified_id: str, limit: int = 3) -> list[str]:
        """Naeheste Katalog-Treffer zu einer unbekannten Kennung."""
        needle = qualified_id.strip().lower()
        scored: list[tuple[int, str]] = []
        for info in self.all():
            key = info.qualified_id.lower()
            score = _levenshtein(needle, key)
            scored.append((score, info.qualified_id))
        scored.sort()
        return [name for _, name in scored[:limit]]

    def provider_from_prefix(self, raw: str) -> tuple[ChatProvider, str]:
        """Zerlegt eine Nutzereingabe in ``(provider, modell_id)``.

        Akzeptiert beide Formen:

        * ``openrouter:ibm-granite/granite-4.2-8b``  -- Praefix gewinnt
        * ``ibm-granite/granite-4.2-8b``            -- ohne Praefix, wenn die ID
          eindeutig nur bei einem Provider existiert

        :raises UnknownModelError: wenn die ID keinem Provider zugeordnet
            werden kann oder bei mehreren vorkommt.
        """
        raw = raw.strip()
        if ":" in raw:
            prefix, _, rest = raw.partition(":")
            try:
                provider = ChatProvider(prefix.strip().lower())
            except ValueError as exc:
                raise UnknownModelError(raw, self.suggest(raw)) from exc
            return provider, rest.strip()

        hits = self._by_id.get(raw.lower(), [])
        if len(hits) == 1:
            return hits[0].provider, hits[0].id
        if len(hits) > 1:
            providers = ", ".join(h.provider.value for h in hits)
            raise UnknownModelError(
                raw, [f"{p}:{raw}" for p in providers.split(", ")]
            )
        raise UnknownModelError(raw, self.suggest(raw))

    def resolve(self, raw: str) -> ModelInfo:
        """Aufloesung in einem Schritt: beliebige Eingabe -> :class:`ModelInfo`."""
        provider, model_id = self.provider_from_prefix(raw)
        return self.get(f"{provider.value}:{model_id}")


class UnknownModelError(LookupError):
    """Ein Modell wurde angefragt, das nicht im Katalog steht."""

    env_var = "CHAT_MODEL"

    def __init__(self, requested: str, suggestions: list[str] | None = None) -> None:
        self.requested = requested
        self.suggestions = suggestions or []
        parts = [
            f"Unknown model: {requested!r}. "
            f"Setze {self.env_var} auf eine Kennung aus dem Katalog."
        ]
        if self.suggestions:
            parts.append("Meintest du: " + ", ".join(self.suggestions) + "?")
        super().__init__(" ".join(parts))


def _levenshtein(a: str, b: str) -> int:
    """Editierabstand -- fuer die Vorschlagsliste, kein externes Paket noetig."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,        # Loeschen
                    current[j - 1] + 1,     # Einfuegen
                    previous[j - 1] + (ca != cb),  # Ersetzen
                )
            )
        previous = current
    return previous[-1]


#: Standard-Register der Anwendung.
registry = ModelRegistry()


def list_models(provider: ChatProvider | None = None) -> tuple[ModelInfo, ...]:
    """Bequemer Zugriff fuer CLI und Tests."""
    if provider is None:
        return registry.all()
    return registry.for_provider(provider)


def format_catalog(provider: ChatProvider | None = None) -> str:
    """Mehrzeilige Katalogausgabe. ``*`` markiert das Default je Provider."""
    header = f"{'MODELL':<52} {'TYP':<13}"
    lines = [header, "-" * len(header)]
    if provider is None:
        for current in ChatProvider:
            lines.append(f"\n[{current.value}]")
            lines.extend(str(m) for m in registry.for_provider(current))
    else:
        lines.extend(str(m) for m in registry.for_provider(provider))
    return "\n".join(lines)


__all__ = [
    "ModelInfo",
    "ModelKind",
    "ModelRegistry",
    "UnknownModelError",
    "registry",
    "list_models",
    "format_catalog",
    "RAW_MODELS",
    "OPENAI_MODELS",
    "OPENROUTER_MODELS",
]
