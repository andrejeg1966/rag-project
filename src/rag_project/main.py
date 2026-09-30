"""Einstiegspunkt der Anwendung.

Aktueller Stand: Nur die Konfigurations- und Provider-Schicht. Eine
RAG-Pipeline ist bewusst noch nicht implementiert -- dieser Einstieg dient
dazu, Setup und Modellwahl zu pruefen.

Aufruf ueber den Skript-Eintrag aus ``pyproject.toml``::

    uv run rag-project
    uv run rag-project --check
    uv run rag-project --models
    uv run rag-project --models openai
    uv run rag-project --model openai:gpt-4o-mini
"""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from rag_project.config import ChatProvider, describe_settings, get_settings
from rag_project.models import UnknownModelError, format_catalog
from rag_project.providers import (
    ProviderError,
    build_llm,
    provider_report,
    resolve_target,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-project",
        description="RAG Project Demo -- Konfiguration und Modellverwaltung.",
    )
    parser.add_argument(
        "--models",
        nargs="?",
        const="all",
        metavar="PROVIDER",
        help="Modellkatalog anzeigen, optional gefiltert nach openai|openrouter",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Konfiguration pruefen: Keys vorhanden, Modell aufloesbar",
    )
    parser.add_argument(
        "--model",
        metavar="KENNUNG",
        help="Modell fuer diesen Lauf, z. B. openrouter:z-ai/glm-5.3",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Aktive Einstellungen anzeigen (Keys nur als Fingerabdruck)",
    )
    return parser


def cmd_models(argument: str) -> int:
    if argument == "all":
        print(format_catalog())
        return 0
    try:
        provider = ChatProvider(argument.lower())
    except ValueError:
        valid = ", ".join(p.value for p in ChatProvider)
        print(f"Unbekannter Provider {argument!r}. Erlaubt: {valid}", file=sys.stderr)
        return 2
    print(format_catalog(provider))
    return 0


def cmd_check(model: str | None) -> int:
    settings = get_settings()
    target = resolve_target(model, settings)
    print("Konfiguration in Ordnung.")
    print(f"  Provider : {target.provider.value}")
    print(f"  Modell   : {target.model.qualified_id} ({target.model.label})")
    print(f"  Typ      : {target.model.kind.value}")
    print(f"  Base-URL : {target.base_url}")
    print(f"  Ziel     : {target.summary()}")
    return 0


def cmd_show_config() -> int:
    info = provider_report()
    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"{key:<{width}} : {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.models:
            return cmd_models(args.models)
        if args.show_config:
            return cmd_show_config()
        if args.check:
            return cmd_check(args.model)
    except UnknownModelError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        # Umfasst fehlende Keys (config) und Fehler der Provider-Schicht.
        print(f"Konfigurationsfehler: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"Ungueltige Einstellungen:\n{exc}", file=sys.stderr)
        return 1

    # Standardlauf ohne Argumente: kurzer Statusbericht.
    settings = get_settings()
    info = describe_settings(settings)
    print("RAG Project Demo")
    print(f"  Provider : {info['chat_provider']}")
    print(f"  Modell   : {info['chat_model']}")
    print(f"  Base-URL : {info['base_url']}")
    print()
    print("Noch keine RAG-Pipeline implementiert.")
    print("Optionen: --check  --models  --show-config  --model <kennung>")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
