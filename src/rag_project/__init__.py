"""Paket-Initialisierung fuer ``rag_project``.

Stellt den Einstiegspunkt ``main`` bereit, auf den der Skript-Eintrag in
``pyproject.toml`` verweist::

    [project.scripts]
    rag-project = "rag_project:main"

Die Public API bleibt absichtlich klein -- sie umfasst genau die Bausteine,
die die Konfigurations- und Provider-Schicht nach aussen anbietet.
"""

from __future__ import annotations

__version__ = "0.1.0"

from rag_project.config import (
    ChatProvider,
    Settings,
    describe_settings,
    get_settings,
    load_environment,
)
from rag_project.main import build_parser, main
from rag_project.models import (
    ModelInfo,
    ModelKind,
    ModelRegistry,
    UnknownModelError,
    format_catalog,
    list_models,
    registry,
)
from rag_project.providers import (
    ClientBuildError,
    MissingCredentialsError,
    ProviderError,
    ResolvedTarget,
    build_llm,
    provider_report,
    resolve_target,
)

__all__ = [
    "__version__",
    # config
    "ChatProvider",
    "Settings",
    "get_settings",
    "load_environment",
    "describe_settings",
    # models
    "ModelInfo",
    "ModelKind",
    "ModelRegistry",
    "UnknownModelError",
    "list_models",
    "format_catalog",
    "registry",
    # providers
    "ProviderError",
    "MissingCredentialsError",
    "ClientBuildError",
    "ResolvedTarget",
    "resolve_target",
    "build_llm",
    "provider_report",
    # entry point
    "main",
    "build_parser",
]
