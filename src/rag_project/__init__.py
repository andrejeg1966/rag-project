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

from rag_project.core.config import (
    ChatProvider,
    Settings,
    describe_settings,
    get_settings,
    load_environment,
)
from rag_project.app.main import build_parser, main
from rag_project.core.models import (
    ModelInfo,
    ModelKind,
    ModelRegistry,
    UnknownModelError,
    format_catalog,
    list_models,
    registry,
)
from rag_project.app.rag_lib import (
    PipelineResult,
    RagSystem,
    create_rag_system,
    run_pipeline,
)

from rag_project.llm.providers import (
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
    # rag pipeline
    "RagSystem",
    "PipelineResult",
    "create_rag_system",
    "run_pipeline",
    # entry point
    "main",
    "build_parser",
]
