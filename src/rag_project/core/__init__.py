"""Core project configuration and shared data objects."""

from rag_project.core.config import ChatProvider, Settings, describe_settings, get_settings, load_environment
from rag_project.core.models import ModelInfo, ModelKind, ModelRegistry, UnknownModelError, format_catalog, list_models, registry

__all__ = [
    "ChatProvider",
    "Settings",
    "get_settings",
    "load_environment",
    "describe_settings",
    "ModelInfo",
    "ModelKind",
    "ModelRegistry",
    "UnknownModelError",
    "list_models",
    "format_catalog",
    "registry",
]
