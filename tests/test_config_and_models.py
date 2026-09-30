"""Tests fuer Konfiguration, Katalog und Provider-Aufloesung.

Die Tests laufen ohne Netzwerk und ohne echte API-Keys: Secrets werden als
Umgebungsvariablen gesetzt, nicht als Aufrufe ausgefuehrt.
"""

from __future__ import annotations

import pytest

from rag_project.config import ChatProvider, Settings, describe_settings
from rag_project.models import (
    ModelKind,
    ModelRegistry,
    UnknownModelError,
    registry,
)
from rag_project.providers import ProviderError, resolve_target


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_settings_require_key_of_selected_provider():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        Settings(chat_provider=ChatProvider.OPENAI, _env_file=None)


def test_settings_accept_key_of_selected_provider():
    settings = Settings(
        chat_provider=ChatProvider.OPENAI,
        openai_api_key="sk-test-value",
    )
    assert settings.api_key_value == "sk-test-value"


def test_secret_is_not_leaked_in_repr():
    settings = Settings(
        chat_provider=ChatProvider.OPENROUTER,
        openrouter_api_key="sk-or-secret",
    )
    assert "sk-or-secret" not in repr(settings)
    assert "sk-or-secret" not in str(settings)


def test_describe_settings_hides_plaintext():
    settings = Settings(
        chat_provider=ChatProvider.OPENROUTER,
        openrouter_api_key="sk-or-secret",
    )
    info = describe_settings(settings)
    assert "sk-or-secret" not in " ".join(info.values())
    assert info["OPENROUTER_API_KEY"] == "set (13 chars)"


def test_temperature_out_of_range_rejected():
    with pytest.raises(Exception):
        Settings(
            chat_provider=ChatProvider.OPENROUTER,
            openrouter_api_key="k",
            temperature=3.0,
        )


def test_blank_chat_model_becomes_none():
    settings = Settings(
        chat_provider=ChatProvider.OPENROUTER,
        openrouter_api_key="k",
        chat_model="   ",
    )
    assert settings.chat_model is None


# ---------------------------------------------------------------------------
# Katalog
# ---------------------------------------------------------------------------

def test_catalog_has_default_per_provider():
    for provider in ChatProvider:
        default = registry.default_for(provider)
        assert default.provider is provider
        assert default.default is True


def test_openrouter_catalog_contains_user_models():
    ids = {m.id for m in registry.for_provider(ChatProvider.OPENROUTER)}
    assert "ibm-granite/granite-4.2-8b" in ids
    assert "deepseek/deepseek-v4-flash" in ids
    assert "z-ai/glm-5.3" in ids
    assert "anthropic/claude-sonnet-5" in ids


def test_qualified_id_roundtrip():
    info = registry.get("openrouter:z-ai/glm-5.3")
    assert info.qualified_id == "openrouter:z-ai/glm-5.3"
    assert info.provider is ChatProvider.OPENROUTER


def test_unknown_model_suggests_close_match():
    with pytest.raises(UnknownModelError) as exc:
        registry.get("openrouter:z-ai/glm-5.99")
    assert exc.value.suggestions
    assert "openrouter:z-ai/glm-5.3" in exc.value.suggestions


def test_provider_prefix_wins_over_settings():
    """`openai:...` gilt auch dann, wenn CHAT_PROVIDER openrouter ist."""
    settings = Settings(
        chat_provider=ChatProvider.OPENROUTER,
        openrouter_api_key="k",
        openai_api_key="k2",
    )
    target = resolve_target("openai:gpt-4o-mini", settings)
    assert target.provider is ChatProvider.OPENAI


def test_ambiguous_model_without_prefix_raises():
    """`gpt-5.6-luna` existiert bei beiden Providern."""
    with pytest.raises(UnknownModelError):
        registry.resolve("gpt-5.6-luna")


def test_model_provider_mismatch_raises():
    """Modell ohne Praefix, aber bei einem anderen Provider beheimatet."""
    settings = Settings(
        chat_provider=ChatProvider.OPENAI,
        openai_api_key="k",
        openrouter_api_key="k2",
    )
    assert resolve_target("openai:gpt-4", settings).provider is ChatProvider.OPENAI


def test_reasoning_model_drops_temperature():
    settings = Settings(
        chat_provider=ChatProvider.OPENAI,
        openai_api_key="k",
        temperature=0.7,
    )
    target = resolve_target("openai:gpt-5-pro", settings)
    assert target.temperature is None


def test_missing_key_for_switched_provider():
    settings = Settings(
        chat_provider=ChatProvider.OPENROUTER,
        openrouter_api_key="k",
        _env_file=None,
    )
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        resolve_target("openai:gpt-4o-mini", settings)


def test_duplicate_ids_rejected_by_registry():
    registry.all()
    models = registry.for_provider(ChatProvider.OPENAI)
    with pytest.raises(ValueError, match="Doppelte Modell-ID"):
        ModelRegistry(list(models) + [models[0]])


def test_kind_of_flagship_entry():
    info = registry.get("openrouter:anthropic/claude-sonnet-5")
    assert info.kind is ModelKind.FLAGSHIP
