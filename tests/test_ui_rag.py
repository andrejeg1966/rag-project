from pathlib import Path

import gradio as gr

from rag_project.app import ui_rag


def test_model_choices_are_four_openrouter_models():
    model_values = [value for _, value in ui_rag.MODEL_CHOICES]

    assert model_values == [
        "deepseek/deepseek-v4-flash",
        "gpt-5.6-luna",
        "anthropic/claude-sonnet-5",
        "z-ai/glm-5.3",
    ]
    assert ui_rag.MODEL_CHOICES[0][1] == "deepseek/deepseek-v4-flash"


def test_get_system_prompt_uses_selected_language():
    assert "Kontext" in ui_rag.get_system_prompt("de")
    assert "Context" in ui_rag.get_system_prompt("en")


def test_rag_answer_includes_metadata():
    metadata = ui_rag.format_rag_metadata(
        "deepseek/deepseek-v4-flash",
        temperature=0.3,
    )

    assert "Model: DeepSeek V4 Flash" in metadata
    assert "Provider: openrouter" in metadata
    assert "Base URL: https://openrouter.ai/api/v1" in metadata
    assert "Temperature: 0.3" in metadata
    assert "Embedding model: openai/text-embedding-3-small" in metadata
    assert "Vector database: Qdrant" in metadata
    assert "Chunking method: 800 chars, 100 overlap, recursive" in metadata


def test_clear_file_output_resets_file_input():
    result = ui_rag.clear_file_output()

    assert result == ("", gr.update(value=None))


def test_inspect_file_processes_uploaded_document(tmp_path):
    source = tmp_path / "test.txt"
    source.write_text("Kapitel 1\n\nEin paar   Zeilen.\n\n- Punkt 1\n- Punkt 2\n", encoding="utf-8")

    result = ui_rag.inspect_file(source)

    assert "### Chunk" in result
    assert "Kapitel" in result
    assert "Abschnitt" in result
    assert "Ein paar Zeilen" in result
