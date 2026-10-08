from __future__ import annotations

import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag_project.app import rag_lib
from rag_project.core.config import get_settings
from rag_project.core.models import registry
from rag_project.pipeline.cleaning import clean_documents
from rag_project.pipeline.chunking import chunks_to_markdown, split_documents
from rag_project.pipeline.loading import load_documents


MODEL_CHOICES = [
    ("OpenRouter Fast", "deepseek/deepseek-v4-flash"),
    ("OpenRouter Balanced", "gpt-5.6-luna"),
    ("OpenRouter Sonnet", "anthropic/claude-sonnet-5"),
    ("OpenRouter GLM", "z-ai/glm-5.3"),
]

SYSTEM_PROMPTS: dict[str, str] = {
    "de": (
        "Du bist ein hilfreicher Assistent. Nutze den folgenden Kontext, "
        "um die Frage zu beantworten. "
        "Wenn der Kontext keine relevanten Informationen enthält, sage: "
        "'Diese Information ist nicht in meinen Dokumenten enthalten.' "
        "Halte die Antwort kurz und prägnant.\n\n"
        "Kontext:\n\n{context}"
    ),
    "en": (
        "You are a helpful assistant. Use the following context to answer the "
        "question. If the context contains no relevant information, say: "
        "'This information is not in my documents.' Keep the answer short and "
        "to the point.\n\n"
        "Context:\n\n{context}"
    ),
}


DEFAULT_PROMPT = "Hallo, ich bin ein RAG-Chatbot. Bitte stellen Sie Ihre Fragen."


def get_system_prompt(language: str) -> str:
    """Gibt den Systemprompt für die ausgewählte Sprache zurück."""
    return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["de"])


def format_rag_metadata(model: str, *, temperature: float) -> str:
    """Erstellt eine Metadatenzeile für die RAG-Antwort."""
    target = registry.resolve(model)
    metadata = [
        f"Model: {target.label}",
        f"Provider: {target.provider.value}",
        f"Base URL: {target.provider.base_url}",
        f"Temperature: {temperature:g}",
        f"Embedding model: {rag_lib.EMBEDDING_MODELS[0]}",
        f"Vector database: Qdrant",
        (
            "Chunking method: "
            f"{rag_lib.DEFAULT_CHUNK_SIZE} chars, "
            f"{rag_lib.DEFAULT_CHUNK_OVERLAP} overlap, recursive"
        ),
    ]
    return "\n".join(metadata)


def clear_file_output() -> tuple[str, gr.update]:
    """Löscht die Dateiverarbeitungsausgabe und setzt den Upload zurück."""
    return "", gr.update(value=None)


def inspect_file(file_input) -> str:
    """Lädt, bereinigt und chunkt das ausgewählte Dokument."""
    if file_input is None:
        return "Bitte wählen Sie eine Datei aus."

    file_path = getattr(file_input, "path", None) or str(file_input)
    if not file_path:
        return "Die ausgewählte Datei ist ungültig."

    try:
        loaded = load_documents([str(file_path)])
        cleaned, _ = clean_documents(loaded)
        chunks, _ = split_documents(cleaned)
        return chunks_to_markdown(chunks)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return f"Dateiverarbeitung fehlgeschlagen: {exc}"


def ask_rag(
    question: str,
    language: str,
    model: str,
    temperature: float,
) -> str:
    """Fragt das dokumentenspezifische RAG-System in der gewählten Sprache."""
    if not question or not question.strip():
        return "Bitte geben Sie eine Frage ein."

    system_prompt = get_system_prompt(language)
    try:
        api_key = rag_lib.openrouter_api_key()
        pipeline = rag_lib.run_pipeline(language=language, fmt="pdf")
        embeddings = rag_lib.build_embeddings(
            "openai/text-embedding-3-small",
            api_key,
        )
        store = rag_lib.build_vectorstore(
            pipeline.chunks,
            embeddings,
            retrieval_mode=rag_lib.RetrievalMode.DENSE,
        )
        retriever = rag_lib.build_retriever(store)
        llm = rag_lib.build_llm(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        chain = (
            {"context": retriever | rag_lib.format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)
        return (
            f"{answer}\n\n"
            f"{format_rag_metadata(model, temperature=temperature)}"
        )
    except (rag_lib.MissingCredentialsError, rag_lib.RagError, ValueError) as exc:
        return f"RAG-Fehler: {exc}"


with gr.Blocks() as rag_blocks:
    with gr.Sidebar():
        gr.Markdown("### ⚙️ Einstellungen")
        language = gr.Radio(
            choices=[("Deutsch", "de"), ("English", "en")],
            value="de",
            label="Sprache",
        )
        prompt_in = gr.Textbox(
            value=get_system_prompt("de"),
            label="System Prompt",
            lines=4,
        )
        language.change(
            lambda selected: get_system_prompt(selected),
            inputs=language,
            outputs=prompt_in,
        )
        model = gr.Dropdown(
            label="Model",
            choices=MODEL_CHOICES,
            value=MODEL_CHOICES[0][1],
        )
        temperature = gr.Slider(
            label="Temperature",
            minimum=0.0,
            maximum=1.0,
            value=get_settings().temperature,
            step=0.1,
        )

    with gr.Tab("RAG build"):
        file_input = gr.File(label="Textdatei auswählen", file_types=[".pdf", ".txt"])
        file_button = gr.Button("Inspect file")
        file_output = gr.Markdown()
        file_button.click(inspect_file, inputs=file_input, outputs=file_output)

        clear_button = gr.Button("Reset", variant="secondary")
        clear_button.click(
            clear_file_output,
            outputs=[file_output, file_input],
            js="window.location.reload()",
        )

        msg_in = gr.Textbox(label="Ihre Nachricht")
        rag_button = gr.Button("Send to RAG")
        rag_output = gr.Markdown()
        rag_button.click(
            ask_rag,
            inputs=[msg_in, language, model, temperature],
            outputs=rag_output,
        )

    with gr.Tab("RAG eval"):
        gr.Markdown("Die Evaluierung wird später implementiert.")


def run() -> None:
    """Startet die Gradio-Anwendung und öffnet sie im Browser."""
    rag_blocks.launch(inbrowser=True, share=False)


def main() -> None:
    """Entry-Point für den uv-Skriptbefehl."""
    run()


if __name__ == "__main__":
    run()
