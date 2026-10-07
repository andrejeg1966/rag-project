from __future__ import annotations

from collections.abc import Iterator, Sequence

import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from rag_project.core.config import get_settings
from rag_project.llm.providers import build_llm, resolve_target

ASSISTANT_PROMPT = (
    "You are a friendly and helpful AI chatbot that responds to users' questions."
    "Keep the answer short and to the point."
)
GRUMPY_EXPERT_PROMPT = (
    "You are a grumpy and sarcastic AI expert who answers questions with a bad attitude."
)
PIRATE_PROMPT = "You are a pirate who answers questions in pirate speak and loves to use nautical metaphors."

PROMPT_CHOICES = [
    ("Assistant", ASSISTANT_PROMPT),
    ("Grumpy Expert", GRUMPY_EXPERT_PROMPT),
    ("Pirate", PIRATE_PROMPT),
]

MODEL_CHOICES = [
    ("OpenRouter Fast", "openrouter:deepseek/deepseek-v4-flash"),
    ("OpenRouter Balanced", "openrouter:gpt-5.6-luna"),
    ("OpenAI Fast", "openai:gpt-4o-mini"),
    ("OpenAI Balanced", "openai:gpt-4o"),
    ("OpenAI Reasoning", "openai:GPT-5 Pro"),
    ("OpenAI FLAGSHIP", "openai:GPT-5.6 Luna"),
]


def _to_langchain_messages(history: Sequence[object]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
            continue

        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            continue

        human, assistant = item
        messages.append(HumanMessage(content=human))
        if assistant:
            messages.append(AIMessage(content=assistant))
    return messages


def _as_text(message: object) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


class StreamingWebChatbot:
    """Streaming-Chatbot mit zentral aufgeloesten Provider-Modell-Konfigurationen."""

    def __init__(self, model: str | None = None, temperature: float | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.chat_model or self.default_model
        self.temperature = (
            temperature if temperature is not None else settings.temperature
        )
        self.llm = build_llm(self.model, streaming=True, temperature=self.temperature)
        self.history = [SystemMessage(content=ASSISTANT_PROMPT)]

    @property
    def default_model(self) -> str:
        return resolve_target().model.qualified_id

    def respond(
        self,
        message: str,
        history: Sequence[Sequence[str]],
        system_prompt: str,
        model: str,
        temperature: float,
    ) -> Iterator[str]:
        target = resolve_target(model)
        llm = build_llm(model, streaming=True, temperature=temperature)
        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        messages.extend(_to_langchain_messages(history))
        messages.append(HumanMessage(content=message))

        answer_parts: list[str] = []
        for chunk in llm.stream(messages):
            text = _as_text(chunk)
            if text:
                answer_parts.append(text)
                yield "".join(answer_parts)

        answer_text = "".join(answer_parts)
        yield (
            f"{answer_text}\n\n"
            f"Model: {target.model.label}\n"
            f"Provider: {target.provider.value}\n"
            f"Base URL: {target.base_url}\n"
            f"Temperature: {temperature:g}"
        )

    def reset(self, system_prompt: str | None = None) -> None:
        self.history = [SystemMessage(content=system_prompt or ASSISTANT_PROMPT)]


with gr.Blocks() as blocks_app:
    with gr.Sidebar():
        gr.Markdown("### ⚙️ Einstellungen")
        system_prompt = gr.Dropdown(
            label="System Prompt",
            choices=PROMPT_CHOICES,
            value=ASSISTANT_PROMPT,
        )
        model = gr.Dropdown(
            label="Model",
            choices=MODEL_CHOICES,
            value=MODEL_CHOICES[0][1],
        )
        temperature = gr.Slider(
            label="Temperature",
            minimum=0.0,
            maximum=2.0,
            value=get_settings().temperature,
            step=0.05,
        )

    chatbot = StreamingWebChatbot()
    chat_history = gr.Chatbot(
        value=[],
        render_markdown=True,
    )
    reset_button = gr.Button("Reset Chat", variant="secondary")
    reset_button.click(list, outputs=chat_history)

    gr.ChatInterface(
        fn=chatbot.respond,
        chatbot=chat_history,
        title="Streaming Web Chatbot",
        additional_inputs=[system_prompt, model, temperature],
    )


def run() -> None:
    """Startet die Gradio-Anwendung und öffnet sie im Browser."""
    app = blocks_app
    app.launch(inbrowser=True, share=False)


def main() -> None:
    """Entry-Point für den uv-Skriptbefehl."""
    run()


if __name__ == "__main__":
    run()