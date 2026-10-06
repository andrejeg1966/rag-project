"""LLM-Judges und Evaluierungshilfen für RAG-Antworten."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel, Field, SecretStr


def _install_quiet_async_client_cleanup():
    """Silence spurious ``Event loop is closed`` tracebacks from DeepEval runs.

    DeepEval metrics with ``async_mode=False`` (the way the evaluation decks
    use them) still send their LLM calls through a fresh ``AsyncOpenAI``
    client inside a short-lived ``asyncio.run()`` loop — and never close the
    client (``DeepEvalOpenAICompatibleModel._generate`` / ``_a_generate``
    in deepeval 4.0.x; reported upstream as
    https://github.com/confident-ai/deepeval/issues/3120 — this patch can go
    once that is fixed). When the garbage collector later
    finds such a client while a *different* event loop is running, the openai
    SDK's ``AsyncHttpxClientWrapper.__del__`` schedules ``aclose()`` on that
    loop; the client's connections, however, belong to the original,
    already-closed loop, so the cleanup task dies with ``RuntimeError: Event
    loop is closed`` — and asyncio prints a "Task exception was never
    retrieved" traceback into whatever notebook cell happens to be executing.

    The connections cannot be closed properly at that point anyway (their
    event loop is gone; the OS reclaims them when the objects are collected),
    so this patch replaces the destructor's fire-and-forget cleanup task with
    one that retrieves and drops that expected failure. Cleanups that *can*
    succeed still run exactly as before. Idempotent; a silent no-op if the
    openai SDK's private layout changes.
    """
    try:
        from openai import _base_client

        wrapper_cls = _base_client.AsyncHttpxClientWrapper
    except (ImportError, AttributeError):
        return
    if getattr(wrapper_cls, "_quiet_cleanup_installed", False):
        return

    async def _aclose_quietly(client):
        try:
            await client.aclose()
        except Exception:
            # Destructor-time cleanup; the expected failure is "Event loop is
            # closed" for connections owned by a finished asyncio.run() loop.
            # Nothing here is actionable, so drop it instead of letting
            # asyncio print an unretrieved-task traceback into the notebook.
            pass

    def _quiet_del(self):
        if self.is_closed:
            return
        try:
            asyncio.get_running_loop().create_task(_aclose_quietly(self))
        except Exception:
            pass

    wrapper_cls.__del__ = _quiet_del
    wrapper_cls._quiet_cleanup_installed = True


_install_quiet_async_client_cleanup()


#: Judges call ``with_structured_output``, and granite-4.2 loses schema
#: adherence when reasoning is fully off — measured on the ``Judgement``
#: schema: 11/12 valid with ``"none"`` vs 12/12 with ``"minimal"``, which is
#: also ~2.7x faster than both ``"none"`` and the reasoning-on default.
_JUDGE_EFFORT = "minimal"


def _make_llm(api_key, model, temperature):
    return ChatOpenRouter(
        api_key=api_key,
        model=model,
        temperature=temperature,
        reasoning={"effort": _JUDGE_EFFORT},
    )


class Judgement(BaseModel):
    """A judge's verdict: a binary decision plus a one-sentence justification."""

    verdict: bool = Field(description="True if the criterion is satisfied")
    explanation: str = Field(description="One sentence justifying the verdict")


#: Judge prompt: is every claim in the answer supported by the context?
#: (The English judge-prompt word for faithfulness is "grounded".)
FAITHFULNESS_PROMPT = """\
You check whether an answer is grounded in the given context.
The answer must only use information found in or reasonably inferred from the context.
If it contains any claim not supported by the context, the verdict is False.

Context:
{context}

Answer:
{answer}"""

#: Judge prompt: does the answer address the question that was asked?
RELEVANCE_PROMPT = """\
You check whether an answer addresses the question that was asked.
A relevant answer responds to the question, even if it turns out to be wrong.
An answer about a different topic has the verdict False.

Question:
{question}

Answer:
{answer}"""

#: Judge prompt: is the answer consistent with the reference answer?
CORRECTNESS_PROMPT = """\
You compare a generated answer with a reference answer.
The verdict is True if the generated answer is consistent with the reference.
Minor wording differences are acceptable.
If the reference says the information is not available, the generated answer
should indicate the same.

Question:
{question}

Reference answer:
{reference}

Generated answer:
{answer}"""


def make_judges(judge_llm):
    """Bind the three evaluation judges to ``judge_llm``.

    The ``llm_as_judge`` decks build these judges cell by cell; later decks
    import them from here so the prompts and the :class:`Judgement` schema
    stay in one canonical place.

    Returns the triple
    ``(judge_faithfulness, judge_relevance, judge_correctness)``, each a
    function returning a :class:`Judgement`.
    """
    judge = judge_llm.with_structured_output(Judgement)

    def judge_faithfulness(answer, context):
        return judge.invoke(
            FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        )

    def judge_relevance(question, answer):
        return judge.invoke(
            RELEVANCE_PROMPT.format(question=question, answer=answer)
        )

    def judge_correctness(question, reference, answer):
        return judge.invoke(
            CORRECTNESS_PROMPT.format(
                question=question, reference=reference, answer=answer
            )
        )

    return judge_faithfulness, judge_relevance, judge_correctness


def create_judge_llm(model, temperature=0, api_key=None):
    """Build a standalone judge LLM for a given model name.

    Swapping the judge model is a one-liner with this helper — a different
    model sidesteps self-preference and gives an independent view. ``api_key``
    defaults to the ``OPENROUTER_API_KEY`` environment variable.
    """
    load_dotenv()
    if api_key is None:
        api_key = SecretStr(os.getenv("OPENROUTER_API_KEY", ""))
        if not api_key.get_secret_value():
            raise RuntimeError("OPENROUTER_API_KEY is not set")
    return _make_llm(api_key, model, temperature)


__all__ = [
    "Judgement",
    "FAITHFULNESS_PROMPT",
    "RELEVANCE_PROMPT",
    "CORRECTNESS_PROMPT",
    "make_judges",
    "create_judge_llm",
]
