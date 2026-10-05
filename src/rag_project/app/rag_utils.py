"""Shared helpers to build the small ML-knowledge-base RAG system.

This module backs the Week-14/15 RAG-evaluation decks. The first deck
(``rag_evaluation_intro``) builds the system cell by cell to review the
construction; the later decks import ``create_rag_system`` from here so the
setup stays in one canonical place. The same pattern applies to the three
LLM judges: the ``llm_as_judge`` decks build them cell by cell, and
``running_the_evaluation`` (and later decks) import them via
:func:`make_judges`.

Each call builds a fresh, self-contained RAG system (new in-memory vector
store, retriever, and chain), so it is safe to call from independently
evaluated notebooks and to call twice with different parameters (e.g. a
dense-only store for comparison).

The module also provides :func:`format_text` / :func:`print_wrapped` to
display long texts (documents, LLM answers) without truncated lines while
keeping the structure of the reply (lists, code blocks) intact.

Importing this module also installs a small cleanup patch that keeps
DeepEval runs from printing spurious ``RuntimeError: Event loop is closed``
tracebacks into notebook cells — see
:func:`_install_quiet_async_client_cleanup`.
"""

import asyncio
import os
import re
from textwrap import wrap

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
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


#: Default model used for the RAG LLM (and reused as a judge model elsewhere).
DEFAULT_MODEL = "ibm-granite/granite-4.2-8b"

#: The small ML knowledge base used throughout the evaluation week, per
#: language. These are exactly the documents the first deck
#: (``rag_evaluation_intro``) builds its RAG system from, cell by cell —
#: the German decks query a German corpus, the English decks an English one.
DEFAULT_DOCS = {
    "de": [
        """Linear Regression ist eine Methode des überwachten Lernens.
    Sie modelliert lineare Beziehungen zwischen Features und Zielvariable.
    Die Normalgleichung kann verwendet werden, um optimale Parameter zu finden.
    Mean Squared Error (MSE) wird als Loss-Funktion verwendet.""",
        """Neuronale Netze bestehen aus Schichten von Neuronen.
    Jedes Neuron berechnet eine gewichtete Summe und wendet eine Aktivierungsfunktion an.
    Training erfolgt durch Backpropagation und Gradient Descent.
    Typische Aktivierungsfunktionen sind ReLU und Sigmoid.""",
        """Overfitting tritt auf, wenn ein Modell die Trainingsdaten auswendig lernt.
    Regularisierung und Cross-Validation helfen, Overfitting zu vermeiden.
    Ein gutes Modell generalisiert auf neue, ungesehene Daten.
    Dropout ist eine weitere Technik gegen Overfitting bei Neuronalen Netzen.""",
        """Large Language Models sind sehr große neuronale Netze.
    Sie werden auf Milliarden von Wörtern trainiert.
    RAG hilft LLMs, präziser mit spezifischem Wissen zu antworten.
    Prompt Engineering ist wichtig für gute Ergebnisse mit LLMs.""",
        """Decision Trees teilen Daten anhand von Entscheidungsregeln auf.
    Random Forests kombinieren viele Decision Trees für bessere Ergebnisse.
    Feature Importance zeigt, welche Features am wichtigsten sind.
    Ensemble-Methoden sind oft genauer als einzelne Modelle.""",
    ],
    "en": [
        """Linear Regression is a supervised learning method.
    It models linear relationships between features and the target variable.
    The normal equation can be used to find optimal parameters.
    Mean Squared Error (MSE) is used as the loss function.""",
        """Neural Networks consist of layers of neurons.
    Each neuron computes a weighted sum and applies an activation function.
    Training happens through backpropagation and gradient descent.
    Typical activation functions are ReLU and Sigmoid.""",
        """Overfitting occurs when a model memorizes the training data.
    Regularization and cross-validation help avoid overfitting.
    A good model generalizes to new, unseen data.
    Dropout is another technique against overfitting in neural networks.""",
        """Large Language Models are very large neural networks.
    They are trained on billions of words.
    RAG helps LLMs answer more precisely with specific knowledge.
    Prompt engineering is important for good results with LLMs.""",
        """Decision Trees split data based on decision rules.
    Random Forests combine many decision trees for better results.
    Feature Importance shows which features matter most.
    Ensemble methods are often more accurate than single models.""",
    ],
}

#: The system prompts, per language: answer from the context only, admit
#: ignorance otherwise — again exactly as built in ``rag_evaluation_intro``.
SYSTEM_PROMPTS = {
    "de": (
        "Du bist ein hilfreicher Assistent. Nutze den folgenden Kontext, um die Frage zu beantworten. "
        "Wenn der Kontext keine relevanten Informationen enthält, sage: "
        "'Diese Information ist nicht in meinen Dokumenten enthalten.' "
        "Halte die Antwort präzise.\n\n"
        "Context: {context}"
    ),
    "en": (
        "You are a helpful assistant. Use the following context to answer the question. "
        "If the context doesn't contain relevant information, say: "
        "'This information is not in my documents.' "
        "Keep the answer concise.\n\n"
        "Context: {context}"
    ),
}


def format_docs(docs):
    """Join retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


#: A line that starts a bullet ("- ", "* ", "+ ", "• ") or numbered ("1. ",
#: "2) ") list item, with optional leading indentation.
_LIST_ITEM_RE = re.compile(r"^(\s*)([-*+•]\s+|\d{1,3}[.)]\s+)")


def format_text(text, width=80, indent=0):
    """Wrap long lines while keeping the structure of the text intact.

    Plain ``print`` shows long documents and LLM answers as over-long lines
    that get cut off on slides; ``textwrap.wrap`` on the whole text destroys
    the reply's formatting. This wrapper works line by line instead:

    - blank lines are preserved,
    - list items wrap with a hanging indent under their marker,
    - fenced code blocks (``` ... ```) and table lines (``| ... |``) are
      passed through unwrapped,
    - everything else wraps to ``width``, keeping its leading indentation.

    ``indent`` prefixes every output line with that many spaces (the prefix
    counts toward ``width``). Long unbreakable words (e.g. URLs) are not
    split. Returns the formatted text; see :func:`print_wrapped` to print it
    directly.
    """
    pad = " " * indent
    formatted_lines = []
    in_code_block = False
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            formatted_lines.append(pad + line.rstrip())
            continue
        if in_code_block or stripped.startswith("|"):
            formatted_lines.append(pad + line.rstrip())
            continue
        if not stripped:
            formatted_lines.append("")
            continue
        match = _LIST_ITEM_RE.match(line)
        if match:
            first = pad + match.group(1) + match.group(2)
            hanging = pad + match.group(1) + " " * len(match.group(2))
            body = line[match.end() :]
        else:
            leading_ws = line[: len(line) - len(line.lstrip())]
            first = hanging = pad + leading_ws
            body = stripped
        wrapped = wrap(
            body,
            width=width,
            initial_indent=first,
            subsequent_indent=hanging,
            break_long_words=False,
            break_on_hyphens=False,
        )
        formatted_lines.extend(wrapped or [first.rstrip()])
    return "\n".join(formatted_lines)


def print_wrapped(text, width=80, indent=0):
    """Print ``text`` via :func:`format_text` (structure-preserving wrap)."""
    print(format_text(text, width=width, indent=indent))


#: Reasoning effort for plain generation: off, so the first token is fast.
_GENERATION_EFFORT = "none"

#: Judges call ``with_structured_output``, and granite-4.2 loses schema
#: adherence when reasoning is fully off — measured on the ``Judgement``
#: schema: 11/12 valid with ``"none"`` vs 12/12 with ``"minimal"``, which is
#: also ~2.7x faster than both ``"none"`` and the reasoning-on default.
_JUDGE_EFFORT = "minimal"


def _make_llm(api_key, model, temperature, effort=_GENERATION_EFFORT):
    return ChatOpenRouter(
        api_key=api_key,
        model=model,
        temperature=temperature,
        reasoning={"effort": effort},
    )


def build_rag_chain(vectorstore, llm, k=2, language="en"):
    """Build a retriever and RAG chain on top of an existing vector store.

    The RAG system has two layers: the **vector store** (the index — every
    document is embedded when it is built, the expensive part) and the
    **retriever + chain** on top (a cheap query-time wrapper). The retrieval
    depth ``k`` lives in the retriever, so comparing different ``k`` values
    only needs this function — the store is never rebuilt and nothing is
    re-embedded. ``language`` selects the system prompt (``"de"`` or
    ``"en"``); it should match the language of the corpus and the questions.
    :func:`create_rag_system` uses this internally.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPTS[language]), ("human", "{input}")]
    )
    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return retriever, chain


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
    stay in one canonical place (the same pattern as
    :func:`create_rag_system`).

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
    return _make_llm(api_key, model, temperature, effort=_JUDGE_EFFORT)


def create_rag_system(
    k=2,
    retrieval_mode=RetrievalMode.HYBRID,
    docs_content=None,
    collection_name="ml_docs_rag",
    model=DEFAULT_MODEL,
    temperature=0,
    api_key=None,
    judge_model=None,
    language="en",
):
    """Build a fresh RAG system over the small ML knowledge base.

    Parameters
    ----------
    k:
        Number of documents the retriever returns.
    retrieval_mode:
        ``RetrievalMode.HYBRID`` (dense + BM25) by default; pass
        ``RetrievalMode.DENSE`` for a dense-only comparison.
    docs_content:
        The corpus texts. Defaults to ``DEFAULT_DOCS[language]``.
    language:
        ``"de"`` or ``"en"`` — selects the default corpus and the system
        prompt, so the whole system speaks the language of the questions.
    collection_name:
        Name of the in-memory Qdrant collection (unique per call when a
        notebook builds more than one store).
    model, temperature, api_key:
        LLM configuration. ``api_key`` defaults to the ``OPENROUTER_API_KEY``
        environment variable (loaded via ``.env``).
    judge_model:
        Optional model name for a **second**, independent LLM to use as a
        judge (sidesteps self-preference). When omitted the judge is the same
        model as the RAG LLM.

    Returns
    -------
    (retriever, chain, vectorstore, llm, judge_llm)
        The retriever (``search_kwargs={"k": k}``), the full RAG chain, the
        underlying vector store (pass it to :func:`build_rag_chain` to compare
        a different ``k`` without re-embedding), the RAG LLM, and the judge
        LLM (== ``llm`` when
        ``judge_model`` is not given). Both LLMs share ``temperature``.
    """
    load_dotenv()
    if api_key is None:
        api_key = SecretStr(os.getenv("OPENROUTER_API_KEY", ""))
        if not api_key.get_secret_value():
            raise RuntimeError("OPENROUTER_API_KEY is not set")

    retrieval_mode = RetrievalMode(retrieval_mode)

    llm = _make_llm(api_key, model, temperature)
    # The judge always gets its own client: it drives ``with_structured_output``
    # and needs _JUDGE_EFFORT even when it runs on the same model as ``llm``.
    judge_llm = _make_llm(
        api_key,
        judge_model if judge_model is not None else model,
        temperature,
        effort=_JUDGE_EFFORT,
    )

    if docs_content is None:
        docs_content = DEFAULT_DOCS[language]
    documents = [Document(page_content=d) for d in docs_content]

    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model="openai/text-embedding-3-small",
    )

    if retrieval_mode == RetrievalMode.HYBRID:
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        vectorstore = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            collection_name=collection_name,
            location=":memory:",
            retrieval_mode=RetrievalMode.HYBRID,
        )
    else:
        vectorstore = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            location=":memory:",
            retrieval_mode=retrieval_mode,
        )

    retriever, chain = build_rag_chain(vectorstore, llm, k=k, language=language)

    return retriever, chain, vectorstore, llm, judge_llm
