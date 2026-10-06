"""RAG-Pipeline auf Basis der Projektstufen: laden -> bereinigen -> chunken.

Die drei Vorverarbeitungsstufen kommen aus dem Projekt, nicht aus diesem
Modul:

    laden     :mod:`rag_project.pipeline.loading`   ``load_documents``
    bereinigen:mod:`rag_project.pipeline.cleaning`  ``clean_documents``
    chunken   :mod:`rag_project.pipeline.chunking`  ``split_documents``

Hier liegt nur, was das Projekt nicht hat: der Weg von den Chunks in einen
Qdrant-Store und die RAG-Kette darueber.

Sprache: ``--language de|en`` waehlt das Quelldokument. Deutsch ist die
Werkseinstellung. Jede Sprache hat ihren eigenen strengen System-Prompt.

Keine Evaluierung: keine Judges, keine Metriken.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import wrap
from typing import Any, Iterator

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from pydantic import SecretStr

from rag_project.core.config import (
    DEFAULT_DOCS_DIR,
    DEFAULT_DOCUMENT_NAME,
    PROJECT_ROOT,
    Settings,
    get_settings,
)

# --- Stufen aus dem Projekt -------------------------------------------------
from rag_project.core.paths import DocumentPathError, resolve_documents
from rag_project.pipeline.chunking import (
    HANDBOOK_SEPARATORS,
    ChunkingStats,
    split_documents,
)
from rag_project.pipeline.cleaning import (
    CleaningStats,
    clean_documents,
    format_cleaning_stats,
)
from rag_project.pipeline.loading import (
    LoaderDependencyError,
    UnsupportedFormatError,
    describe,
    format_load_report,
    load_documents,
)

# ---------------------------------------------------------------------------
# Pfade und Sprache
# ---------------------------------------------------------------------------

DOCS_DIR = PROJECT_ROOT / DEFAULT_DOCS_DIR

DEFAULT_LANGUAGE = "de"
LANGUAGES: tuple[str, ...] = ("de", "en")
LANGUAGE_LABELS: dict[str, str] = {"de": "Deutsch", "en": "English"}

#: Queldokumente je Sprache. Beide Formate stehen zur Wahl; das PDF ist die
#: Werkseinstellung, die Textfassung daneben enthaelt denselben Inhalt.
SOURCE_FILES: dict[str, dict[str, str]] = {
    "de": {"pdf": "handbuch.pdf", "txt": "handbuch.txt"},
    "en": {"pdf": "handbook.pdf", "txt": "handbook.txt"},
}

# ---------------------------------------------------------------------------
# Modelle
# ---------------------------------------------------------------------------

#: Die beiden Embedding-Modelle des Vergleichs.
EMBEDDING_MODELS: list[str] = [
    "openai/text-embedding-3-small",
    "nvidia/llama-nemotron-embed-vl-1b-v2",
]

#: Kandidaten, die als Embedding-Modell ausscheiden, mit Begruendung.
REJECTED_EMBEDDING_MODELS: dict[str, str] = {
    "ibm-granite/granite-4.1-8b": (
        "Chat-Modell ohne Embedding-Ausgabe -- als LLM nutzbar, nicht als "
        "Embedding-Modell."
    ),
}

#: Sparse-Modell der hybriden Suche.
SPARSE_MODEL = "Qdrant/bm25"

#: Generierungs-LLM: Granite ohne Reasoning.
GENERATION_MODEL = "ibm-granite/granite-4.2-8b"
GENERATION_EFFORT = "none"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Prompt und Fragen
# ---------------------------------------------------------------------------

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

NOT_FOUND_MARKERS: dict[str, str] = {
    "de": "nicht in meinen Dokumenten enthalten",
    "en": "not in my documents",
}

TEST_QUERIES: dict[str, list[str]] = {
    "de": [
        "Wie lange ist der Link zum Zurücksetzen des Passworts gültig?",
        "Nach wie vielen fehlgeschlagenen Anmeldeversuchen wird das Konto gesperrt?",
        "Welche maximale Dateigröße gilt für hochgeladene Dokumente?",
        "Wie viele Urlaubstage stehen mir pro Jahr zu?",
    ],
    "en": [
        "How long is the password reset link valid?",
        "After how many failed sign-in attempts is the account locked?",
        "What is the maximum file size for uploaded documents?",
        "How many vacation days am I entitled to per year?",
    ],
}

DEFAULT_K = 3
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# ---------------------------------------------------------------------------
# Fehler
# ---------------------------------------------------------------------------

class RagError(RuntimeError):
    """Basisfehler der RAG-Schicht -- von der CLI als Meldung ausgegeben."""


class MissingCredentialsError(RagError):
    """Der OpenRouter-Key fehlt."""


class IndexBuildError(RagError):
    """Der Vectorstore liess sich nicht aufbauen."""


# ---------------------------------------------------------------------------
# Sprache
# ---------------------------------------------------------------------------

def check_language(language: str | None) -> str:
    """Prueft die Sprachkennung und gibt sie normalisiert zurueck."""
    value = (language or DEFAULT_LANGUAGE).strip().lower()
    if value not in SOURCE_FILES:
        raise RagError(
            f"Unbekannte Sprache {language!r}. Verfuegbar: {', '.join(LANGUAGES)}"
        )
    return value


def source_path(language: str, *, fmt: str = "pdf") -> Path:
    """Pfad des Quelldokuments einer Sprache (``pdf`` oder ``txt``)."""
    language = check_language(language)
    if fmt not in SOURCE_FILES[language]:
        raise RagError(f"Unbekanntes Format {fmt!r}. Verfuegbar: pdf, txt")
    settings = get_settings()
    return (settings.docs_path / SOURCE_FILES[language][fmt]).resolve()


# ---------------------------------------------------------------------------
# Stufe 1: laden
# ---------------------------------------------------------------------------

def stage_load(language: str = DEFAULT_LANGUAGE, *, fmt: str = "pdf", on_event=None):
    """Stufe 1 -- laden ueber die Projekt-Bibliothek.

    ``load_documents`` erkennt das Format an der Endung und nimmt fuer ``.pdf``
    den ``PyPDFLoader``, fuer ``.txt`` den ``TextLoader``. Die Metadaten
    (``source``, ``file_name``, ``format``, ``size_bytes``, ``page_index``)
    setzt die Funktion selbst.

    :raises DocumentPathError: Quelldatei fehlt.
    :raises UnsupportedFormatError: Endung wird nicht unterstuetzt.
    :raises LoaderDependencyError: ein Loader-Paket fehlt.
    """
    language = check_language(language)
    paths = resolve_documents([source_path(language, fmt=fmt)])
    return load_documents(paths, on_event=on_event)


# ---------------------------------------------------------------------------
# Stufe 2: bereinigen
# ---------------------------------------------------------------------------

def stage_clean(documents, *, on_event=None, **options):
    """Stufe 2 -- bereinigen ueber die Projekt-Bibliothek.

    ``clean_documents`` entfernt Steuerzeichen, normalisiert Unicode, loest
    Silbentrennung auf, vereinheitlicht Aufzaehlungszeichen und normalisiert
    Leerraum. Je Dokument wandern ``chars_before``, ``chars_after`` und
    ``cleaned`` in die Metadaten.

    :return: ``(documents, CleaningStats)``
    """
    return clean_documents(documents, on_event=on_event, **options)


# ---------------------------------------------------------------------------
# Stufe 3: chunken
# ---------------------------------------------------------------------------

def stage_chunk(
    documents,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: str = "recursive",
    handbook: bool = True,
    on_event=None,
) -> tuple[list[Document], ChunkingStats]:
    """Stufe 3 -- chunken ueber die Projekt-Bibliothek.

    ``split_documents`` zerlegt die bereinigten Dokumente und ergaenzt je Chunk
    ``chunk_index``, ``chunk_chars``, ``chapter``, ``section`` und ``title``.
    ``handbook=True`` nimmt die Trennzeichen-Reihenfolge fuer Handbuecher, bei
    der die Kapitelueberschrift vor dem Absatzumbruch greift.

    :return: ``(chunks, ChunkingStats)``
    """
    separators = HANDBOOK_SEPARATORS if handbook else None
    return split_documents(
        documents,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        on_event=on_event,
    )


# ---------------------------------------------------------------------------
# Stufe 4: indexieren
# ---------------------------------------------------------------------------

def openrouter_api_key() -> SecretStr:
    """Liest den OpenRouter-Key aus der Umgebung (``.env`` via load_dotenv)."""
    load_dotenv()
    raw = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not raw:
        raise MissingCredentialsError(
            "OPENROUTER_API_KEY ist nicht gesetzt. Lege den Key in der .env an "
            "oder exportiere ihn in der Umgebung."
        )
    return SecretStr(raw)


def build_embeddings(model_name: str, api_key: SecretStr) -> OpenAIEmbeddings:
    """Ein Embedding-Client je Modell, beide ueber den OpenRouter-Endpunkt."""
    if model_name in REJECTED_EMBEDDING_MODELS:
        raise RagError(
            f"{model_name!r} ist kein Embedding-Modell: "
            f"{REJECTED_EMBEDDING_MODELS[model_name]} "
            f"Verfuegbar: {', '.join(EMBEDDING_MODELS)}"
        )
    return OpenAIEmbeddings(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        model=model_name,
    )


def build_vectorstore(
    chunks: list[Document],
    embeddings: OpenAIEmbeddings,
    *,
    retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
    collection_name: str | None = None,
) -> QdrantVectorStore:
    """Baut einen frischen In-Memory-Store aus den Chunks.

    ``location=":memory:"`` -- der Index lebt nur im Prozess. Fuer die hybride
    Suche kommt das Sparse-Modell dazu; ohne es gibt es keine BM25-Komponente
    und Qdrant lehnt den Modus ab.

    :raises IndexBuildError: wenn der Store nicht aufgebaut werden konnte.
    """
    common: dict[str, Any] = {
        "documents": chunks,
        "embedding": embeddings,
        "collection_name": collection_name or f"handbuch_{retrieval_mode.value}",
        "location": ":memory:",
        "retrieval_mode": retrieval_mode,
    }
    try:
        if retrieval_mode in (RetrievalMode.SPARSE, RetrievalMode.HYBRID):
            return QdrantVectorStore.from_documents(
                sparse_embedding=FastEmbedSparse(model_name=SPARSE_MODEL),
                **common,
            )
        return QdrantVectorStore.from_documents(**common)
    except Exception as exc:
        raise IndexBuildError(
            f"Vectorstore ({retrieval_mode.value}) nicht aufgebaut: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Stufe 5: suchen und antworten
# ---------------------------------------------------------------------------

def format_docs(documents) -> str:
    """Fuegt Treffer zu einem Kontextblock zusammen -- mit Herkunft.

    Die Herkunftszeile nennt Kapitel und Abschnitt aus den Chunk-Metadaten.
    Damit ist im Prompt sichtbar, woher ein Treffer stammt, auch wenn die
    Ueberschrift im Text keine Nummer mehr traegt.
    """
    blocks = []
    for doc in documents:
        meta = doc.metadata
        chapter = meta.get("chapter") or "?"
        section = meta.get("section")
        title = meta.get("title") or ""
        origin = f"[Kapitel {chapter}"
        if section:
            origin += f", Abschnitt {section}"
        origin += f": {title}]"
        blocks.append(f"{origin}\n{doc.page_content}")
    return "\n\n".join(blocks)


#: A line that starts a bullet ("- ", "* ", "+ ", "• ") or numbered ("1. ",
#: "2) ") list item, with optional leading indentation.
_LIST_ITEM_RE = re.compile(r"^(\s*)([-*+•]\s+|\d{1,3}[.)]\s+)")


def format_text(text, width=80, indent=0):
    """Wrap long lines while preserving their original structure.

    Plain ``print`` can produce lines that are too long for slides. This helper
    wraps line by line so blank lines, lists, fenced code blocks and tables keep
    their structure.
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
    """Print text via :func:`format_text` while preserving its structure."""
    print(format_text(text, width=width, indent=indent))


def build_retriever(store: QdrantVectorStore, k: int = DEFAULT_K):
    """Retriever auf einem bestehenden Store."""
    return store.as_retriever(search_kwargs={"k": k})


def build_retrieval_chain(store: QdrantVectorStore, k: int = DEFAULT_K):
    """Retriever plus Formatierung: die Kontextstufe der RAG-Kette."""
    return build_retriever(store, k) | format_docs


def build_llm(api_key: SecretStr, *, model: str = GENERATION_MODEL,
              temperature: float = 0.0):
    """Das Generierungs-LLM -- ohne Reasoning."""
    return ChatOpenRouter(
        api_key=api_key,
        model=model,
        temperature=temperature,
        reasoning={"effort": GENERATION_EFFORT},
    )


def build_rag_chain(
    store: QdrantVectorStore,
    llm,
    k: int = DEFAULT_K,
    *,
    language: str = DEFAULT_LANGUAGE,
    retrieval_chain=None,
):
    """Die vollstaendige RAG-Kette.

    ``{"context": retrieval_chain, "input": RunnablePassthrough()}`` schiebt die
    Frage an zwei Stellen weiter: einmal durch den Retriever, einmal
    unveraendert in den Prompt.
    """
    language = check_language(language)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPTS[language]),
            ("human", "{input}"),
        ]
    )
    retrieval_chain = retrieval_chain or build_retrieval_chain(store, k)
    return (
        {"context": retrieval_chain, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# ---------------------------------------------------------------------------
# Buendel
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PipelineResult:
    """Was die drei Vorverarbeitungsstufen geliefert haben.

    ``loaded`` sind die Seiten-Documents des Loaders, ``cleaned`` dieselben
    nach der Bereinigung, ``chunks`` das Ergebnis des Splitters. Die drei
    Zaehler und die Kennzahlen kommen direkt aus den Projekt-Modulen.
    """

    language: str
    format: str
    loaded: list[Document] = field(repr=False)
    cleaned: list[Document] = field(repr=False)
    chunks: list[Document] = field(repr=False)
    cleaning_stats: CleaningStats
    chunking_stats: ChunkingStats

    @property
    def load_report(self) -> str:
        """Die Tabelle aus dem loading-Modul."""
        return format_load_report(describe(self.loaded))

    @property
    def cleaning_report(self) -> str:
        """Die Bilanz aus dem cleaning-Modul."""
        return format_cleaning_stats(self.cleaning_stats)


@dataclass(slots=True)
class RagSystem:
    """Ein fertiges System: Store, Retriever und Kette in einem Buendel."""

    language: str
    mode: RetrievalMode
    embeddings_model: str
    pipeline: PipelineResult = field(repr=False)
    vectorstore: QdrantVectorStore = field(repr=False)
    retriever: Any = field(repr=False)
    retrieval_chain: Any = field(repr=False)
    rag_chain: Any = field(repr=False)

    @property
    def label(self) -> str:
        return f"{LANGUAGE_LABELS[self.language]} | {self.mode.value}"

    def retrieve(self, question: str) -> list[Document]:
        """Die rohen Treffer -- ohne LLM-Aufruf."""
        return self.retriever.invoke(question)

    def context(self, question: str) -> str:
        """Der aufbereitete Kontext -- ohne LLM-Aufruf."""
        return self.retrieval_chain.invoke(question)

    def answer(self, question: str) -> str:
        """Die fertige Antwort der Kette."""
        return self.rag_chain.invoke(question)

    def is_honest(self, answer: str) -> bool:
        """Hat die Antwort die Grenze des Korpus benannt?"""
        return NOT_FOUND_MARKERS[self.language] in answer


def run_pipeline(
    *,
    language: str = DEFAULT_LANGUAGE,
    fmt: str = "pdf",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: str = "recursive",
    handbook: bool = True,
    on_event=None,
    **cleaning_options: Any,
) -> PipelineResult:
    """Faehrt die drei Projektstufen nacheinander: laden, bereinigen, chunken."""
    language = check_language(language)

    loaded = stage_load(language, fmt=fmt, on_event=on_event)
    cleaned, cleaning_stats = stage_clean(
        loaded, on_event=on_event, **cleaning_options
    )
    chunks, chunking_stats = stage_chunk(
        cleaned,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
        handbook=handbook,
        on_event=on_event,
    )

    return PipelineResult(
        language=language,
        format=fmt,
        loaded=loaded,
        cleaned=cleaned,
        chunks=chunks,
        cleaning_stats=cleaning_stats,
        chunking_stats=chunking_stats,
    )


def create_rag_system(
    *,
    language: str = DEFAULT_LANGUAGE,
    mode: RetrievalMode = RetrievalMode.DENSE,
    embedding_model: str = EMBEDDING_MODELS[0],
    pipeline: PipelineResult | None = None,
    chunks: list[Document] | None = None,
    llm=None,
    k: int = DEFAULT_K,
    collection_name: str | None = None,
    api_key: SecretStr | None = None,
    **pipeline_options: Any,
) -> RagSystem:
    """Baut Stufe 4 und 5 auf einem fertigen Vorverarbeitungsergebnis.

    Ohne ``pipeline`` oder ``chunks`` laufen die drei Projektstufen hier
    noch einmal -- bequem fuer einen Einzelaufruf, unnoetig, wenn die CLI sie
    schon gefahren ist.
    """
    language = check_language(language)
    if pipeline is None:
        pipeline = run_pipeline(language=language, **pipeline_options)

    index_chunks = chunks if chunks is not None else pipeline.chunks
    key = api_key or openrouter_api_key()

    store = build_vectorstore(
        index_chunks,
        build_embeddings(embedding_model, key),
        retrieval_mode=mode,
        collection_name=collection_name or f"handbuch_{language}_{mode.value}",
    )
    retriever = build_retriever(store, k)
    retrieval_chain = retriever | format_docs
    rag_chain = build_rag_chain(
        store,
        llm if llm is not None else build_llm(key),
        k,
        language=language,
        retrieval_chain=retrieval_chain,
    )

    return RagSystem(
        language=language,
        mode=mode,
        embeddings_model=embedding_model,
        pipeline=pipeline,
        vectorstore=store,
        retriever=retriever,
        retrieval_chain=retrieval_chain,
        rag_chain=rag_chain,
    )


def iter_test_queries(language: str = DEFAULT_LANGUAGE) -> Iterator[str]:
    """Die Testfragen einer Sprache, der Reihe nach."""
    yield from TEST_QUERIES[check_language(language)]


__all__ = [
    "PROJECT_ROOT",
    "DOCS_DIR",
    "DEFAULT_LANGUAGE",
    "LANGUAGES",
    "LANGUAGE_LABELS",
    "SOURCE_FILES",
    "EMBEDDING_MODELS",
    "REJECTED_EMBEDDING_MODELS",
    "SPARSE_MODEL",
    "GENERATION_MODEL",
    "GENERATION_EFFORT",
    "OPENROUTER_BASE_URL",
    "SYSTEM_PROMPTS",
    "NOT_FOUND_MARKERS",
    "TEST_QUERIES",
    "DEFAULT_K",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "RagError",
    "MissingCredentialsError",
    "IndexBuildError",
    "DocumentPathError",
    "UnsupportedFormatError",
    "LoaderDependencyError",
    "check_language",
    "source_path",
    "stage_load",
    "stage_clean",
    "stage_chunk",
    "run_pipeline",
    "openrouter_api_key",
    "build_embeddings",
    "build_vectorstore",
    "format_docs",
    "build_retriever",
    "build_retrieval_chain",
    "build_llm",
    "build_rag_chain",
    "PipelineResult",
    "RagSystem",
    "create_rag_system",
    "iter_test_queries",
    "format_text",
    "print_wrapped",
]
