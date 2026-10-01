"""Data preparation pipeline modules."""

from rag_project.pipeline.loading import (
    LoaderDependencyError,
    LoadReport,
    SourceNotFoundError,
    UnsupportedFormatError,
    detect_format,
    detect_source_kind,
    format_load_report,
    load_documents,
)
from rag_project.pipeline.cleaning import (
    CleaningStats,
    clean_documents,
    format_cleaning_report,
    format_cleaning_stats,
    join_cleaned_text,
)
from rag_project.pipeline.chunking import (
    ChunkingStats,
    build_splitter,
    compare_strategies,
    format_chunk_preview,
    format_chunking_stats,
    split_documents,
)

__all__ = [
    "LoadReport",
    "LoaderDependencyError",
    "SourceNotFoundError",
    "UnsupportedFormatError",
    "detect_format",
    "detect_source_kind",
    "format_load_report",
    "load_documents",
    "CleaningStats",
    "clean_documents",
    "format_cleaning_report",
    "format_cleaning_stats",
    "join_cleaned_text",
    "ChunkingStats",
    "build_splitter",
    "compare_strategies",
    "format_chunk_preview",
    "format_chunking_stats",
    "split_documents",
]
