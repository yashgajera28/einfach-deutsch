"""Public preprocessing API for the German text simplification system."""

from src.preprocessing.aligner import align_sentences
from src.preprocessing.cleaner import clean_html, clean_wiki_markup
from src.preprocessing.quality import (
    apply_quality_filters,
    filter_identical,
    filter_length_ratio,
    filter_max_length,
    filter_non_german,
)
from src.preprocessing.spacy_pipe import entities, extract_legal_references, get_nlp, sentences

__all__ = [
    "align_sentences",
    "clean_html",
    "clean_wiki_markup",
    "entities",
    "extract_legal_references",
    "get_nlp",
    "sentences",
    "filter_identical",
    "filter_non_german",
    "filter_length_ratio",
    "filter_max_length",
    "apply_quality_filters",
]
