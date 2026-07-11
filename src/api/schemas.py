"""Pydantic request/response models for the simplification API."""

from __future__ import annotations

import warnings
from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    warnings.warn(f"pydantic is not installed: {exc}", stacklevel=2)

    class BaseModel:  # type: ignore[no-redef]
        """Fallback base class when pydantic is unavailable."""

        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        """Fallback Field stub when pydantic is unavailable."""
        return None


class ReadabilityMetrics(BaseModel):
    """Container for LIX and Wiener Sachtextformel scores."""

    lix: float = Field(..., description="LIX readability score")
    wstf: float = Field(..., description="Wiener Sachtextformel readability score")


class SimplifyRequest(BaseModel):
    """Request body for text simplification."""

    text: str = Field(..., description="German text to simplify")
    target_level: str = Field(default="B1", description="Target language level (A2, B1, B2)")
    preserve_entities: bool = Field(
        default=True, description="Re-inject named entities dropped during simplification"
    )
    explain: bool = Field(default=False, description="Include rule-based change explanations")


class SimplifyResponse(BaseModel):
    """Response body for a single simplification result."""

    simplified: str = Field(..., description="Simplified German text")
    readability_before: ReadabilityMetrics = Field(..., description="Readability before simplification")
    readability_after: ReadabilityMetrics = Field(..., description="Readability after simplification")
    confidence: float = Field(..., description="Heuristic confidence score in [0, 1]")
    level: str = Field(..., description="Target language level")
    backend: str = Field(..., description="Model backend used for simplification")
    entities_preserved: list[str] = Field(default_factory=list, description="Named entities preserved")
    explanation: list[str] | None = Field(default=None, description="Rule-based explanations for changes")


class PdfSimplifyResponse(BaseModel):
    """Response body for per-page PDF simplification."""

    pages: list[SimplifyResponse] = Field(..., description="Simplification result per page")
    combined_text: str = Field(..., description="Concatenated original text")
    combined_simplified: str = Field(..., description="Concatenated simplified text")


class FileSimplifyResponse(BaseModel):
    """Response body for DOCX/TXT file simplification."""

    source_type: str = Field(..., description="Original file extension (docx, txt, md)")
    original_text: str = Field(..., description="Extracted original text")
    simplified: str = Field(..., description="Simplified text")
    readability_before: ReadabilityMetrics = Field(..., description="Readability before simplification")
    readability_after: ReadabilityMetrics = Field(..., description="Readability after simplification")
    confidence: float = Field(..., description="Heuristic confidence score in [0, 1]")
    level: str = Field(..., description="Target language level")
    backend: str = Field(..., description="Model backend used for simplification")
    entities_preserved: list[str] = Field(default_factory=list, description="Named entities preserved")
    explanation: list[str] | None = Field(default=None, description="Rule-based explanations for changes")


class EvaluateRequest(BaseModel):
    """Request body for metric evaluation."""

    source: str = Field(..., description="Original complex text")
    prediction: str = Field(..., description="Simplified prediction")
    reference: list[str] = Field(..., description="Reference simplifications")


class EvaluateResponse(BaseModel):
    """Response body for evaluation metrics."""

    sari: float = Field(..., description="SARI score")
    bleu: float = Field(..., description="Sentence-level BLEU score")
    readability_before: ReadabilityMetrics = Field(..., description="Readability of source")
    readability_after: ReadabilityMetrics = Field(..., description="Readability of prediction")


class ModelInfoResponse(BaseModel):
    """Response body for model metadata."""

    name: str = Field(..., description="Model name or path")
    training_date: str | None = Field(default=None, description="Training date if known")
    dataset_size: int | None = Field(default=None, description="Training dataset size if known")
    backend: str = Field(..., description="Model backend")


class HealthResponse(BaseModel):
    """Response body for health checks."""

    status: str = Field(..., description="ok or degraded")
