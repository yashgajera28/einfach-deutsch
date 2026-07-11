"""FastAPI backend for German text simplification."""

from __future__ import annotations

import asyncio
import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"

app: Any | None = None

_FastAPI: Any = None
_File: Any = None
_HTTPException: Any = None
_Request: Any = None
_UploadFile: Any = None
_RequestValidationError: Any = None
_JSONResponse: Any = None

_load_config: Any = None
_extract_text_from_pdf: Any = None

schemas: Any = None
_compute_sentence_bleu: Any = None
_readability_delta: Any = None
_compute_sari: Any = None

try:
    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    _FastAPI = FastAPI
    _File = File
    _HTTPException = HTTPException
    _Request = Request
    _UploadFile = UploadFile
    _RequestValidationError = RequestValidationError
    _JSONResponse = JSONResponse
except ImportError as exc:
    warnings.warn(f"fastapi is not installed: {exc}", stacklevel=2)

if _FastAPI is not None:
    try:
        from src.api import schemas as _schemas
        from src.api.extract import extract_text_from_pdf as _extract_text_from_pdf
        from src.evaluation.bleu import compute_sentence_bleu as _compute_sentence_bleu
        from src.evaluation.readability import readability_delta as _readability_delta
        from src.evaluation.sari import compute_sari as _compute_sari
        from src.utils.config import load_config as _load_config
    except ImportError as exc:
        warnings.warn(f"API dependencies are missing: {exc}", stacklevel=2)
        _FastAPI = None


if _FastAPI is not None:

    def _load_simplifier(config: dict[str, Any]) -> Any | None:
        """Load the simplifier when inference dependencies are present."""
        try:
            from src.models.simplifier import Simplifier

            simplifier = Simplifier(config, backend="lora")
            return simplifier if simplifier.model is not None else None
        except Exception as exc:
            logger.warning("Could not load simplifier: %s", exc)
            return None

    @asynccontextmanager
    async def lifespan(app: _FastAPI):
        """Load configuration, model, and concurrency limit on startup."""
        config = _load_config(CONFIG_PATH)
        app.state.config = config
        app.state.semaphore = asyncio.Semaphore(config["api"]["max_concurrent_requests"])
        app.state.simplifier = _load_simplifier(config)
        app.state.model_loaded = app.state.simplifier is not None
        yield
        app.state.simplifier = None
        app.state.model_loaded = False

    app = _FastAPI(
        title="Einfach Deutsch API",
        description="German text simplification backend.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(_RequestValidationError)
    async def validation_exception_handler(
        request: _Request,
        exc: _RequestValidationError,
    ) -> _JSONResponse:
        """Return structured validation errors."""
        return _JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(_HTTPException)
    async def http_exception_handler(
        request: _Request,
        exc: _HTTPException,
    ) -> _JSONResponse:
        """Return structured HTTP errors."""
        return _JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: _Request,
        exc: Exception,
    ) -> _JSONResponse:
        """Return a generic 500 error without leaking internals."""
        logger.exception("Unhandled error: %s", exc)
        return _JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/health", response_model=_schemas.HealthResponse)
    async def health() -> _schemas.HealthResponse:
        """Return service health status."""
        status = "ok" if app.state.model_loaded else "degraded"
        return _schemas.HealthResponse(status=status)

    @app.get("/model/info", response_model=_schemas.ModelInfoResponse)
    async def model_info() -> _schemas.ModelInfoResponse:
        """Return metadata about the loaded model."""
        simplifier = app.state.simplifier
        if simplifier is None:
            return _schemas.ModelInfoResponse(name="unknown", backend="unknown")
        return _schemas.ModelInfoResponse(
            name=simplifier.model_path,
            backend=simplifier.backend,
        )

    async def _simplify_text(
        text: str,
        level: str,
        preserve_entities: bool,
        explain: bool,
    ) -> dict[str, Any]:
        """Run simplification under the concurrency semaphore."""
        simplifier = app.state.simplifier
        if simplifier is None:
            raise _HTTPException(
                status_code=503,
                detail="Simplification model is not available",
            )

        async with app.state.semaphore:
            return await asyncio.to_thread(
                simplifier.simplify,
                text,
                level,
                preserve_entities,
                explain,
            )

    @app.post("/simplify", response_model=_schemas.SimplifyResponse)
    async def simplify(request: _schemas.SimplifyRequest) -> _schemas.SimplifyResponse:
        """Simplify a single German text."""
        result = await _simplify_text(
            request.text,
            request.target_level,
            request.preserve_entities,
            request.explain,
        )
        return _schemas.SimplifyResponse(**result)

    @app.post("/simplify/pdf", response_model=_schemas.PdfSimplifyResponse)
    async def simplify_pdf(file: _UploadFile = _File(...)) -> _schemas.PdfSimplifyResponse:
        """Simplify text extracted from an uploaded PDF file."""
        content = await file.read()
        pages = _extract_text_from_pdf(content)

        results: list[_schemas.SimplifyResponse] = []
        for page_num in sorted(pages):
            text = pages[page_num]
            if not text.strip():
                continue
            result = await _simplify_text(text, "B1", True, False)
            results.append(_schemas.SimplifyResponse(**result))

        combined_text = "\n".join(pages[num] for num in sorted(pages))
        combined_simplified = "\n".join(result.simplified for result in results)
        return _schemas.PdfSimplifyResponse(
            pages=results,
            combined_text=combined_text,
            combined_simplified=combined_simplified,
        )

    @app.post("/evaluate", response_model=_schemas.EvaluateResponse)
    async def evaluate(request: _schemas.EvaluateRequest) -> _schemas.EvaluateResponse:
        """Compute SARI, BLEU, and readability deltas for a prediction."""
        sari = _compute_sari(request.source, request.prediction, request.reference)
        bleu = _compute_sentence_bleu(request.prediction, request.reference)
        delta = _readability_delta(request.source, request.prediction)

        return _schemas.EvaluateResponse(
            sari=sari,
            bleu=bleu,
            readability_before=_schemas.ReadabilityMetrics(
                lix=delta["lix_before"],
                wstf=delta["wstf_before"],
            ),
            readability_after=_schemas.ReadabilityMetrics(
                lix=delta["lix_after"],
                wstf=delta["wstf_after"],
            ),
        )
