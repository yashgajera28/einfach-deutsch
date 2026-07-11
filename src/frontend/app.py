"""Streamlit frontend for German text simplification."""

from __future__ import annotations

import base64
import difflib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[2]))

_STREAMLIT_AVAILABLE = False
st: Any = None

try:
    import streamlit as st

    _STREAMLIT_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    warnings.warn(f"streamlit is not installed: {exc}", stacklevel=2)

_PANDAS_AVAILABLE = False
pd: Any = None
try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    pass

_PLOTLY_AVAILABLE = False
px: Any = None
try:
    import plotly.express as px

    _PLOTLY_AVAILABLE = True
except ImportError:
    pass

_FPDF_AVAILABLE = False
FPDF: Any = None
try:
    from fpdf import FPDF

    _FPDF_AVAILABLE = True
except ImportError:
    pass

_EXTRACT_AVAILABLE = False
_extract_text: Any = None
try:
    from src.api.extract import extract_text as _extract_text

    _EXTRACT_AVAILABLE = True
except ImportError as exc:
    warnings.warn(f"Text extractor is unavailable: {exc}", stacklevel=2)

_SIMPLIFIER_AVAILABLE = False
Simplifier: Any = None
try:
    from src.models.simplifier import Simplifier

    _SIMPLIFIER_AVAILABLE = True
except ImportError as exc:
    warnings.warn(f"Simplifier is unavailable: {exc}", stacklevel=2)

_BLEU_AVAILABLE = False
compute_sentence_bleu: Any = None
try:
    from src.evaluation.bleu import compute_sentence_bleu

    _BLEU_AVAILABLE = True
except ImportError as exc:
    warnings.warn(f"BLEU metric is unavailable: {exc}", stacklevel=2)

_SARI_AVAILABLE = False
compute_sari: Any = None
try:
    from src.evaluation.sari import compute_sari

    _SARI_AVAILABLE = True
except ImportError as exc:
    warnings.warn(f"SARI metric is unavailable: {exc}", stacklevel=2)

from src.evaluation.readability import lix_score, readability_delta, wstf_score

_CONFIG_AVAILABLE = False
load_config: Any = None
try:
    from src.utils.config import load_config

    _CONFIG_AVAILABLE = True
except ImportError as exc:
    warnings.warn(f"Config loader is unavailable: {exc}", stacklevel=2)

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"
TRANSLATIONS_PATH = Path(__file__).parent / "translations.json"

# Backend URL for a future HTTP API mode. The default local mode still calls the
# Simplifier directly so the frontend works without a running API server.
API_URL = os.environ.get("API_URL", "")

_CSS = """
<style>
:root {
  --bg: #0b0f19;
  --surface: #151b2b;
  --surface-hover: #1e293b;
  --border: #334155;
  --text: #f8fafc;
  --muted: #94a3b8;
  --primary: #6366f1;
  --primary-hover: #4f46e5;
  --secondary: #14b8a6;
  --success: #22c55e;
  --warning: #f59e0b;
  --danger: #f43f5e;
}

html, body, .stApp, [data-testid="stAppViewContainer"], .main .block-container {
  background-color: var(--bg) !important;
  color: var(--text);
}

header[data-testid="stHeader"],
footer,
#MainMenu,
[data-testid="stToolbar"] {
  display: none !important;
}

[data-testid="stSidebar"] {
  background-color: var(--surface) !important;
  border-right: 1px solid var(--surface-hover);
}

h1, h2, h3, h4, h5, h6 {
  color: var(--text) !important;
  font-weight: 700;
  letter-spacing: -0.01em;
}

p, li, .stMarkdown {
  color: var(--text);
}

.stCaption {
  color: var(--muted) !important;
}

label, .stWidgetLabel {
  color: var(--muted) !important;
  font-weight: 500 !important;
}

.stButton > button {
  background-color: var(--primary) !important;
  color: var(--text) !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.6rem 1.25rem !important;
  font-weight: 600 !important;
  transition: filter 0.2s ease, transform 0.1s ease;
}

.stButton > button:hover {
  filter: brightness(1.1);
  transform: translateY(-1px);
}

.stButton > button:active {
  transform: translateY(0);
}

.stTextArea textarea {
  background-color: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

.stTextArea textarea:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

.stSelectbox [data-baseweb="select"] {
  background-color: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

.stSelectbox [data-baseweb="select"] > div {
  background-color: transparent !important;
  color: var(--text) !important;
}

.stSelectbox svg {
  fill: var(--muted) !important;
}

[data-testid="stFileUploader"] {
  background-color: var(--surface) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 12px !important;
}

[data-testid="stFileUploader"]:hover {
  border-color: var(--primary) !important;
}

[data-testid="stFileUploader"] button {
  background-color: var(--surface-hover) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

.stRadio > div[role="radiogroup"] > label {
  color: var(--text) !important;
}

.stRadio input[type="radio"] {
  accent-color: var(--primary);
}

.stCheckbox label {
  color: var(--text) !important;
}

.stCheckbox input[type="checkbox"] {
  accent-color: var(--primary);
}

[data-testid="stProgressBar"] > div > div {
  background-color: var(--primary) !important;
}

.card {
  background-color: var(--surface);
  border: 1px solid var(--surface-hover);
  border-radius: 12px;
  padding: 1.25rem;
  margin-bottom: 1rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}

.card-title {
  color: var(--text);
  font-weight: 700;
  font-size: 1rem;
  margin-bottom: 0.75rem;
}

.card-original {
  border-top: 4px solid #3b82f6;
}

.card-simplified {
  border-top: 4px solid var(--secondary);
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.6rem;
  border-radius: 9999px;
  font-size: 0.8rem;
  font-weight: 700;
  color: #fff;
  margin-right: 0.4rem;
  margin-bottom: 0.4rem;
}

.badge-green {
  background-color: var(--success);
}

.badge-yellow {
  background-color: var(--warning);
  color: #0f172a;
}

.badge-red {
  background-color: var(--danger);
}

.badge-secondary {
  background-color: rgba(20, 184, 166, 0.15);
  color: #2dd4bf;
  border: 1px solid rgba(20, 184, 166, 0.3);
}

.diff-highlight {
  background-color: rgba(20, 184, 166, 0.15);
  color: #2dd4bf;
  border-radius: 4px;
  padding: 0.1rem 0.25rem;
}

.entity-chip {
  display: inline-block;
  background-color: rgba(99, 102, 241, 0.12);
  color: #c7d2fe;
  border: 1px solid rgba(99, 102, 241, 0.25);
  border-radius: 9999px;
  padding: 0.2rem 0.5rem;
  font-size: 0.85rem;
  margin: 0 0.35rem 0.35rem 0;
}

.explanation-list {
  list-style: none;
  padding-left: 0;
  margin: 0.5rem 0 0;
}

.explanation-list li {
  margin-bottom: 0.4rem;
  color: var(--text);
}

.explanation-list li::before {
  content: "•";
  color: var(--secondary);
  font-weight: 700;
  margin-right: 0.5rem;
}

.metrics-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.metric-box {
  background-color: var(--surface-hover);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  min-width: 5.5rem;
  text-align: center;
}

.metric-value {
  font-size: 1.25rem;
  font-weight: 800;
  color: var(--text);
}

.metric-label {
  font-size: 0.7rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.download-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  background-color: var(--surface-hover);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.5rem 1rem;
  text-decoration: none;
  font-weight: 600;
  font-size: 0.9rem;
  transition: background-color 0.2s ease;
}

.download-btn:hover {
  background-color: #27354f;
}

.settings-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  align-items: flex-end;
}

.disclaimer {
  color: var(--muted);
  font-size: 0.8rem;
  margin-top: 0.5rem;
}

.sidebar-title {
  font-weight: 700;
  color: var(--text);
  font-size: 1.1rem;
  margin-bottom: 0.25rem;
}

.sidebar-subtitle {
  color: var(--muted);
  font-size: 0.85rem;
  margin-bottom: 1rem;
}

hr {
  border-color: var(--surface-hover) !important;
}
</style>
"""


def load_translations(path: Path | str = TRANSLATIONS_PATH) -> dict[str, dict[str, str]]:
    """Load translation strings from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def t(key: str, translations: dict[str, dict[str, str]], lang: str) -> str:
    """Return the translation for ``key`` in ``lang`` or the key itself."""
    return translations.get(lang, {}).get(key, key)


def lix_badge_class(lix: float, config: dict[str, Any]) -> str:
    """Pick a CSS class based on the LIX score."""
    green = config.get("readability", {}).get("lix_green_max", 35)
    yellow = config.get("readability", {}).get("lix_yellow_max", 50)
    if lix < green:
        return "badge-green"
    if lix < yellow:
        return "badge-yellow"
    return "badge-red"


def diff_highlight(original: str, simplified: str) -> str:
    """Highlight changed words in the simplified text using difflib."""
    original_words = original.split()
    simplified_words = simplified.split()
    matcher = difflib.SequenceMatcher(None, original_words, simplified_words)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.extend(simplified_words[j1:j2])
        else:
            for word in simplified_words[j1:j2]:
                parts.append(f'<span class="diff-highlight">{word}</span>')
    return " ".join(parts)


def extract_uploaded_file(uploaded_file: Any) -> tuple[str, dict[int, str] | str]:
    """Extract text from an uploaded PDF, DOCX, or TXT file.

    Returns a tuple of (source_type, content) where content is either a
    page-by-page mapping for PDFs or the full text for other formats.
    """
    if uploaded_file is None:
        return ("", "")

    if not _EXTRACT_AVAILABLE or _extract_text is None:
        raise RuntimeError("Text extractor is not available. Install the required dependencies.")

    name = uploaded_file.name
    file_bytes = uploaded_file.read()
    extracted = _extract_text(name, file_bytes)

    if name.lower().endswith(".pdf"):
        return ("pdf", extracted)
    if name.lower().endswith(".docx"):
        return ("docx", extracted)
    return ("txt", extracted)


def make_txt_download(text: str, filename: str) -> str:
    """Return a base64 data URI for a plain text download."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    return f"data:text/plain;charset=utf-8;base64,{encoded}"


def make_pdf_download(text: str, filename: str) -> str:
    """Return a base64 data URI for a simple PDF download."""
    if _FPDF_AVAILABLE:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", size=12)
        for line in text.splitlines():
            pdf.cell(0, 8, txt=line, ln=True)
        pdf_bytes = pdf.output(dest="S")
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode("latin-1", errors="ignore")
        encoded = base64.b64encode(pdf_bytes).decode("utf-8")
        return f"data:application/pdf;base64,{encoded}"

    html = f"""
    <html>
    <head><meta charset="utf-8"><title>{filename}</title></head>
    <body><pre style="font-family: system-ui, sans-serif; white-space: pre-wrap;">{text}</pre></body>
    </html>
    """
    encoded = base64.b64encode(html.encode("utf-8")).decode("utf-8")
    return f"data:text/html;charset=utf-8;base64,{encoded}"


def _metric_box(label: str, value: str) -> str:
    """Return the HTML for a compact metric box."""
    return (
        f"<div class='metric-box'>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-label'>{label}</div>"
        f"</div>"
    )


def _render_result_card(
    original: str,
    simplified: str,
    translations: dict[str, dict[str, str]],
    lang: str,
) -> None:
    """Render original and simplified text side-by-side."""
    original_col, simplified_col = st.columns(2)
    with original_col:
        st.markdown(
            f"<div class='card card-original'>"
            f"<div class='card-title'>{t('original', translations, lang)}</div>"
            f"<p>{original}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with simplified_col:
        highlighted = diff_highlight(original, simplified)
        st.markdown(
            f"<div class='card card-simplified'>"
            f"<div class='card-title'>{t('simplified', translations, lang)}</div>"
            f"<p>{highlighted}</p>"
            f"<div class='disclaimer'>{t('diff_legend', translations, lang)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_metrics(
    result: dict[str, Any],
    config: dict[str, Any],
    translations: dict[str, dict[str, str]],
    lang: str,
) -> None:
    """Render readability, confidence, and explanation metadata."""
    before_lix = result["readability_before"]["lix"]
    before_wstf = result["readability_before"]["wstf"]
    after_lix = result["readability_after"]["lix"]
    after_wstf = result["readability_after"]["wstf"]

    html = f"<div class='card'><div class='card-title'>{t('metrics', translations, lang)}</div>"
    html += (
        f"<div style='margin-bottom:0.75rem;'>"
        f"<div style='color:var(--muted);font-size:0.85rem;margin-bottom:0.35rem;'>{t('readability_before', translations, lang)}</div>"
        f"<span class='badge {lix_badge_class(before_lix, config)}'>LIX {before_lix:.1f}</span>"
        f"<span class='badge badge-secondary'>WSTF {before_wstf:.1f}</span>"
        f"</div>"
    )
    html += (
        f"<div style='margin-bottom:0.75rem;'>"
        f"<div style='color:var(--muted);font-size:0.85rem;margin-bottom:0.35rem;'>{t('readability_after', translations, lang)}</div>"
        f"<span class='badge {lix_badge_class(after_lix, config)}'>LIX {after_lix:.1f}</span>"
        f"<span class='badge badge-secondary'>WSTF {after_wstf:.1f}</span>"
        f"</div>"
    )
    html += "<div class='metrics-row'>"
    html += _metric_box(t("confidence", translations, lang), f"{result['confidence']:.0%}")
    html += _metric_box(t("level_label", translations, lang), result["level"])
    html += _metric_box("Backend", result["backend"])
    html += "</div>"

    entities = result.get("entities_preserved") or []
    if entities:
        chips = "".join(f"<span class='entity-chip'>{ent}</span>" for ent in entities)
        html += (
            f"<div style='margin-top:0.75rem;'>"
            f"<div style='color:var(--muted);font-size:0.85rem;margin-bottom:0.35rem;'>{t('entities', translations, lang)}</div>"
            f"{chips}"
            f"</div>"
        )

    explanation = result.get("explanation")
    if explanation:
        items = "".join(f"<li>{item}</li>" for item in explanation)
        html += (
            f"<div style='margin-top:0.75rem;'>"
            f"<div style='color:var(--muted);font-size:0.85rem;margin-bottom:0.35rem;'>{t('explanation', translations, lang)}</div>"
            f"<ul class='explanation-list'>{items}</ul>"
            f"</div>"
        )

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_export_buttons(
    simplified_text: str,
    translations: dict[str, dict[str, str]],
    lang: str,
) -> None:
    """Render download buttons for the simplified text."""
    export_col1, export_col2 = st.columns(2)
    export_col1.markdown(
        f"<a class='download-btn' href='{make_txt_download(simplified_text, 'simplified.txt')}' "
        f"download='simplified.txt'>{t('export_txt', translations, lang)}</a>",
        unsafe_allow_html=True,
    )
    export_col2.markdown(
        f"<a class='download-btn' href='{make_pdf_download(simplified_text, 'simplified.pdf')}' "
        f"download='simplified.pdf'>{t('export_pdf', translations, lang)}</a>",
        unsafe_allow_html=True,
    )


def render_simplify_page(
    config: dict[str, Any],
    translations: dict[str, dict[str, str]],
    lang: str,
) -> None:
    """Render the single-text simplification page."""
    st.header(t("simplify_tab", translations, lang))

    if not _SIMPLIFIER_AVAILABLE:
        st.warning("Simplifier is not available. Some features will not work.")

    st.markdown(
        f"<div class='card'><div class='card-title'>{t('input_title', translations, lang)}</div>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        t("file_upload", translations, lang),
        type=["pdf", "docx", "txt"],
        help=t("drag_drop", translations, lang),
    )

    source_type = ""
    extracted_content: dict[int, str] | str = ""
    if uploaded_file is not None:
        try:
            source_type, extracted_content = extract_uploaded_file(uploaded_file)
        except Exception as exc:
            st.warning(str(exc))

    text_value = ""
    if source_type != "pdf" and isinstance(extracted_content, str):
        text_value = extracted_content

    text = st.text_area(
        t("input_label", translations, lang),
        value=text_value,
        height=200,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='card'><div class='card-title'>{t('settings', translations, lang)}</div>",
        unsafe_allow_html=True,
    )
    levels = config.get("simplification", {}).get("levels", ["A2", "B1", "B2"])
    default_level = config.get("simplification", {}).get("default_level", "B1")

    level_col, preserve_col, explain_col, button_col = st.columns(4)
    with level_col:
        level = st.selectbox(
            t("level_label", translations, lang),
            levels,
            index=levels.index(default_level) if default_level in levels else 1,
        )
    with preserve_col:
        preserve_entities = st.checkbox(t("preserve_entities", translations, lang), value=True)
    with explain_col:
        explain = st.checkbox(t("explain", translations, lang), value=False)
    with button_col:
        st.markdown("<div style='padding-bottom:0.5rem;'> </div>", unsafe_allow_html=True)
        run = st.button(t("simplify_button", translations, lang), type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if not run:
        return

    if not _SIMPLIFIER_AVAILABLE:
        st.error("Cannot run simplification because the Simplifier model is unavailable.")
        return

    backend = config.get("frontend", {}).get("backend", "baseline")
    simplifier = Simplifier(config, backend=backend)

    if source_type == "pdf" and isinstance(extracted_content, dict):
        pages = {num: txt for num, txt in extracted_content.items() if txt.strip()}
        if not pages:
            st.info(t("empty_input", translations, lang))
            return

        st.markdown(
            f"<div class='card'><div class='card-title'>{t('output_title', translations, lang)}</div>",
            unsafe_allow_html=True,
        )
        simplified_pages: list[str] = []
        for page_num in sorted(pages):
            with st.spinner(f"{t('loading', translations, lang)} ({t('page', translations, lang)} {page_num})"):
                result = simplifier.simplify(
                    pages[page_num],
                    level=level,
                    preserve_entities=preserve_entities,
                    explain=explain,
                )
            simplified_pages.append(result["simplified"])
            with st.expander(f"{t('pdf', translations, lang)} – {t('page', translations, lang)} {page_num}"):
                _render_result_card(pages[page_num], result["simplified"], translations, lang)
                _render_metrics(result, config, translations, lang)

        st.markdown(
            f"<div class='disclaimer'>{t('model_disclaimer', translations, lang)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        combined_simplified = "\n".join(simplified_pages)
        _render_export_buttons(combined_simplified, translations, lang)
        return

    active_text = text.strip()
    if not active_text and source_type in {"docx", "txt"} and isinstance(extracted_content, str):
        active_text = extracted_content.strip()

    if not active_text:
        st.info(t("empty_input", translations, lang))
        return

    with st.spinner(t("loading", translations, lang)):
        result = simplifier.simplify(
            active_text,
            level=level,
            preserve_entities=preserve_entities,
            explain=explain,
        )

    st.markdown(
        f"<div class='card'><div class='card-title'>{t('output_title', translations, lang)}</div>",
        unsafe_allow_html=True,
    )
    _render_result_card(active_text, result["simplified"], translations, lang)
    _render_metrics(result, config, translations, lang)
    _render_export_buttons(result["simplified"], translations, lang)
    st.markdown(
        f"<div class='disclaimer'>{t('model_disclaimer', translations, lang)}</div></div>",
        unsafe_allow_html=True,
    )


def render_batch_page(
    config: dict[str, Any],
    translations: dict[str, dict[str, str]],
    lang: str,
) -> None:
    """Render the batch CSV processing page."""
    st.header(t("batch_tab", translations, lang))

    if not _PANDAS_AVAILABLE:
        st.error("pandas is not installed; batch processing is unavailable.")
        return
    if not _SIMPLIFIER_AVAILABLE:
        st.error("Simplifier is not available; batch processing is unavailable.")
        return

    uploaded_file = st.file_uploader(t("upload_csv", translations, lang), type=["csv"])
    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)
    st.markdown(
        f"<div class='card'><div class='card-title'>{t('preview', translations, lang)}</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    text_column = st.selectbox(t("text_column", translations, lang), df.columns)
    if text_column not in df.columns:
        st.error(t("missing_column", translations, lang).format(column=text_column))
        return

    levels = config.get("simplification", {}).get("levels", ["A2", "B1", "B2"])
    default_level = config.get("simplification", {}).get("default_level", "B1")
    level = st.selectbox(
        t("level_label", translations, lang),
        levels,
        index=levels.index(default_level) if default_level in levels else 1,
    )

    if not st.button(t("simplify_button", translations, lang), type="primary"):
        return

    backend = config.get("frontend", {}).get("backend", "baseline")
    simplifier = Simplifier(config, backend=backend)

    texts = df[text_column].astype(str).tolist()
    results: list[dict[str, Any]] = []
    progress = st.progress(0.0)

    for idx, text in enumerate(texts):
        result = simplifier.simplify(text, level=level, preserve_entities=True, explain=False)
        results.append({
            "simplified": result["simplified"],
            "lix_before": result["readability_before"]["lix"],
            "lix_after": result["readability_after"]["lix"],
            "wstf_before": result["readability_before"]["wstf"],
            "wstf_after": result["readability_after"]["wstf"],
            "confidence": result["confidence"],
        })
        progress.progress((idx + 1) / len(texts))

    result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    st.markdown(
        f"<div class='card'><div class='card-title'>{t('output_title', translations, lang)}</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(result_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    csv = result_df.to_csv(index=False).encode("utf-8")
    encoded = base64.b64encode(csv).decode("utf-8")
    st.markdown(
        f"<a class='download-btn' href='data:text/csv;charset=utf-8;base64,{encoded}' "
        f"download='batch_results.csv'>{t('download_results', translations, lang)}</a>",
        unsafe_allow_html=True,
    )


def render_eval_page(
    config: dict[str, Any],
    translations: dict[str, dict[str, str]],
    lang: str,
) -> None:
    """Render the evaluation page."""
    st.header(t("eval_tab", translations, lang))

    if not _PANDAS_AVAILABLE:
        st.error("pandas is not installed; evaluation is unavailable.")
        return

    uploaded_file = st.file_uploader(t("upload_csv", translations, lang), type=["csv"])
    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)
    columns = list(df.columns)

    source_col = st.selectbox(
        t("source_column", translations, lang),
        columns,
        index=columns.index("source") if "source" in columns else 0,
    )
    reference_col = st.selectbox(
        t("reference_column", translations, lang),
        columns,
        index=columns.index("reference") if "reference" in columns else 0,
    )
    has_prediction = "prediction" in columns
    prediction_col = st.selectbox(
        t("prediction_column", translations, lang),
        columns,
        index=columns.index("prediction") if has_prediction else 0,
    )

    if not st.button(t("run_eval", translations, lang), type="primary"):
        return

    if not _SARI_AVAILABLE or not _BLEU_AVAILABLE:
        st.error("SARI or BLEU metrics are unavailable; cannot run evaluation.")
        return

    backend = config.get("frontend", {}).get("backend", "baseline")
    simplifier = None
    if not has_prediction and _SIMPLIFIER_AVAILABLE:
        simplifier = Simplifier(config, backend=backend)
    elif not has_prediction:
        st.error("No prediction column found and Simplifier is unavailable.")
        return

    sari_scores: list[float] = []
    bleu_scores: list[float] = []
    lix_deltas: list[float] = []
    wstf_deltas: list[float] = []
    predictions: list[str] = []

    progress = st.progress(0.0)
    for idx, row in df.iterrows():
        source = str(row[source_col])
        reference = str(row[reference_col])
        prediction = str(row[prediction_col]) if has_prediction else ""

        if simplifier is not None:
            prediction = simplifier.simplify(source, level="B1", preserve_entities=True, explain=False)["simplified"]

        predictions.append(prediction)
        sari_scores.append(compute_sari(source, prediction, [reference]))
        bleu_scores.append(compute_sentence_bleu(prediction, [reference]))
        delta = readability_delta(source, prediction)
        lix_deltas.append(delta["lix_delta"])
        wstf_deltas.append(delta["wstf_delta"])
        progress.progress((idx + 1) / len(df))

    html = f"<div class='card'><div class='card-title'>{t('metrics', translations, lang)}</div><div class='metrics-row'>"
    html += _metric_box(t("mean_sari", translations, lang), f"{sum(sari_scores) / len(sari_scores):.2f}")
    html += _metric_box(t("mean_bleu", translations, lang), f"{sum(bleu_scores) / len(bleu_scores):.2f}")
    html += _metric_box(t("mean_lix_delta", translations, lang), f"{sum(lix_deltas) / len(lix_deltas):.2f}")
    html += _metric_box(t("mean_wstf_delta", translations, lang), f"{sum(wstf_deltas) / len(wstf_deltas):.2f}")
    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)

    if _PLOTLY_AVAILABLE:
        fig = px.histogram(
            x=[lix_score(str(row[source_col])) for _, row in df.iterrows()],
            nbins=20,
            title=t("histogram_title", translations, lang),
            labels={"x": "LIX"},
            color_discrete_sequence=["#6366f1"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#151b2b",
            font_color="#f8fafc",
            title_font_color="#f8fafc",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        source_lix = [lix_score(str(row[source_col])) for _, row in df.iterrows()]
        pred_lix = [lix_score(prediction) for prediction in predictions]
        chart_data = pd.DataFrame({"source": source_lix, "prediction": pred_lix})
        st.bar_chart(chart_data)


def render_about_page(
    config: dict[str, Any],
    translations: dict[str, dict[str, str]],
    lang: str,
) -> None:
    """Render the about / model info page."""
    st.header(t("about_tab", translations, lang))

    st.markdown(
        f"<div class='card'><div class='card-title'>{t('about_tab', translations, lang)}</div>"
        f"<p>{t('about_text', translations, lang)}</p></div>",
        unsafe_allow_html=True,
    )

    baseline_name = config.get("models", {}).get("baseline", {}).get("name", "unknown")
    lora_name = config.get("models", {}).get("lora", {}).get("name", "unknown")
    levels = ", ".join(config.get("simplification", {}).get("levels", ["A2", "B1", "B2"]))

    st.markdown(
        f"""
        <div class='card'>
            <div class='card-title'>{t('model_info', translations, lang)}</div>
            <p><strong>Baseline model:</strong> {baseline_name}</p>
            <p><strong>LoRA model:</strong> {lora_name}</p>
            <p><strong>Supported levels:</strong> {levels}</p>
            <p><strong>SpaCy model:</strong> {config.get("spacy", {}).get("model", "unknown")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    examples = [
        "Die Kommission hat beschlossen, die Richtlinie zu überarbeiten.",
        "Der Antragsteller legte umfangreiche Unterlagen vor, die geprüft wurden.",
        "Aufgrund der aktuellen gesetzlichen Regelung ist eine Anpassung erforderlich.",
    ]
    examples_html = "".join(f"<li>{example}</li>" for example in examples)
    st.markdown(
        f"<div class='card'><div class='card-title'>{t('example_inputs', translations, lang)}</div>"
        f"<ul class='explanation-list'>{examples_html}</ul></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    """Entry point for the Streamlit frontend."""
    if not _STREAMLIT_AVAILABLE:
        raise RuntimeError("streamlit is not installed. Install it with: pip install streamlit")

    if not _CONFIG_AVAILABLE or load_config is None:
        raise RuntimeError("Project config loader is unavailable; install pyyaml and ensure src.utils.config is importable.")

    config = load_config(CONFIG_PATH)
    translations = load_translations()

    if "language" not in st.session_state:
        st.session_state.language = "de"

    lang = st.session_state.language

    st.set_page_config(
        page_title=t("app_title", translations, lang),
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            f"<div class='sidebar-title'>{t('app_title', translations, lang)}</div>"
            f"<div class='sidebar-subtitle'>{t('app_subtitle', translations, lang)}</div>",
            unsafe_allow_html=True,
        )
        lang = st.selectbox(
            t("language", translations, lang),
            ["de", "en"],
            index=0 if lang == "de" else 1,
        )
        st.session_state.language = lang

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<div class='sidebar-title'>{t('nav', translations, lang)}</div>", unsafe_allow_html=True)
        page = st.radio(
            label="page",
            options=["simplify", "batch", "eval", "about"],
            format_func=lambda p: t(f"{p}_tab", translations, lang),
            label_visibility="collapsed",
        )

    st.title(t("app_title", translations, lang))
    st.caption(t("app_subtitle", translations, lang))

    if page == "simplify":
        render_simplify_page(config, translations, lang)
    elif page == "batch":
        render_batch_page(config, translations, lang)
    elif page == "eval":
        render_eval_page(config, translations, lang)
    else:
        render_about_page(config, translations, lang)


if __name__ == "__main__":
    main()
