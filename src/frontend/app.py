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
html, body, [class*="stApp"] {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #334155;
    background-color: #f8fafc;
}

h1, h2, h3, h4, h5, h6 {
    color: #0f172a;
    font-weight: 700;
    letter-spacing: -0.01em;
}

.stButton > button {
    background-color: #f59e0b;
    color: #0f172a;
    border: none;
    border-radius: 0.5rem;
    padding: 0.6rem 1.25rem;
    font-weight: 600;
    transition: background-color 0.2s ease, transform 0.1s ease;
}

.stButton > button:hover {
    background-color: #d97706;
    color: #fffbeb;
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0);
}

.card {
    background-color: #ffffff;
    border-radius: 0.75rem;
    padding: 1.25rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    margin-bottom: 1rem;
}

.card-original {
    border-top: 4px solid #3b82f6;
}

.card-simplified {
    border-top: 4px solid #f59e0b;
}

.highlight {
    background-color: #fffbeb;
    padding: 0.1rem 0.2rem;
    border-radius: 0.25rem;
    font-weight: 500;
}

.badge {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 700;
    color: #ffffff;
    margin-right: 0.5rem;
}

.badge-green { background-color: #10b981; }
.badge-yellow { background-color: #f59e0b; }
.badge-red { background-color: #ef4444; }

.metric-box {
    background-color: #ffffff;
    border-radius: 0.5rem;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #0f172a;
}

.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.sidebar-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.5rem;
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
                parts.append(f'<span class="highlight">{word}</span>')
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


def _render_result_card(
    original: str,
    simplified: str,
    config: dict[str, Any],
    translations: dict[str, dict[str, str]],
    lang: str,
) -> None:
    """Render original and simplified text side-by-side."""
    original_col, simplified_col = st.columns(2)
    with original_col:
        st.markdown("<div class='card card-original'>", unsafe_allow_html=True)
        st.markdown(f"**{t('original', translations, lang)}**")
        st.markdown(original)
        st.markdown("</div>", unsafe_allow_html=True)

    with simplified_col:
        st.markdown("<div class='card card-simplified'>", unsafe_allow_html=True)
        st.markdown(f"**{t('simplified', translations, lang)}**")
        highlighted = diff_highlight(original, simplified)
        st.markdown(highlighted, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_entities_box(
    result: dict[str, Any],
    translations: dict[str, dict[str, str]],
    lang: str,
) -> None:
    """Render a compact info box with extracted entities."""
    entities = result.get("entities_preserved") or []
    if not entities:
        return

    items = " ".join(
        f"<span style='background-color:#e0f2fe;color:#0c4a6e;padding:0.2rem 0.5rem;border-radius:0.35rem;font-size:0.85rem;margin-right:0.35rem;display:inline-block;margin-bottom:0.35rem;'>{ent}</span>"
        for ent in entities
    )
    st.markdown(
        f"<div style='background-color:#f0f9ff;border-left:4px solid #0ea5e9;border-radius:0.5rem;padding:0.75rem;margin-bottom:1rem;'>"
        f"<strong>{t('entities', translations, lang)}</strong><br>{items}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_explanation_list(
    result: dict[str, Any],
    translations: dict[str, dict[str, str]],
    lang: str,
) -> None:
    """Render explanations as a styled list with amber bullets."""
    explanation = result.get("explanation")
    if not explanation:
        return

    items = "".join(
        f"<li style='margin-bottom:0.35rem;'><span style='color:#f59e0b;font-weight:700;margin-right:0.4rem;'>•</span>{item}</li>"
        for item in explanation
    )
    st.markdown(
        f"<div style='margin-top:0.75rem;'>"
        f"<strong>{t('explanation', translations, lang)}</strong>"
        f"<ul style='list-style:none;padding-left:0;margin-top:0.4rem;'>{items}</ul>"
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
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"**{t('readability_before', translations, lang)}**")
    before_lix = result["readability_before"]["lix"]
    before_wstf = result["readability_before"]["wstf"]
    st.markdown(
        f"<span class='badge {lix_badge_class(before_lix, config)}'>LIX {before_lix:.1f}</span>"
        f"<span class='badge badge-yellow'>WSTF {before_wstf:.1f}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{t('readability_after', translations, lang)}**")
    after_lix = result["readability_after"]["lix"]
    after_wstf = result["readability_after"]["wstf"]
    st.markdown(
        f"<span class='badge {lix_badge_class(after_lix, config)}'>LIX {after_lix:.1f}</span>"
        f"<span class='badge badge-yellow'>WSTF {after_wstf:.1f}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric(t("confidence", translations, lang), f"{result['confidence']:.2%}")
    cols[1].metric(t("level_label", translations, lang), result["level"])
    cols[2].metric("Backend", result["backend"])

    _render_entities_box(result, translations, lang)
    _render_explanation_list(result, translations, lang)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_export_buttons(simplified_text: str, translations: dict[str, dict[str, str]], lang: str) -> None:
    """Render download buttons for the simplified text."""
    export_col1, export_col2 = st.columns(2)
    export_col1.markdown(
        f"<a href='{make_txt_download(simplified_text, 'simplified.txt')}' download='simplified.txt'>"
        f"<button>{t('export_txt', translations, lang)}</button></a>",
        unsafe_allow_html=True,
    )
    export_col2.markdown(
        f"<a href='{make_pdf_download(simplified_text, 'simplified.pdf')}' download='simplified.pdf'>"
        f"<button>{t('export_pdf', translations, lang)}</button></a>",
        unsafe_allow_html=True,
    )


def render_simplify_page(config: dict[str, Any], translations: dict[str, dict[str, str]], lang: str) -> None:
    """Render the single-text simplification page."""
    st.header(t("simplify_tab", translations, lang))

    if not _SIMPLIFIER_AVAILABLE:
        st.warning("Simplifier is not available. Some features will not work.")

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

    text = st.text_area(t("input_label", translations, lang), value=text_value, height=200)

    levels = config.get("simplification", {}).get("levels", ["A2", "B1", "B2"])
    default_level = config.get("simplification", {}).get("default_level", "B1")
    level = st.selectbox(t("level_label", translations, lang), levels, index=levels.index(default_level) if default_level in levels else 1)

    col_a, col_b = st.columns(2)
    preserve_entities = col_a.checkbox(t("preserve_entities", translations, lang), value=True)
    explain = col_b.checkbox(t("explain", translations, lang), value=False)

    if st.button(t("simplify_button", translations, lang), type="primary"):
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

            st.markdown("<div class='card'><h4>{}</h4></div>".format(t("output_label", translations, lang)), unsafe_allow_html=True)
            simplified_pages: list[str] = []
            for page_num in sorted(pages):
                with st.spinner(t("loading", translations, lang)):
                    result = simplifier.simplify(
                        pages[page_num],
                        level=level,
                        preserve_entities=preserve_entities,
                        explain=explain,
                    )
                simplified_pages.append(result["simplified"])
                with st.expander(f"{t('pdf', translations, lang)} - {t('page', translations, lang)} {page_num}"):
                    _render_result_card(pages[page_num], result["simplified"], config, translations, lang)
                    _render_metrics(result, config, translations, lang)

            combined_original = "\n".join(pages[num] for num in sorted(pages))
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

        st.markdown("<div class='card'><h4>{}</h4></div>".format(t("output_label", translations, lang)), unsafe_allow_html=True)
        _render_result_card(active_text, result["simplified"], config, translations, lang)
        _render_metrics(result, config, translations, lang)
        _render_export_buttons(result["simplified"], translations, lang)


def render_batch_page(config: dict[str, Any], translations: dict[str, dict[str, str]], lang: str) -> None:
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
    text_column = st.selectbox(t("text_column", translations, lang), df.columns)

    if text_column not in df.columns:
        st.error(t("missing_column", translations, lang).format(column=text_column))
        return

    levels = config.get("simplification", {}).get("levels", ["A2", "B1", "B2"])
    default_level = config.get("simplification", {}).get("default_level", "B1")
    level = st.selectbox(t("level_label", translations, lang), levels, index=levels.index(default_level) if default_level in levels else 1)

    if st.button(t("simplify_button", translations, lang), type="primary"):
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
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")
        encoded = base64.b64encode(csv).decode("utf-8")
        st.markdown(
            f"<a href='data:text/csv;charset=utf-8;base64,{encoded}' download='batch_results.csv'>"
            f"<button>{t('download_results', translations, lang)}</button></a>",
            unsafe_allow_html=True,
        )


def render_eval_page(config: dict[str, Any], translations: dict[str, dict[str, str]], lang: str) -> None:
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

    source_col = st.selectbox(t("source_column", translations, lang), columns, index=columns.index("source") if "source" in columns else 0)
    reference_col = st.selectbox(t("reference_column", translations, lang), columns, index=columns.index("reference") if "reference" in columns else 0)
    has_prediction = "prediction" in columns
    prediction_col = st.selectbox(t("prediction_column", translations, lang), columns, index=columns.index("prediction") if has_prediction else 0)

    if st.button(t("run_eval", translations, lang), type="primary"):
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

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t("mean_sari", translations, lang), f"{sum(sari_scores) / len(sari_scores):.2f}")
        col2.metric(t("mean_bleu", translations, lang), f"{sum(bleu_scores) / len(bleu_scores):.2f}")
        col3.metric(t("mean_lix_delta", translations, lang), f"{sum(lix_deltas) / len(lix_deltas):.2f}")
        col4.metric(t("mean_wstf_delta", translations, lang), f"{sum(wstf_deltas) / len(wstf_deltas):.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

        if _PLOTLY_AVAILABLE:
            fig = px.histogram(
                x=[lix_score(str(row[source_col])) for _, row in df.iterrows()],
                nbins=20,
                title=t("histogram_title", translations, lang),
                labels={"x": "LIX"},
                color_discrete_sequence=["#0f172a"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            source_lix = [lix_score(str(row[source_col])) for _, row in df.iterrows()]
            pred_lix = [lix_score(prediction) for prediction in predictions]
            chart_data = pd.DataFrame({"source": source_lix, "prediction": pred_lix})
            st.bar_chart(chart_data)


def render_about_page(config: dict[str, Any], translations: dict[str, dict[str, str]], lang: str) -> None:
    """Render the about / model info page."""
    st.header(t("about_tab", translations, lang))
    st.markdown(f"<div class='card'>{t('about_text', translations, lang)}</div>", unsafe_allow_html=True)

    st.subheader(t("model_info", translations, lang))
    baseline_name = config.get("models", {}).get("baseline", {}).get("name", "unknown")
    lora_name = config.get("models", {}).get("lora", {}).get("name", "unknown")
    levels = ", ".join(config.get("simplification", {}).get("levels", ["A2", "B1", "B2"]))

    st.markdown(
        f"""
        <div class='card'>
            <p><strong>Baseline model:</strong> {baseline_name}</p>
            <p><strong>LoRA model:</strong> {lora_name}</p>
            <p><strong>Supported levels:</strong> {levels}</p>
            <p><strong>SpaCy model:</strong> {config.get("spacy", {}).get("model", "unknown")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(t("example_inputs", translations, lang))
    examples = [
        "Die Kommission hat beschlossen, die Richtlinie zu überarbeiten.",
        "Der Antragsteller legte umfangreiche Unterlagen vor, die geprüft wurden.",
        "Aufgrund der aktuellen gesetzlichen Regelung ist eine Anpassung erforderlich.",
    ]
    for example in examples:
        st.markdown(f"<div class='card'>{example}</div>", unsafe_allow_html=True)


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

    st.set_page_config(
        page_title="Einfach Deutsch",
        page_icon=None,
        layout="wide",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<div class='sidebar-title'>{}</div>".format(t("app_title", translations, st.session_state.language)), unsafe_allow_html=True)
        lang = st.selectbox(t("language", translations, st.session_state.language), ["de", "en"], index=0 if st.session_state.language == "de" else 1)
        st.session_state.language = lang

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-title'>{}</div>".format(t("nav", translations, lang)), unsafe_allow_html=True)
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
