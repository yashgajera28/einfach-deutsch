"""NiceGUI frontend for German text simplification."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from nicegui import app, ui

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
TRANSLATIONS_PATH = Path(__file__).parent / "translations.json"

Simplifier: Any = None
extract_text: Any = None
load_config: Any = None
compute_sentence_bleu: Any = None
compute_sari: Any = None
lix_score: Any = None
wstf_score: Any = None

warnings.filterwarnings("ignore")

try:
    from src.models.simplifier import Simplifier
except Exception as exc:
    warnings.warn(f"Simplifier is unavailable: {exc}", stacklevel=2)

try:
    from src.api.extract import extract_text
except Exception as exc:
    warnings.warn(f"Text extractor is unavailable: {exc}", stacklevel=2)

try:
    from src.utils.config import load_config
except Exception as exc:
    warnings.warn(f"Config loader is unavailable: {exc}", stacklevel=2)

try:
    from src.evaluation.bleu import compute_sentence_bleu
except Exception:
    pass

try:
    from src.evaluation.sari import compute_sari
except Exception:
    pass

try:
    from src.evaluation.readability import lix_score, wstf_score
except Exception:
    pass


def _load_translations() -> dict[str, dict[str, str]]:
    if TRANSLATIONS_PATH.exists():
        with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


TRANSLATIONS = _load_translations()


def _t(key: str, lang: str = "en") -> str:
    return TRANSLATIONS.get(lang, {}).get(key) or TRANSLATIONS.get("en", {}).get(key) or key


class AppState:
    def __init__(self) -> None:
        self.lang = "en"
        self.theme = "dark"
        self.page = "simplify"
        self.simplifier: Any = None
        self.config: dict[str, Any] = {}
        self._load_config()
        self._init_simplifier()

    def _load_config(self) -> None:
        if load_config is not None:
            try:
                self.config = load_config(CONFIG_PATH)
            except Exception:
                self.config = {}
        else:
            self.config = {}

    def _init_simplifier(self) -> None:
        if Simplifier is None or not self.config:
            return
        try:
            self.simplifier = Simplifier(self.config, backend="baseline")
        except Exception as exc:
            warnings.warn(f"Failed to load baseline simplifier: {exc}", stacklevel=2)


state = AppState()
sidebar: ui.drawer
content_container: ui.element
theme_button: ui.button


def _set_theme(theme: str) -> None:
    state.theme = theme
    app.storage.user["theme"] = theme
    ui.run_javascript(f"""
        document.body.classList.remove('theme-dark', 'theme-light');
        document.body.classList.add('theme-{theme}');
    """)
    if theme_button is not None:
        theme_button.text = "☀️" if theme == "dark" else "🌙"


def _toggle_theme() -> None:
    _set_theme("light" if state.theme == "dark" else "dark")


def _set_lang(lang: str) -> None:
    state.lang = lang
    _switch_page(state.page)


def _switch_page(name: str) -> None:
    state.page = name
    content_container.clear()
    with content_container:
        if name == "simplify":
            simplify_page()
        elif name == "batch":
            batch_page()
        elif name == "evaluate":
            evaluate_page()
        elif name == "about":
            about_page()


def _header() -> None:
    global theme_button
    with ui.header().classes("items-center justify-between"):
        with ui.row().classes("items-center gap-2"):
            ui.button(icon="menu", on_click=lambda: sidebar.toggle()).props("flat round color=white")
            ui.label("Einfach Deutsch").classes("text-h6 font-bold")
        with ui.row().classes("items-center gap-2"):
            theme_button = ui.button("☀️", on_click=_toggle_theme).props("flat round color=white")
            with ui.button(icon="language").props("flat round color=white"):
                with ui.menu() as lang_menu:
                    ui.menu_item("English", lambda: _set_lang("en"))
                    ui.menu_item("Deutsch", lambda: _set_lang("de"))


def _sidebar() -> ui.drawer:
    drawer = ui.drawer(value=True, side="left", bordered=True).classes("p-4")
    with drawer:
        ui.label(_t("nav", state.lang)).classes("text-subtitle1 text-weight-bold q-mb-sm")
        _nav_button(_t("simplify_tab", state.lang), "simplify")
        _nav_button(_t("batch_tab", state.lang), "batch")
        _nav_button(_t("eval_tab", state.lang), "evaluate")
        _nav_button(_t("about_tab", state.lang), "about")
        ui.separator().classes("q-my-md")
        ui.label(_t("model_info", state.lang)).classes("text-caption text-grey")
        ui.label("Baseline (mT5-small)").classes("text-body2")
    return drawer


def _nav_button(label: str, page_name: str) -> ui.button:
    is_active = state.page == page_name
    color = "primary" if is_active else None
    props = "align=left" if is_active else "align=left flat"
    btn = ui.button(label, color=color).props(props).classes("w-full q-mb-xs")
    btn.on("click", lambda p=page_name: _switch_page(p))
    return btn


def _card(title: str = "") -> ui.card:
    card = ui.card().classes("w-full q-pa-md")
    if title:
        with card:
            ui.label(title).classes("text-h6 q-mb-sm")
    return card


def _metric_chip(text: str) -> ui.chip:
    return ui.chip(text).props("color=primary text-color=white")


def simplify_page() -> None:
    with ui.column().classes("w-full gap-4"):
        with _card():
            ui.label(_t("simplify_tab", state.lang)).classes("text-h5 text-weight-bold")
            ui.label(_t("app_subtitle", state.lang)).classes("text-body2 text-grey")

        input_text = ui.textarea(label=_t("input_label", state.lang), placeholder="...").classes("w-full min-h-[160px]")
        uploaded_label = ui.label("").classes("hidden")

        def on_upload(e: Any) -> None:
            if extract_text is None:
                ui.notify("Extractor unavailable", type="negative")
                return
            try:
                name = e.name.lower()
                content = e.content.read()
                if name.endswith(".pdf"):
                    pages = extract_text(content)
                    text = "\n\n".join(pages.values())
                elif name.endswith(".docx"):
                    text = extract_text(content)
                else:
                    text = content.decode("utf-8")
                input_text.value = text
                uploaded_label.set_text(f"Uploaded: {e.name}")
                uploaded_label.classes(remove="hidden")
                ui.notify("File loaded", type="positive")
            except Exception as exc:
                ui.notify(f"Upload failed: {exc}", type="negative")

        with _card(_t("file_upload", state.lang)):
            ui.upload(label=_t("drag_drop", state.lang), on_upload=on_upload).props("accept=.pdf,.docx,.txt")
            uploaded_label

        with _card(_t("settings", state.lang)):
            with ui.row().classes("w-full items-center gap-4 wrap"):
                level = ui.select(["A2", "B1", "B2"], label=_t("level_label", state.lang), value="B1").classes("w-32")
                preserve = ui.checkbox(_t("preserve_entities", state.lang), value=True)
                explain = ui.checkbox(_t("explain", state.lang), value=False)

        output_card = _card(_t("output_title", state.lang))
        output_card.classes("hidden")

        def do_simplify() -> None:
            if state.simplifier is None:
                ui.notify("Model not loaded", type="negative")
                return
            text = input_text.value or ""
            if not text.strip():
                ui.notify(_t("empty_input", state.lang), type="warning")
                return
            spinner = ui.spinner("dots", size="lg")
            try:
                result = state.simplifier.simplify(
                    text,
                    level=level.value,
                    preserve_entities=preserve.value,
                    explain=explain.value,
                )
                output_card.clear()
                with output_card:
                    ui.label(result.get("simplified", "")).classes("text-body1 q-mb-md")
                    with ui.row().classes("gap-2 q-mb-md"):
                        before = result.get("readability_before", {})
                        after = result.get("readability_after", {})
                        _metric_chip(f"LIX {before.get('lix', 0):.1f} → {after.get('lix', 0):.1f}")
                        _metric_chip(f"WSTF {before.get('wstf', 0):.1f} → {after.get('wstf', 0):.1f}")
                        _metric_chip(f"Conf {result.get('confidence', 0):.2f}")
                    if result.get("confidence", 0) < 0.1:
                        ui.badge("Low quality output — model needs more training", color="negative").classes("q-mb-md")
                    if result.get("entities_preserved"):
                        with ui.row().classes("gap-2 q-mt-sm items-center"):
                            ui.label(_t("entities_preserved", state.lang) + ":").classes("text-caption")
                            for ent in result["entities_preserved"]:
                                ui.chip(ent).props("outline color=secondary")
                    if result.get("explanation"):
                        ui.label(_t("explanation", state.lang)).classes("text-subtitle2 q-mt-md")
                        for item in result["explanation"]:
                            ui.label("• " + item).classes("text-body2")
                output_card.classes(remove="hidden")
            except Exception as exc:
                ui.notify(f"Simplification failed: {exc}", type="negative")
            finally:
                spinner.delete()

        ui.button(_t("simplify_button", state.lang), on_click=do_simplify).props("unelevated color=primary size=lg").classes("w-full")


def batch_page() -> None:
    files: list[Any] = []

    with ui.column().classes("w-full gap-4"):
        _card(_t("batch_tab", state.lang))

        def on_upload(e: Any) -> None:
            files.append(e)
            ui.notify(f"Added {e.name}", type="positive")

        with _card(_t("file_upload", state.lang)):
            ui.upload(label=_t("drag_drop", state.lang), on_upload=on_upload, multiple=True).props("accept=.pdf,.docx,.txt")

        level = ui.select(["A2", "B1", "B2"], label=_t("level_label", state.lang), value="B1").classes("w-32")

        result_area = ui.column().classes("w-full gap-2")

        def do_batch() -> None:
            if state.simplifier is None:
                ui.notify("Model not loaded", type="negative")
                return
            if not files:
                ui.notify(_t("no_results", state.lang), type="warning")
                return
            result_area.clear()
            spinner = ui.spinner("dots", size="lg")
            try:
                for f in files:
                    name = f.name.lower()
                    content = f.content.read()
                    if name.endswith(".pdf") and extract_text is not None:
                        pages = extract_text(content)
                        text = "\n\n".join(pages.values())
                    elif name.endswith(".docx") and extract_text is not None:
                        text = extract_text(content)
                    else:
                        text = content.decode("utf-8")
                    result = state.simplifier.simplify(text, level=level.value)
                    with result_area:
                        with ui.card().classes("w-full q-pa-sm"):
                            ui.label(f.name).classes("text-subtitle2")
                            ui.label(result.get("simplified", "")).classes("text-body2")
                ui.notify("Batch complete", type="positive")
            except Exception as exc:
                ui.notify(f"Batch failed: {exc}", type="negative")
            finally:
                spinner.delete()

        ui.button(_t("simplify_button", state.lang), on_click=do_batch).props("unelevated color=primary size=lg").classes("w-full")


def evaluate_page() -> None:
    with ui.column().classes("w-full gap-4"):
        _card(_t("eval_tab", state.lang))

        with ui.row().classes("w-full gap-4"):
            original = ui.textarea(label=_t("original", state.lang)).classes("flex-1 min-h-[160px]")
            simplified = ui.textarea(label=_t("simplified", state.lang)).classes("flex-1 min-h-[160px]")

        metrics_card = _card(_t("metrics", state.lang))
        metrics_card.classes("hidden")

        def do_evaluate() -> None:
            orig = original.value or ""
            simp = simplified.value or ""
            if not orig.strip() or not simp.strip():
                ui.notify(_t("empty_input", state.lang), type="warning")
                return
            metrics_card.clear()
            with metrics_card:
                with ui.row().classes("gap-4 wrap"):
                    if compute_sentence_bleu is not None:
                        try:
                            bleu = compute_sentence_bleu(orig, simp)
                            _metric_chip(f"BLEU {bleu:.2f}")
                        except Exception:
                            pass
                    if compute_sari is not None:
                        try:
                            sari = compute_sari(orig, simp, [simp])
                            _metric_chip(f"SARI {sari:.2f}")
                        except Exception:
                            pass
                    if lix_score is not None:
                        _metric_chip(f"LIX {lix_score(simp):.1f}")
                    if wstf_score is not None:
                        _metric_chip(f"WSTF {wstf_score(simp):.1f}")
            metrics_card.classes(remove="hidden")

        ui.button(_t("run_eval", state.lang), on_click=do_evaluate).props("unelevated color=primary size=lg").classes("w-full")


def about_page() -> None:
    with ui.column().classes("w-full gap-4"):
        card = _card()
        with card:
            ui.label("Einfach Deutsch").classes("text-h5 q-mb-sm")
            ui.label(_t("about_text", state.lang)).classes("text-body1")
            ui.separator().classes("q-my-md")
            ui.label(_t("model_info", state.lang)).classes("text-body2")


@ui.page("/")
def index() -> None:
    global sidebar, content_container

    ui.add_css("""
    :root {
      --ed-radius: 12px;
      --ed-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    body.theme-dark {
      --q-primary: #6366f1;
      --q-secondary: #14b8a6;
      --q-dark: #0b0f19;
      --q-dark-page: #0b0f19;
      --ed-bg: #0b0f19;
      --ed-surface: #151b2b;
      --ed-border: #334155;
      --ed-text: #f8fafc;
      --ed-muted: #94a3b8;
      background: var(--ed-bg) !important;
      color: var(--ed-text) !important;
    }
    body.theme-light {
      --q-primary: #4f46e5;
      --q-secondary: #0d9488;
      --q-dark: #ffffff;
      --q-dark-page: #f1f5f9;
      --ed-bg: #f1f5f9;
      --ed-surface: #ffffff;
      --ed-border: #cbd5e1;
      --ed-text: #0f172a;
      --ed-muted: #64748b;
      background: var(--ed-bg) !important;
      color: var(--ed-text) !important;
    }
    body.theme-dark .q-card,
    body.theme-light .q-card {
      background: var(--ed-surface) !important;
      border: 1px solid var(--ed-border) !important;
      border-radius: var(--ed-radius) !important;
      box-shadow: var(--ed-shadow) !important;
      color: var(--ed-text) !important;
    }
    body.theme-dark .q-field__native,
    body.theme-dark .q-field__label,
    body.theme-light .q-field__native,
    body.theme-light .q-field__label {
      color: var(--ed-text) !important;
    }
    body.theme-dark .q-field__control,
    body.theme-light .q-field__control {
      background: var(--ed-surface) !important;
      border-color: var(--ed-border) !important;
    }
    body.theme-dark .q-drawer,
    body.theme-light .q-drawer {
      background: var(--ed-surface) !important;
      border-right: 1px solid var(--ed-border) !important;
    }
    body.theme-dark .q-header,
    body.theme-light .q-header {
      background: var(--ed-surface) !important;
      border-bottom: 1px solid var(--ed-border) !important;
      color: var(--ed-text) !important;
    }
    body.theme-dark .q-menu,
    body.theme-light .q-menu {
      background: var(--ed-surface) !important;
      border: 1px solid var(--ed-border) !important;
    }
    .q-notification__message {
      color: #0f172a !important;
    }
    .q-notification--negative .q-notification__message,
    .q-notification--warning .q-notification__message {
      color: #fff !important;
    }
    """)

    saved_theme = app.storage.user.get("theme", "dark")

    ui.add_body_html(f'''
        <script>
            document.body.classList.add('theme-{saved_theme}');
        </script>
    ''')

    _header()
    if theme_button is not None:
        theme_button.text = "☀️" if saved_theme == "dark" else "🌙"
    sidebar = _sidebar()
    with ui.column().classes("w-full max-w-5xl mx-auto q-pa-md gap-4"):
        content_container = ui.column().classes("w-full gap-4")
        with content_container:
            simplify_page()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host="0.0.0.0",
        port=8501,
        title="Einfach Deutsch",
        favicon="📝",
        reload=False,
        show=False,
        storage_secret="einfach-deutsch-secret",
    )
