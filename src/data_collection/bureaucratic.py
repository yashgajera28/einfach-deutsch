"""Collect and generate German bureaucratic / legal simplification pairs."""

import logging
import random
import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_MUSTER_BASE = "https://www.muster-vorlage.net"
_MUSTER_TEMPLATES = [
    "/mietvertrag-muster/",
    "/arbeitsvertrag-muster/",
]

_STATIC_FALLBACK_PAIRS: list[dict[str, str]] = [
    {
        "complex": "Der Arbeitgeber ist berechtigt, dem Arbeitnehmer Weisungen hinsichtlich Art, Umfang und Ort der Tätigkeit zu erteilen.",
        "simple": "Der Chef darf dem Mitarbeiter sagen, was er wo und wie lange arbeiten soll.",
    },
    {
        "complex": "Der Mieter verpflichtet sich, die Mietsache pfleglich zu behandeln und alle vertragsgemäßen Nebenkosten rechtzeitig zu entrichten.",
        "simple": "Der Mieter muss die Wohnung gut pflegen und die Nebenkosten pünktlich zahlen.",
    },
    {
        "complex": "Im Falle einer Kündigung ist eine schriftliche Mitteilung unter Einhaltung der vereinbarten Kündigungsfrist erforderlich.",
        "simple": "Bei einer Kündigung muss man schriftlich kündigen und die Kündigungsfrist einhalten.",
    },
]

_LEGAL_NOUNS: dict[str, str] = {
    "Vertragspartei": "Vertragspartner",
    "Vertragsparteien": "Vertragspartner",
    "Vermieter": "Vermieter",
    "Mieter": "Mieter",
    "Arbeitgeber": "Chef",
    "Arbeitnehmer": "Mitarbeiter",
    "Rechtsverhältnis": "Rechtsverhältnis",
    "Kündigungsfrist": "Kündigungsfrist",
    "Nebenkosten": "Nebenkosten",
    "Mietverhältnis": "Mietverhältnis",
    "Arbeitsverhältnis": "Arbeitsverhältnis",
    "Pflichtverletzung": "Pflichtverletzung",
    "Schadensersatz": "Schadenersatz",
    "Vergütung": "Bezahlung",
    "vertragsgemäßen": "vertraglichen",
    "schriftliche": "schriftliche",
    "wesentlichen": "wichtigsten",
}

_PHRASE_REPLACEMENTS: dict[str, str] = {
    "ist verpflichtet": "muss",
    "ist berechtigt": "darf",
    "verpflichtet sich": "muss sich",
    "hat ... zu": "muss",
    "hat dem": "muss dem",
    "hat die": "muss die",
    "auszuhändigen": "geben",
    "zu entrichten": "zahlen",
    "zu zahlen": "zahlen",
    "zu behandeln": "behandeln",
    "zu vermeiden": "vermeiden",
    "zu beenden": "beenden",
    "nicht untervermieten": "nicht weitervermieten",
    "verlangen": "fordern",
    "verantwortlich": "hat die Schuld",
    "vorsätzlich": "absichtlich",
    "fahrlässig": "leichtfertig",
    "Untervermietung": "Weitervermietung",
}

_PASSIVE_PATTERNS = [
    (re.compile(r"\b(wurde|wurden)\s+([a-zA-ZäöüßÄÖÜ]+)\b"), r"hatte \2"),
]


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize punctuation."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _split_clauses(sentence: str) -> list[str]:
    """Split a long sentence at common clause boundaries."""
    parts = re.split(r"[,;](?=\s+(und|oder|sowie|sowohl|wobei|falls|sofern))", sentence)
    return [p.strip(" ,;") for p in parts if p.strip(" ,;")]


def _simplify_sentence(sentence: str) -> str:
    """Apply rule-based simplifications to a German sentence."""
    sentence = _clean_text(sentence)
    for complex_form, simple_form in _LEGAL_NOUNS.items():
        sentence = re.sub(rf"\b{complex_form}\b", simple_form, sentence)

    for complex_form, simple_form in _PHRASE_REPLACEMENTS.items():
        sentence = sentence.replace(complex_form, simple_form)

    # Remove commas that appear directly after a modal verb.
    sentence = re.sub(r"\b(muss|darf|kann|soll|will)\s*,\s*", r"\1 ", sentence)

    # Convert passive-ish constructions heuristically.
    for pattern, replacement in _PASSIVE_PATTERNS:
        sentence = pattern.sub(replacement, sentence)

    # Split into shorter clauses and rejoin with simple connectors.
    clauses = _split_clauses(sentence)
    if len(clauses) > 1:
        sentence = ". ".join(clauses) + "."

    sentence = sentence.replace("..", ".").replace(". .", ".")
    return _clean_text(sentence)


def scrape_muster_vorlage(url: str | None = None, delay: float = 1.0) -> list[dict[str, str]]:
    """Scrape contract clauses from Muster-Vorlage.net.

    Args:
        url: Specific page to scrape; if ``None`` the configured templates
            are scraped.
        delay: Seconds to sleep between requests.

    Returns:
        A list of complex/simple pairs with ``source`` set to
        ``bureaucratic``. Falls back to static pairs on network failure.
    """
    urls = [url] if url else [f"{_MUSTER_BASE}{path}" for path in _MUSTER_TEMPLATES]
    pairs: list[dict[str, str]] = []

    for page_url in urls:
        try:
            response = requests.get(page_url, timeout=30, headers={"User-Agent": "einfach-deutsch-data/0.1"})
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Scraping failed for %s: %s", page_url, exc)
            return [dict(p, source="bureaucratic") for p in _STATIC_FALLBACK_PAIRS]

        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("article") or soup.find("main") or soup.find("div", class_="entry-content")
        if content is None:
            logger.warning("Could not locate article body at %s", page_url)
            return [dict(p, source="bureaucratic") for p in _STATIC_FALLBACK_PAIRS]

        paragraphs = [p.get_text() for p in content.find_all(["p", "li"])]
        for paragraph in paragraphs:
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                sentence = _clean_text(sentence)
                if len(sentence.split()) < 8:
                    continue
                simple = _simplify_sentence(sentence)
                if simple != sentence:
                    pairs.append({"complex": sentence, "simple": simple, "source": "bureaucratic"})

        time.sleep(delay)

    if not pairs:
        return [dict(p, source="bureaucratic") for p in _STATIC_FALLBACK_PAIRS]
    return pairs


def generate_synthetic_pairs(seed: int | None = None, count: int = 80) -> list[dict[str, str]]:
    """Generate rule-based bureaucratic simplification pairs.

    Args:
        seed: Random seed for reproducibility.
        count: Number of pairs to generate.

    Returns:
        Synthetic pairs with ``source`` set to ``bureaucratic``.
    """
    rng = random.Random(seed)

    templates: list[dict[str, Any]] = [
        {
            "complex": "Der {a} ist verpflichtet, dem {b} die vereinbarte Vergütung fristgerecht zu zahlen.",
            "slots": {"a": ["Arbeitgeber", "Vermieter"], "b": ["Arbeitnehmer", "Mieter"]},
        },
        {
            "complex": "Der {b} hat die ihm überlassene Sache sorgfältig zu behandeln und Schäden zu vermeiden.",
            "slots": {"b": ["Arbeitnehmer", "Mieter", "Vertragspartner"]},
        },
        {
            "complex": "Jede Vertragspartei kann das {relation} unter Einhaltung der gesetzlichen Kündigungsfrist beenden.",
            "slots": {"relation": ["Arbeitsverhältnis", "Mietverhältnis"]},
        },
        {
            "complex": "Der Mieter darf die Wohnung ohne {modus} schriftliche Zustimmung des Vermieters nicht untervermieten.",
            "slots": {"modus": ["vorherige", "ausdrückliche"]},
        },
        {
            "complex": "Bei {fault} kann der geschädigte Vertragspartner Schadensersatz verlangen.",
            "slots": {"fault": ["Pflichtverletzung", "Vertragsbruch"]},
        },
        {
            "complex": "Der Arbeitnehmer erhält den gesetzlichen Mindesturlaub von {days} Werktagen im Jahr.",
            "slots": {"days": ["20", "24", "30"]},
        },
        {
            "complex": "Änderungen dieses Vertrages bedürfen zu ihrer Wirksamkeit der {form}.",
            "slots": {"form": ["Schriftform", "schriftlichen Zustimmung beider Parteien"]},
        },
        {
            "complex": "Der Vermieter überlässt dem Mieter die Wohnung zum {use}.",
            "slots": {"use": ["vertragsgemäßen Gebrauch", "Wohnen"]},
        },
        {
            "complex": "Der Arbeitgeber hat dem Arbeitnehmer eine schriftliche Aufzeichnung über die {conditions} auszuhändigen.",
            "slots": {"conditions": ["wesentlichen Vertragsbedingungen", "wichtigsten Arbeitsbedingungen"]},
        },
        {
            "complex": "Der Mieter ist für Schäden verantwortlich, die er {intent} verursacht.",
            "slots": {"intent": ["vorsätzlich oder fahrlässig", "absichtlich oder leichtfertig"]},
        },
    ]

    prefixes: list[str] = [
        "",
        "Zusätzlich gilt: ",
        "Vertraglich vereinbart wird: ",
        "Es gilt außerdem: ",
    ]

    seen: set[str] = set()
    pairs: list[dict[str, str]] = []
    attempts = 0
    while len(pairs) < count and attempts < count * 10:
        attempts += 1
        template = rng.choice(templates)
        slots = template.get("slots", {})
        values = {key: rng.choice(options) for key, options in slots.items()}
        complex_text = template["complex"].format(**values)
        prefix = rng.choice(prefixes)
        complex_text = _clean_text(prefix + complex_text)
        simple_text = _simplify_sentence(complex_text)
        key = complex_text + "\n" + simple_text
        if simple_text != complex_text and key not in seen:
            seen.add(key)
            pairs.append({"complex": complex_text, "simple": simple_text, "source": "bureaucratic"})

    return pairs
