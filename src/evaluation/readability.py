"""German readability metrics: LIX and Wiener Sachtextformel."""

import re


_LONG_WORD_THRESHOLD = 6

_SYLLABLE_EXCEPTIONS: dict[str, int] = {
    "die": 1,
    "der": 1,
    "das": 1,
    "und": 1,
    "zu": 1,
    "in": 1,
    "ist": 1,
    "von": 1,
    "für": 1,
    "mit": 1,
    "auf": 1,
}


_VOWEL_GROUPS = re.compile(r"[aeiouäöüy]+", re.IGNORECASE)
_DIPHTHONGS = {"au", "ei", "eu", "äu", "ie"}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _words(text: str) -> list[str]:
    """Extract alphabetic words."""
    return re.findall(r"[a-zäöüßA-ZÄÖÜ]+(?:-[a-zäöüßA-ZÄÖÜ]+)?", text)


def _count_syllables(word: str) -> int:
    """Estimate German syllable count."""
    lowered = word.lower()
    if lowered in _SYLLABLE_EXCEPTIONS:
        return _SYLLABLE_EXCEPTIONS[lowered]

    groups = _VOWEL_GROUPS.findall(lowered)
    count = len(groups)

    for diph in _DIPHTHONGS:
        count -= lowered.count(diph)

    return max(count, 1)


def lix_score(text: str) -> float:
    """Compute LIX readability score."""
    sentences = _split_sentences(text)
    words = _words(text)

    num_sentences = len(sentences) if sentences else 1
    num_words = len(words) if words else 1
    long_words = sum(1 for w in words if len(w) > _LONG_WORD_THRESHOLD)

    return num_words / num_sentences + long_words * 100 / num_words


def wstf_score(text: str) -> float:
    """Compute Wiener Sachtextformel readability score."""
    sentences = _split_sentences(text)
    words = _words(text)

    num_sentences = len(sentences) if sentences else 1
    num_words = len(words) if words else 1

    ms = num_words / num_sentences
    syllables = [_count_syllables(w) for w in words]
    total_syllables = sum(syllables)
    sl = total_syllables / num_words
    iw = sum(1 for s in syllables if s > 3) / num_words * 100
    es = sum(1 for s in syllables if s == 1) / num_words * 100

    return 0.1935 * ms + 0.1672 * sl + 0.1297 * iw - 0.0327 * es - 0.875


def readability_delta(before: str, after: str) -> dict[str, float]:
    """Return readability metrics before, after, and their deltas."""
    before_lix = lix_score(before)
    after_lix = lix_score(after)
    before_wstf = wstf_score(before)
    after_wstf = wstf_score(after)

    return {
        "lix_before": before_lix,
        "lix_after": after_lix,
        "lix_delta": before_lix - after_lix,
        "wstf_before": before_wstf,
        "wstf_after": after_wstf,
        "wstf_delta": before_wstf - after_wstf,
    }
