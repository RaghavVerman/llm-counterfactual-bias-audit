"""Validated word lists for stereotype-associated language.

Entries ending in ``*`` are stems (prefix matches), following the convention of
the original word lists. These replace user-typed "bias words": the categories
come from published research, so the analysis is not circular.

Sources
-------
* Gaucher, Friesen & Kay (2011). "Evidence that gendered wording in job
  advertisements exists and sustains gender inequality." JPSP 101(1).
  -> masculine-coded / feminine-coded word lists.
* Madera, Hebl & Martin (2009). "Gender and letters of recommendation for
  academia: agentic and communal differences." J. Applied Psychology 94(6).
  -> agentic / communal descriptor categories.

The agentic/communal lists below merge both sources (lower-cased, deduplicated).
They are adapted, not verbatim reproductions; cite them as "adapted from".
"""

from __future__ import annotations

import re
from functools import lru_cache

LEXICONS: dict[str, list[str]] = {
    "agentic": [
        "active", "adventurous", "aggress*", "ambitio*", "analy*", "assert*",
        "athlet*", "autonom*", "boast*", "challeng*", "compet*", "confident",
        "courag*", "decide", "decisive", "decision*", "determin*", "domina*",
        "force*", "headstrong", "hierarch*", "independen*", "individual*",
        "intellect*", "lead*", "logic*", "objective", "outspoken", "persist*",
        "principle*", "self-confiden*", "self-relian*", "self-sufficien*",
        "superior", "achiev*", "accomplish*", "brilliant", "skill*", "expert*",
        "authorit*", "powerful", "strong*", "daring", "driven",
    ],
    "communal": [
        "affection*", "cheer*", "commit*", "communal", "compassion*",
        "connect*", "considerate", "cooperat*", "depend*", "emotiona*",
        "empath*", "gentle", "honest", "interpersona*", "interdependen*",
        "kind", "kindness", "kinship", "loyal*", "modest*", "nurtur*",
        "pleasant*", "polite", "quiet*", "sensitiv*", "submissive",
        "support*", "sympath*", "tender*", "together*", "trust*",
        "understand*", "warm*", "yield*", "helpful", "caring", "care",
        "sweet", "agreeable", "friendly", "patient",
    ],
    # Physical-appearance and family-role words: a common axis of gendered
    # description in generated biographies (see also BOLD, Dhamala et al. 2021).
    "appearance": [
        "beaut*", "pretty", "attractive", "gorgeous", "handsome", "sexy",
        "slim", "lovely", "elegant", "cute", "stunning",
    ],
    "family": [
        "wife", "husband", "mother", "father", "children", "child", "kids",
        "family", "married", "marriage", "home*", "housewife", "baby",
    ],
}


@lru_cache(maxsize=None)
def _pattern(category: str) -> re.Pattern:
    parts = []
    for entry in LEXICONS[category]:
        if entry.endswith("*"):
            parts.append(re.escape(entry[:-1]) + r"[\w-]*")
        else:
            parts.append(re.escape(entry))
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


_WORD = re.compile(r"\b[\w'-]+\b")


def lexicon_matches(text: str, category: str) -> list[str]:
    return [m.group(0) for m in _pattern(category).finditer(text)]


def lexicon_scores(text: str) -> dict[str, float]:
    """Matches per 100 words for every category, plus an agentic-communal index."""
    n_words = max(len(_WORD.findall(text)), 1)
    out = {}
    for cat in LEXICONS:
        out[f"lex_{cat}"] = 100.0 * len(lexicon_matches(text, cat)) / n_words
    out["lex_agentic_minus_communal"] = out["lex_agentic"] - out["lex_communal"]
    return out
