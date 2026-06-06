"""Lightweight entity extraction for graph construction.

We avoid spaCy because the project's NumPy pin breaks current builds.
For HOVER-style claims the cheapest reliable signal is:

* the wiki **titles** of the candidate docs themselves – these are
  guaranteed entity strings,
* runs of **capitalised words** inside sentences (proper nouns),
* quoted phrases such as ``"Life Goes On"``.

This file produces the entity surface forms used by the heterogeneous
graph. It is deliberately recall-oriented; downstream PPR and the
cross-encoder rerank will filter the noise.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set


_QUOTE_RE = re.compile(r"[\"“”]([A-Z][^\"“”]{1,80})[\"“”]")
_PAREN_RE = re.compile(r"\(([^)]{1,80})\)")
_CAP_SEQ_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z'’&.\-]+)(?:\s+(?:of|the|and|de|von|van|du|le|la|al|el)\s+[A-Z][A-Za-z'’&.\-]+|\s+[A-Z][A-Za-z'’&.\-]+){0,4}\b"
)

_STOP_TITLES = {
    "The", "A", "An", "In", "On", "Of", "At", "By", "For", "With",
    "And", "Or", "But", "It", "This", "That", "These", "Those",
}


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().strip(".,;:")


def extract_from_sentence(sentence: str) -> List[str]:
    """Return a list of unique entity surface forms inside a sentence."""
    if not sentence:
        return []
    found: List[str] = []
    seen: Set[str] = set()

    def _add(token: str) -> None:
        token = _normalize(token)
        if not token or len(token) < 2:
            return
        if token in _STOP_TITLES:
            return
        key = token.lower()
        if key in seen:
            return
        seen.add(key)
        found.append(token)

    for m in _QUOTE_RE.finditer(sentence):
        _add(m.group(1))
    for m in _CAP_SEQ_RE.finditer(sentence):
        _add(m.group(0))
    return found


def extract_from_title(title: str) -> List[str]:
    """Treat the doc title as the canonical entity, plus any (parenthetical)."""
    if not title:
        return []
    out = [_normalize(title)]
    head = title
    # Strip a single trailing parenthetical so "Life Goes On (Fergie song)"
    # also yields the bare "Life Goes On" surface form.
    m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", title)
    if m:
        head = m.group(1).strip()
        if head:
            out.append(_normalize(head))
    # Strip leading articles for matching purposes.
    if head.lower().startswith(("the ", "a ", "an ")):
        bare = head.split(" ", 1)[1].strip()
        if bare:
            out.append(_normalize(bare))
    # Deduplicate while preserving order.
    seen: Set[str] = set()
    uniq: List[str] = []
    for t in out:
        k = t.lower()
        if k and k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


def extract_from_fact_text(fact_text: str, constraint: dict | None = None) -> List[str]:
    """Pull entity-like cues from atomic fact text and its constraint slots."""
    found: List[str] = []
    seen: Set[str] = set()

    def _add(token: str) -> None:
        token = _normalize(token)
        if not token or len(token) < 2:
            return
        # Drop slot placeholders such as ``?province_1``.
        if token.startswith("?"):
            return
        key = token.lower()
        if key in seen:
            return
        seen.add(key)
        found.append(token)

    for m in _CAP_SEQ_RE.finditer(fact_text or ""):
        _add(m.group(0))

    constraint = constraint or {}
    for slot in ("exact_name", "location", "title_role"):
        for v in constraint.get(slot, []) or []:
            if isinstance(v, str):
                _add(v)

    return found
