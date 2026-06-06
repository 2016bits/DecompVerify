"""Sentence-level view over the HOVER wiki corpus.

The corpus stores each Wikipedia page as one JSONL record where
``contents`` is a newline-separated list of sentences (the first
sentence often repeats the title). For our retrieval we need both:

* the raw doc text for BM25 scoring,
* a clean list of ``(title, sent_idx, sentence)`` triples for graph
  construction and evaluation against HOVER gold ``sent_keys``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


_SENT_SPLIT_RE = re.compile(r"\n+|(?<=[.!?])\s+(?=[A-Z\"'(\[])")


@dataclass
class Sentence:
    title: str
    sent_idx: int
    text: str

    @property
    def sid(self) -> str:
        return f"{self.title}::{self.sent_idx}"


def split_sentences(contents: str) -> List[str]:
    """Split a HOVER doc into clean sentences keeping their order."""
    if not contents:
        return []
    parts = []
    for chunk in contents.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Some sentences contain embedded newlines after periods; split
        # again to keep granularity close to HOVER's sent_idx convention.
        sub = _SENT_SPLIT_RE.split(chunk)
        for s in sub:
            s = s.strip()
            if s:
                parts.append(s)
    return parts


def to_sentence_list(title: str, contents: str) -> List[Sentence]:
    sents = split_sentences(contents)
    # HOVER's gold ``index`` is 0-based against the sentence stream that
    # results from the same split rule, with the title-only first line
    # counted as sentence 0. We keep the very first occurrence too.
    return [Sentence(title=title, sent_idx=i, text=s) for i, s in enumerate(sents)]


class DocStore:
    """In-memory lookup of doc title -> sentence list, populated lazily."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[Sentence]] = {}

    def add_doc(self, title: str, contents: str) -> List[Sentence]:
        if title not in self._cache:
            self._cache[title] = to_sentence_list(title, contents)
        return self._cache[title]

    def get(self, title: str) -> List[Sentence]:
        return self._cache.get(title, [])

    def has(self, title: str) -> bool:
        return title in self._cache

    def all_sentences(self) -> List[Sentence]:
        out: List[Sentence] = []
        for sents in self._cache.values():
            out.extend(sents)
        return out

    def __len__(self) -> int:
        return sum(len(v) for v in self._cache.values())
