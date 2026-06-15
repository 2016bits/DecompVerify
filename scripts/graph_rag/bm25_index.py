"""Tantivy-backed BM25 index over the HOVER (HotpotQA-style) corpus.

The HOVER corpus distributed with the project lives at
``/mnt/data/yangjun/data/HOVER/corpus/jsonl_corpus/hotpotqa_corpus.jsonl``.
Each line is::

    {"id": "<wiki title>", "contents": "<title>\n <sent>\n <sent>\n ..."}

We build a tantivy index keyed by ``title`` once and reuse it for every
claim / atomic fact lookup. The index is created on first use and stored
under ``data/graph_rag_cache/tantivy_index``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import tantivy


DEFAULT_CORPUS_PATH = "/mnt/data/yangjun/data/HOVER/corpus/jsonl_corpus/hotpotqa_corpus.jsonl"
DEFAULT_INDEX_PATH = "./data/graph_rag_cache/tantivy_index"


def build_schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("title", stored=True, tokenizer_name="default")
    builder.add_text_field("contents", stored=True, tokenizer_name="default")
    return builder.build()


def build_index(
    corpus_path: str = DEFAULT_CORPUS_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
    log_every: int = 200_000,
) -> str:
    """Build the BM25 index from the JSONL corpus. Idempotent."""

    index_dir = Path(index_path)
    if index_dir.exists():
        # Only treat as "already built" if at least one segment file is
        # present – stray ``.managed.json`` / ``.tantivy-*.lock`` from an
        # interrupted previous build should not block a rebuild.
        has_segment = any(
            p.suffix in {".idx", ".store", ".pos"}
            for p in index_dir.iterdir()
        )
        if has_segment:
            return str(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    schema = build_schema()
    index = tantivy.Index(schema, path=str(index_dir))
    writer = index.writer(heap_size=512_000_000, num_threads=2)

    n = 0
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = doc.get("id") or ""
            contents = doc.get("contents") or ""
            if not title:
                continue
            doc = tantivy.Document()
            doc.add_text("title", title)
            doc.add_text("contents", contents)
            writer.add_document(doc)
            n += 1
            if n % log_every == 0:
                print(f"[bm25_index] indexed {n:,} docs ...", flush=True)
    writer.commit()
    writer.wait_merging_threads()
    print(f"[bm25_index] built BM25 index with {n:,} documents at {index_dir}")
    return str(index_dir)


def _sanitize_query(text: str) -> str:
    """Strip characters that tantivy's query parser cannot handle directly."""
    bad = set("\"+-^~*?:\\/(){}[]")
    return "".join(" " if ch in bad else ch for ch in text).strip()


class BM25Searcher:
    """Thin wrapper around tantivy.Searcher with on-disk reuse."""

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH):
        self.index_path = index_path
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_path}. Run build_index() first."
            )
        self.index = tantivy.Index.open(index_path)
        self.index.reload()
        self.searcher = self.index.searcher()
        self.schema = self.index.schema

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 20) -> List[Tuple[str, float, str]]:
        """Return ``(title, score, contents)`` triples for top ``k`` hits."""
        q = _sanitize_query(query or "")
        if not q:
            return []
        try:
            parsed = self.index.parse_query(q, ["title", "contents"])
        except Exception:
            return []
        out = []
        for score, addr in self.searcher.search(parsed, limit=k).hits:
            doc = self.searcher.doc(addr)
            title_field = doc.get_first("title") or ""
            contents_field = doc.get_first("contents") or ""
            out.append((title_field, float(score), contents_field))
        return out

    # ------------------------------------------------------------ title-only
    def search_title(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        """BM25 against the ``title`` field only.

        For multi-hop fact checking, the answer-entity page almost always has
        a Wikipedia title that literally names the target entity. Issuing a
        title-only BM25 for each fact's ``target_surfaces`` / ``exact_name``
        gives those anchor pages a strong, high-precision entry into the
        candidate pool even when their body text has weak overlap with the
        claim. Used by the role-aware gather stage in ``retriever`` to back
        up the broader ``[title, contents]`` sweep.
        """
        q = _sanitize_query(query or "")
        if not q:
            return []
        try:
            parsed = self.index.parse_query(q, ["title"])
        except Exception:
            return []
        out = []
        for score, addr in self.searcher.search(parsed, limit=k).hits:
            doc = self.searcher.doc(addr)
            title_field = doc.get_first("title") or ""
            contents_field = doc.get_first("contents") or ""
            out.append((title_field, float(score), contents_field))
        return out

    # ------------------------------------------------------------- multi-query
    def multi_search(
        self,
        queries: Sequence[str],
        k_per_query: int = 10,
        max_results: int = 50,
    ) -> List[Tuple[str, float, str, List[str]]]:
        """Run multiple queries, fuse with reciprocal-rank-style scoring.

        Returns ``(title, fused_score, contents, hit_queries)`` ordered by
        fused score. ``hit_queries`` enumerates which queries this doc was
        retrieved by – useful for downstream "role" tagging.
        """
        agg: dict[str, dict] = {}
        for q in queries:
            hits = self.search(q, k=k_per_query)
            for rank, (title, score, contents) in enumerate(hits):
                slot = agg.setdefault(
                    title,
                    {"score": 0.0, "contents": contents, "hits": []},
                )
                # Reciprocal-rank fusion contribution plus raw BM25.
                slot["score"] += 1.0 / (rank + 60.0) + 0.05 * score
                slot["hits"].append(q)
        ranked = sorted(agg.items(), key=lambda kv: -kv[1]["score"])
        out: List[Tuple[str, float, str, List[str]]] = []
        for title, slot in ranked[:max_results]:
            out.append((title, slot["score"], slot["contents"], slot["hits"]))
        return out


# Convenience entry point if run as ``python -m scripts.graph_rag.bm25_index``.
if __name__ == "__main__":  # pragma: no cover - utility CLI
    import argparse

    parser = argparse.ArgumentParser(description="Build the GraphRAG BM25 index.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()
    build_index(args.corpus, args.index)
