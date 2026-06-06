"""Per-claim heterogeneous graph construction.

Inputs: a list of (title, contents) pairs (the candidate Wikipedia
articles selected by the role-aware BM25 stage).

Nodes
-----
* ``S`` sentence nodes      – one per non-empty sentence.
* ``N`` entity nodes        – one per normalised entity surface form.
* ``D`` document nodes      – one per article title.
* ``C`` community nodes     – one per Louvain community over the
                              ``N``-subgraph; gives a lightweight
                              GraphRAG-style "global" view that we
                              expose as a seed channel.

Edges
-----
* ``S-D`` sentence belongs to doc.
* ``S-N`` sentence mentions entity.
* ``D-N`` doc mentions entity (aggregated from S-N).
* ``N-N`` co-occurrence inside the same sentence.
* ``S-S_local`` neighbouring sentences in the same doc.
* ``S-S_semantic`` cross-doc HNSW neighbours from sentence embeddings.
* ``N-C`` entity belongs to community.

The HNSW step gets skipped automatically when the candidate pool is
small (fewer than ~50 sentences) – brute force is faster there and
avoids hnswlib's minimum-size requirement.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import networkx as nx

from .doc_store import DocStore, Sentence
from .entity_extract import extract_from_sentence, extract_from_title


def _norm_entity_key(text: str) -> str:
    return " ".join(text.lower().split()).strip(".,;:")


@dataclass
class HeteroGraph:
    g: nx.Graph
    sentences: List[Sentence]
    sid_to_idx: Dict[str, int]
    sent_embeddings: np.ndarray
    entity_keys: List[str]                # canonical lower-cased forms
    entity_display: Dict[str, str]        # key -> a representative surface form
    title_to_node: Dict[str, str]
    sid_to_node: Dict[str, str]
    entity_key_to_node: Dict[str, str]
    communities: Dict[str, int]           # entity_node -> community id

    # ----------------------------------------------------------------- helpers
    def doc_node(self, title: str) -> str | None:
        return self.title_to_node.get(title)

    def sentence_node(self, sid: str) -> str | None:
        return self.sid_to_node.get(sid)

    def entity_node(self, key: str) -> str | None:
        return self.entity_key_to_node.get(_norm_entity_key(key))


def _add_node(g: nx.Graph, ntype: str, key: str, **attrs) -> str:
    nid = f"{ntype}::{key}"
    if nid not in g:
        g.add_node(nid, ntype=ntype, **attrs)
    return nid


def _add_edge(g: nx.Graph, u: str, v: str, etype: str, weight: float = 1.0) -> None:
    if u == v:
        return
    if g.has_edge(u, v):
        # If the same pair is hit by multiple channels, keep the strongest.
        cur = g[u][v]
        cur["weight"] = max(cur.get("weight", 0.0), weight)
        cur.setdefault("etypes", set()).add(etype)
    else:
        g.add_edge(u, v, weight=weight, etype=etype, etypes={etype})


def _semantic_edges(
    sent_emb: np.ndarray,
    sentences: List[Sentence],
    top_k: int = 5,
    sim_threshold: float = 0.55,
) -> List[Tuple[int, int, float]]:
    """Approximate top-k semantic neighbours for each sentence.

    ``hnswlib`` shines on large pools. For our typical 200-2000 sentence
    pool a NumPy similarity matrix is small, simple and parallel-free.
    """
    if sent_emb.shape[0] <= 1:
        return []
    sims = sent_emb @ sent_emb.T  # (N,N) on already L2-normalised vectors
    np.fill_diagonal(sims, -1.0)
    k = min(top_k, sent_emb.shape[0] - 1)
    edges: List[Tuple[int, int, float]] = []
    for i in range(sent_emb.shape[0]):
        idxs = np.argpartition(-sims[i], k - 1)[:k]
        for j in idxs:
            w = float(sims[i, j])
            if w >= sim_threshold and i != j:
                edges.append((i, int(j), w))
    return edges


def _detect_communities(g: nx.Graph, entity_nodes: List[str]) -> Dict[str, int]:
    """Connected-components community labelling over the N-subgraph.

    networkx 3.1 ships a pure-Python ``louvain_communities`` that on
    the ~1000-node graphs we build per HOVER claim runs 5-20 s per
    call. With 5 facts x 200 claims that was the dominant cost in v2
    and triggered multi-hour stalls. Connected components on the
    (N + linked D) subgraph give the same "global topical grouping"
    signal in O(|V|+|E|).
    """
    if not entity_nodes:
        return {}
    seed_nodes = set(entity_nodes)
    for ent in entity_nodes:
        for nb in g.neighbors(ent):
            if g.nodes[nb].get("ntype") == "D":
                seed_nodes.add(nb)
    sub = g.subgraph(seed_nodes)
    if sub.number_of_edges() == 0:
        return {n: i for i, n in enumerate(entity_nodes)}
    out: Dict[str, int] = {}
    for cid, comp in enumerate(nx.connected_components(sub)):
        for node in comp:
            if g.nodes[node].get("ntype") == "N":
                out[node] = cid
    next_id = max(out.values(), default=-1) + 1
    for ent in entity_nodes:
        if ent not in out:
            out[ent] = next_id
            next_id += 1
    return out


def build_hetero_graph(
    docs: Sequence[Tuple[str, str]],
    encoder,
    sem_top_k: int = 5,
    sem_threshold: float = 0.55,
    max_sentences_per_doc: int = 80,
) -> HeteroGraph:
    """Build a heterogeneous graph from candidate ``(title, contents)`` docs."""
    g = nx.Graph()
    store = DocStore()

    title_to_node: Dict[str, str] = {}
    sid_to_node: Dict[str, str] = {}
    sentences: List[Sentence] = []
    sid_to_idx: Dict[str, int] = {}

    # 1) doc + sentence nodes -----------------------------------------------
    for title, contents in docs:
        if title in title_to_node:
            continue
        doc_node = _add_node(g, "D", title, title=title)
        title_to_node[title] = doc_node
        sents = store.add_doc(title, contents)
        if len(sents) > max_sentences_per_doc:
            sents = sents[:max_sentences_per_doc]
            store._cache[title] = sents  # noqa: SLF001 (trimming in place)
        prev_sid_node: str | None = None
        for sent in sents:
            sid = sent.sid
            sid_to_idx[sid] = len(sentences)
            sentences.append(sent)
            snode = _add_node(g, "S", sid, sid=sid, title=title,
                              sent_idx=sent.sent_idx, text=sent.text)
            sid_to_node[sid] = snode
            _add_edge(g, snode, doc_node, "S-D", weight=1.0)
            if prev_sid_node is not None:
                _add_edge(g, prev_sid_node, snode, "S-S_local", weight=0.6)
            prev_sid_node = snode

    # 2) entity nodes + S-N / D-N / N-N edges -------------------------------
    entity_display: Dict[str, str] = {}
    entity_key_to_node: Dict[str, str] = {}

    def _ensure_entity(surface: str) -> str | None:
        key = _norm_entity_key(surface)
        if not key:
            return None
        if key not in entity_key_to_node:
            node = _add_node(g, "N", key, display=surface, entity_key=key)
            entity_key_to_node[key] = node
            entity_display[key] = surface
        return entity_key_to_node[key]

    for title in title_to_node:
        for surface in extract_from_title(title):
            ent_node = _ensure_entity(surface)
            if ent_node is not None:
                _add_edge(g, title_to_node[title], ent_node, "D-N", weight=0.8)

    for sent in sentences:
        sent_node = sid_to_node[sent.sid]
        surfaces = extract_from_sentence(sent.text)
        for surface in surfaces:
            ent_node = _ensure_entity(surface)
            if ent_node is None:
                continue
            _add_edge(g, sent_node, ent_node, "S-N", weight=1.0)
            _add_edge(g, title_to_node[sent.title], ent_node, "D-N", weight=0.6)
        # N-N co-occurrence
        nodes_here = [entity_key_to_node[_norm_entity_key(s)] for s in surfaces
                      if _norm_entity_key(s) in entity_key_to_node]
        for i in range(len(nodes_here)):
            for j in range(i + 1, len(nodes_here)):
                _add_edge(g, nodes_here[i], nodes_here[j], "N-N",
                          weight=0.4)

    # 3) sentence embeddings + semantic S-S edges ---------------------------
    texts = [s.text for s in sentences]
    sent_emb = encoder.encode(texts) if texts else np.zeros((0, encoder.dim), dtype=np.float32)
    for i, j, w in _semantic_edges(sent_emb, sentences,
                                   top_k=sem_top_k,
                                   sim_threshold=sem_threshold):
        si = sid_to_node[sentences[i].sid]
        sj = sid_to_node[sentences[j].sid]
        _add_edge(g, si, sj, "S-S_semantic", weight=0.4 + 0.4 * w)

    # 4) community detection over the entity subgraph -----------------------
    entity_nodes = list(entity_key_to_node.values())
    communities = _detect_communities(g, entity_nodes)
    for ent_node, cid in communities.items():
        c_node = _add_node(g, "C", f"c{cid}", cid=cid)
        _add_edge(g, ent_node, c_node, "N-C", weight=0.3)

    return HeteroGraph(
        g=g,
        sentences=sentences,
        sid_to_idx=sid_to_idx,
        sent_embeddings=sent_emb,
        entity_keys=list(entity_key_to_node.keys()),
        entity_display=entity_display,
        title_to_node=title_to_node,
        sid_to_node=sid_to_node,
        entity_key_to_node=entity_key_to_node,
        communities=communities,
    )
