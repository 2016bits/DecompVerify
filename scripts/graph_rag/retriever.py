"""End-to-end GraphRAG retrieval for a single HOVER claim.

Pipeline
========

1. **Role-aware BM25 gather** (``_gather_candidate_docs``):
   * one ``claim`` query, one ``fact`` query per atomic fact, optional
     ``constraint`` and ``critical`` queries. We allocate a per-role
     quota (rather than a simple union) so deep-leaf facts cannot be
     drowned by claim-level hits.

2. **Graph construction** (``graph_build.build_hetero_graph``): a local
   S/N/D/C heterogeneous graph with neighbour, mention and HNSW
   semantic edges.

3. **Per-fact seeded PPR** (``_retrieve_for_fact``): facts are processed
   in topological order. For each fact we build a seed mass over
   semantic, entity, constraint, dependency and community channels and
   run a single restart-based PPR pass.

4. **Coverage-aware assembly** (``assembly.assemble_evidence``): we
   trade strong direct-support sentences against bridge / dependency
   closure sentences with a "fact bundle" scheme adapted from
   ``plan5.6``.

The output is a JSON-serialisable dict that can be dropped into
``get_answer.py`` in place of the original ``gold_evidence`` text.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .bm25_index import BM25Searcher
from .doc_store import DocStore, Sentence
from .encoder import SentenceEncoder
from .entity_extract import extract_from_fact_text, extract_from_sentence
from .graph_build import HeteroGraph, build_hetero_graph
from .ppr import personalised_pagerank


VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")


# ---------------------------------------------------------------------------- helpers
def _topological_order(question_items: Sequence[dict]) -> List[dict]:
    """Topologically sort question items by ``rely_on`` edges."""
    by_id = {q.get("fact_id"): q for q in question_items}
    indeg = {fid: 0 for fid in by_id}
    children: Dict[str, List[str]] = {fid: [] for fid in by_id}
    for fid, q in by_id.items():
        for dep in q.get("rely_on", []) or []:
            if dep in by_id:
                indeg[fid] += 1
                children[dep].append(fid)
    ready = [fid for fid, d in indeg.items() if d == 0]
    out: List[dict] = []
    while ready:
        ready.sort()
        fid = ready.pop(0)
        out.append(by_id[fid])
        for child in children.get(fid, []):
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
    # Append any leftover (cyclic) facts to be safe.
    seen = {q["fact_id"] for q in out}
    for fid, q in by_id.items():
        if fid not in seen:
            out.append(q)
    return out


def _fact_search_text(qitem: dict, bindings: Dict[str, str]) -> str:
    base = qitem.get("fact_text") or qitem.get("main_question") or ""
    # Best-effort binding substitution to lift "?province_1" to a real name.
    def _sub(m):
        return bindings.get(m.group(0), m.group(0))
    base = VAR_PATTERN.sub(_sub, base)
    hints = " ".join(qitem.get("search_hints", []) or [])
    return f"{base} {hints}".strip()


def _constraint_query(qitem: dict) -> Optional[str]:
    cons = qitem.get("constraint") or {}
    bits: List[str] = []
    for slot in ("time", "quantity", "comparison", "location", "title_role",
                 "exact_name", "unique_modifier"):
        for v in cons.get(slot, []) or []:
            if isinstance(v, str) and not v.startswith("?"):
                bits.append(v)
    if cons.get("negation"):
        bits.append(str(cons["negation"]))
    if not bits:
        return None
    return " ".join(bits)


# ---------------------------------------------------------------------------- BM25 stage
def _claim_keywords(claim: str) -> str:
    """Pull the rare, high-IDF cues out of the claim to supplement
    every fact-level BM25 query.

    BM25 on a bare atomic fact like "?statement_1 says e-Borders ..."
    drops the unresolved variable and is too short to disambiguate
    "e-Borders" from "River Eea". Always pairing the fact query with
    a compact claim-keyword bag fixes the noisy-doc problem we saw on
    f7 of the UK e-Borders 3-hop sample.
    """
    # The claim itself works well as a supplementary query – BM25
    # naturally down-weights frequent words and the boost benefits
    # most when the atomic fact is short / variable-heavy.
    return claim or ""


def _gather_candidate_docs(
    searcher: BM25Searcher,
    claim: str,
    question_items: Sequence[dict],
    *,
    k_claim: int = 8,
    k_fact: int = 6,
    k_critical: int = 8,
    k_constraint: int = 4,
    max_docs: int = 30,
) -> List[Tuple[str, str, List[str]]]:
    """Role-aware quota gather. Returns ``(title, contents, role_tags)``."""
    pool: Dict[str, Dict] = {}

    def _push(title: str, contents: str, score: float, role: str) -> None:
        slot = pool.setdefault(
            title,
            {"contents": contents, "score": 0.0, "roles": set(), "rank_sum": 0.0},
        )
        slot["score"] = max(slot["score"], score)
        slot["roles"].add(role)

    # ---- claim level ----
    for rank, (title, score, contents) in enumerate(
        searcher.search(claim, k=k_claim)
    ):
        _push(title, contents, score / (1 + 0.05 * rank), "claim")

    # ---- fact level (with role bonuses) ----
    claim_kw = _claim_keywords(claim)
    for qitem in question_items:
        is_critical = bool(qitem.get("critical"))
        k = k_critical if is_critical else k_fact
        fact_only = _fact_search_text(qitem, bindings={})
        # Pair the fact with the claim's keywords so deep-chain facts
        # whose own text is short / variable-heavy still pull topical
        # docs (e-Borders, NOT River Eea).
        query = f"{fact_only} {claim_kw}".strip() if claim_kw else fact_only
        for rank, (title, score, contents) in enumerate(
            searcher.search(query, k=k)
        ):
            bonus = 1.2 if is_critical else 1.0
            _push(title, contents, bonus * score / (1 + 0.05 * rank), "fact")
        # Also issue each ``search_hint`` as a focused query – BM25 is
        # noisier on long concatenated queries (the hints get drowned
        # out by claim tokens). Treating each hint as a separate query
        # surfaces docs that match a specific phrasing.
        hints = qitem.get("search_hints") or []
        for hint in hints[:3]:
            if not isinstance(hint, str) or len(hint.strip()) < 4:
                continue
            for rank, (title, score, contents) in enumerate(
                searcher.search(hint.strip(), k=max(3, k // 2))
            ):
                _push(title, contents, 0.9 * score / (1 + 0.05 * rank),
                      "hint")
        cq = _constraint_query(qitem)
        if cq:
            for rank, (title, score, contents) in enumerate(
                searcher.search(cq, k=k_constraint)
            ):
                _push(title, contents, 0.8 * score / (1 + 0.05 * rank),
                      "constraint")

    ranked = sorted(pool.items(), key=lambda kv: -kv[1]["score"])
    out: List[Tuple[str, str, List[str]]] = []
    for title, slot in ranked[:max_docs]:
        out.append((title, slot["contents"], sorted(slot["roles"])))
    return out


# ---------------------------------------------------------------------------- per-fact retrieval
@dataclass
class FactRetrieval:
    fact_id: str
    fact_text: str
    rely_on: List[str]
    critical: bool
    bindings_applied: Dict[str, str]
    seed_summary: Dict[str, List[str]]
    candidates: List[dict]                 # ordered, contains scores
    direct_supports: List[dict]
    bridge_supports: List[dict]
    notes: List[str] = field(default_factory=list)


def _build_seeds(
    graph: HeteroGraph,
    qitem: dict,
    *,
    encoder: SentenceEncoder,
    parent_results: Dict[str, "FactRetrieval"],
    bindings: Dict[str, str],
    sem_top_k: int = 12,
    entity_weight: float = 1.5,
    semantic_weight: float = 1.0,
    constraint_weight: float = 0.8,
    parent_weight: float = 1.2,
    community_weight: float = 0.4,
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    seeds: Dict[str, float] = {}
    summary: Dict[str, List[str]] = {
        "entity_match": [],
        "semantic": [],
        "constraint": [],
        "parent_carry": [],
        "community": [],
    }

    def _bump(node: str, mass: float, tag: str, label: str = "") -> None:
        if node is None:
            return
        seeds[node] = seeds.get(node, 0.0) + mass
        if label and label not in summary[tag]:
            summary[tag].append(label)

    # 1) entity matches from fact text + constraints + bindings
    surfaces = list(extract_from_fact_text(qitem.get("fact_text", ""),
                                           qitem.get("constraint")))
    surfaces.extend(v for v in bindings.values() if isinstance(v, str))
    for surface in surfaces:
        ent_node = graph.entity_node(surface)
        if ent_node is not None:
            _bump(ent_node, entity_weight, "entity_match", surface)

    # 2) semantic seeds: query embed vs sentence bank
    query = _fact_search_text(qitem, bindings)
    if query and graph.sent_embeddings.shape[0] > 0:
        q_emb = encoder.encode([query])[0]
        sims = graph.sent_embeddings @ q_emb
        k = min(sem_top_k, len(graph.sentences))
        if k > 0:
            top = np.argpartition(-sims, k - 1)[:k]
            for idx in top:
                w = float(sims[int(idx)])
                if w <= 0:
                    continue
                sid = graph.sentences[int(idx)].sid
                node = graph.sentence_node(sid)
                _bump(node, semantic_weight * w, "semantic", sid)

    # 3) constraint signals (boost matching entity nodes already in fact)
    cons = qitem.get("constraint") or {}
    for slot in ("location", "title_role", "exact_name", "unique_modifier"):
        for v in cons.get(slot, []) or []:
            if isinstance(v, str) and not v.startswith("?"):
                ent_node = graph.entity_node(v)
                if ent_node is not None:
                    _bump(ent_node, constraint_weight, "constraint", v)

    # 4) parent carry: every direct-support sentence from upstream facts
    for parent_id in qitem.get("rely_on", []) or []:
        parent = parent_results.get(parent_id)
        if parent is None:
            continue
        for cand in parent.direct_supports[:3]:
            sid = cand["sid"]
            node = graph.sentence_node(sid)
            _bump(node, parent_weight, "parent_carry", sid)
            # also propagate parent's entity nodes
            for surface in cand.get("entities", [])[:5]:
                ent_node = graph.entity_node(surface)
                if ent_node is not None:
                    _bump(ent_node, 0.5 * parent_weight, "parent_carry", surface)

    # 5) community channel: pull a small mass through communities that
    #    contain any of the entity seeds we just placed (provides the
    #    GraphRAG "global" view without dominating).
    seeded_entities = [n for n in seeds if n.startswith("N::")]
    for ent_node in seeded_entities:
        cid = graph.communities.get(ent_node)
        if cid is None:
            continue
        c_node = f"C::c{cid}"
        if c_node in graph.g:
            _bump(c_node, community_weight, "community", f"c{cid}")

    return seeds, summary


def _entity_overlap_score(sent_text: str, surfaces: Sequence[str]) -> float:
    if not surfaces:
        return 0.0
    txt = sent_text.lower()
    hit = sum(1 for s in surfaces if s and s.lower() in txt)
    return hit / max(1, len(surfaces))


def _constraint_match_score(sent_text: str, qitem: dict) -> float:
    cons = qitem.get("constraint") or {}
    needles: List[str] = []
    for slot in ("time", "quantity", "location", "title_role", "exact_name",
                 "unique_modifier"):
        for v in cons.get(slot, []) or []:
            if isinstance(v, str) and not v.startswith("?"):
                needles.append(v.lower())
    if not needles:
        return 0.0
    txt = sent_text.lower()
    hit = sum(1 for n in needles if n in txt)
    return hit / len(needles)


def _direct_support_pass(
    ce_score: float,
    entity_score: float,
    constraint_score: float,
    has_target_entity: bool,
) -> bool:
    """Loose direct-support gate (lessons from plan4.4/4.5 and v1 audit).

    For deep-chain (3/4-hop) facts the target sentence often mentions
    only one of the fact's target entities (the binding resolved via a
    parent), so a hard ``entity_score >= 0.5`` rule excludes too much.
    The new gates lower thresholds and lean on ``has_target_entity``
    or the semantic CE score.
    """
    if ce_score < 0.20:
        return False
    if has_target_entity and ce_score >= 0.30:
        return True
    if entity_score >= 0.34 and ce_score >= 0.28:
        return True
    if constraint_score >= 0.5 and ce_score >= 0.32:
        return True
    if ce_score >= 0.45:
        return True
    return False


def _identify_anchor_titles(
    qitem: dict,
    graph: HeteroGraph,
    target_surfaces: Sequence[str],
) -> set:
    """Find candidate-doc titles that name an entity of the current fact.

    Sentences from anchor docs should dominate the candidate ranking
    so we don't end up picking, say, a "Boone High School" sentence
    for a "Satellite High School" fact.
    """
    anchors: set = set()
    if not graph.title_to_node:
        return anchors
    needles = [s for s in target_surfaces if s and len(s) >= 3]
    if not needles:
        return anchors
    needles_lc = [n.lower() for n in needles]
    for title in graph.title_to_node:
        tl = title.lower()
        for needle in needles_lc:
            # Exact title match or strong substring containment.
            if needle == tl or needle in tl or tl in needle:
                anchors.add(title)
                break
    return anchors


def _retrieve_for_fact(
    qitem: dict,
    graph: HeteroGraph,
    encoder: SentenceEncoder,
    parent_results: Dict[str, FactRetrieval],
    bindings: Dict[str, str],
    *,
    top_k_candidates: int = 25,
    top_k_final: int = 5,
) -> FactRetrieval:
    seeds, summary = _build_seeds(
        graph, qitem, encoder=encoder, parent_results=parent_results,
        bindings=bindings,
    )
    ppr_scores = personalised_pagerank(graph.g, seeds, alpha=0.7)

    target_surfaces = list(
        dict.fromkeys(
            extract_from_fact_text(qitem.get("fact_text", ""),
                                   qitem.get("constraint"))
            + [v for v in bindings.values() if isinstance(v, str)]
        )
    )
    target_keys = {s.lower() for s in target_surfaces}
    anchor_titles = _identify_anchor_titles(qitem, graph, target_surfaces)

    # Score candidates: combine PPR mass with cheap surface checks. We
    # deliberately skip a heavy cross-encoder here – the per-fact graph
    # already pulled in semantically-close sentences and we want this
    # path to be CPU-light enough to run for full HOVER dev (4k claims).
    query = _fact_search_text(qitem, bindings)
    q_emb = encoder.encode([query])[0] if query else None

    # Restrict candidate scoring to sentences that PPR or the semantic
    # top-k actually pulled in, plus all sentences from anchor docs.
    # Skipping the long tail keeps cost roughly linear in the number of
    # interesting sentences rather than the whole local graph.
    candidate_sids: set = set()
    if q_emb is not None and graph.sent_embeddings.shape[0] > 0:
        sims_all = graph.sent_embeddings @ q_emb
        k = min(top_k_candidates, len(graph.sentences))
        for idx in np.argpartition(-sims_all, k - 1)[:k]:
            candidate_sids.add(graph.sentences[int(idx)].sid)
    for node, mass in ppr_scores.items():
        if mass <= 0:
            continue
        if graph.g.nodes[node].get("ntype") == "S":
            sid = graph.g.nodes[node].get("sid")
            if sid:
                candidate_sids.add(sid)
    for sent in graph.sentences:
        if sent.title in anchor_titles:
            candidate_sids.add(sent.sid)

    candidates: List[dict] = []
    for sent in graph.sentences:
        sid = sent.sid
        if sid not in candidate_sids:
            continue
        sent_node = graph.sentence_node(sid)
        ppr_mass = ppr_scores.get(sent_node, 0.0)
        sem_score = 0.0
        if q_emb is not None:
            idx = graph.sid_to_idx.get(sid)
            if idx is not None:
                sem_score = float(graph.sent_embeddings[idx] @ q_emb)
        sentence_surfaces = extract_from_sentence(sent.text)
        ent_score = _entity_overlap_score(sent.text, target_surfaces)
        cons_score = _constraint_match_score(sent.text, qitem)
        has_target = any(s.lower() in target_keys for s in sentence_surfaces)

        # ce_score is a soft surrogate – combination of semantic, surface
        # and PPR signals, normalised into [0, 1].
        ce_score = 0.4 * sem_score + 0.4 * ent_score + 0.2 * cons_score
        ce_score = max(0.0, min(1.0, ce_score))
        is_anchor = sent.title in anchor_titles
        composite = (
            0.35 * sem_score
            + 0.30 * ent_score
            + 0.15 * cons_score
            + 0.20 * math.sqrt(max(ppr_mass, 0.0))
        )
        # Doc anchoring: a sentence from a doc whose title literally
        # names one of the fact's target entities almost always beats
        # a same-topic sentence from an unrelated doc. Large additive
        # boost; non-anchor sentences from a doc that conflicts with
        # one of the anchor's domain get a small penalty so they don't
        # dominate the per-fact ranking.
        if is_anchor:
            composite += 0.35
            ce_score = min(1.0, ce_score + 0.15)
        elif anchor_titles:
            # We have a clear target doc but this sentence isn't from
            # it – suppress unless it directly mentions a target entity.
            if not has_target:
                composite -= 0.20

        direct_pass = _direct_support_pass(
            ce_score=ce_score,
            entity_score=ent_score,
            constraint_score=cons_score,
            has_target_entity=has_target or is_anchor,
        )

        candidates.append({
            "fact_id": qitem.get("fact_id"),
            "sid": sid,
            "title": sent.title,
            "sent_idx": sent.sent_idx,
            "text": sent.text,
            "ppr_score": float(ppr_mass),
            "semantic_score": sem_score,
            "entity_score": ent_score,
            "constraint_score": cons_score,
            "ce_score": ce_score,
            "composite_score": composite,
            "direct_support": direct_pass,
            "has_target_entity": has_target,
            "entities": sentence_surfaces,
        })

    candidates.sort(key=lambda c: -c["composite_score"])
    candidates = candidates[:top_k_candidates]

    direct = [c for c in candidates if c["direct_support"]][:top_k_final]
    bridge = [c for c in candidates if not c["direct_support"]][:top_k_final]

    return FactRetrieval(
        fact_id=qitem.get("fact_id", ""),
        fact_text=qitem.get("fact_text", ""),
        rely_on=list(qitem.get("rely_on", []) or []),
        critical=bool(qitem.get("critical")),
        bindings_applied=dict(bindings),
        seed_summary=summary,
        candidates=candidates,
        direct_supports=direct,
        bridge_supports=bridge,
    )


# ---------------------------------------------------------------------------- top level
@dataclass
class RetrievalConfig:
    # Coarse BM25 quotas. After we dropped the Louvain community pass
    # (graph_build._detect_communities) the per-claim runtime no longer
    # scales super-linearly with the candidate count, so we can afford
    # a wider net. Bigger ``max_docs`` directly attacks the dominant
    # remaining failure mode: 4-hop SUPPORTS misses where the answer-
    # entity doc (Scorpion / Otto von Bismarck / Papa John's Pizza) is
    # not lexically reachable from the claim and only enters the pool
    # through a deeper BM25 sweep.
    k_claim: int = 12
    k_fact: int = 10
    k_critical: int = 14
    k_constraint: int = 6
    max_docs: int = 55
    sem_top_k: int = 6
    sem_threshold: float = 0.50
    fact_top_k_candidates: int = 30
    fact_top_k_final: int = 6
    # Assembly receives these as "base" budgets; ``_dynamic_budgets``
    # may grow them based on fact count / criticality.
    final_max_sentences: int = 12
    final_max_docs: int = 6


def retrieve_for_claim(
    claim: str,
    question_items: Sequence[dict],
    *,
    searcher: BM25Searcher,
    encoder: SentenceEncoder,
    config: RetrievalConfig | None = None,
    initial_bindings: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Run the full GraphRAG retrieval for one claim.

    Returns a dict::

        {
          "candidate_docs": [{"title", "roles"}],
          "graph_stats": {...},
          "fact_results": [FactRetrieval as dict, ...],
          "selected_sentences": [{"sid","title","sent_idx","text", ...}, ...],
          "evidence_text": "...",
          "evidence_per_fact": {fact_id: "..."},
        }
    """
    config = config or RetrievalConfig()
    bindings = dict(initial_bindings or {})

    # 1) BM25 doc gather ----------------------------------------------------
    docs = _gather_candidate_docs(
        searcher=searcher,
        claim=claim,
        question_items=question_items,
        k_claim=config.k_claim,
        k_fact=config.k_fact,
        k_critical=config.k_critical,
        k_constraint=config.k_constraint,
        max_docs=config.max_docs,
    )
    if not docs:
        return {
            "candidate_docs": [],
            "graph_stats": {"num_docs": 0, "num_sentences": 0, "num_entities": 0},
            "fact_results": [],
            "selected_sentences": [],
            "evidence_text": "",
            "evidence_per_fact": {},
        }

    # 2) graph construction -------------------------------------------------
    graph = build_hetero_graph(
        [(t, c) for t, c, _ in docs],
        encoder=encoder,
        sem_top_k=config.sem_top_k,
        sem_threshold=config.sem_threshold,
    )

    # 3) per-fact retrieval in topological order ----------------------------
    fact_results: Dict[str, FactRetrieval] = {}
    ordered = _topological_order(question_items)
    for qitem in ordered:
        result = _retrieve_for_fact(
            qitem=qitem,
            graph=graph,
            encoder=encoder,
            parent_results=fact_results,
            bindings=bindings,
            top_k_candidates=config.fact_top_k_candidates,
            top_k_final=config.fact_top_k_final,
        )
        fact_results[result.fact_id] = result

    # 4) assembly -----------------------------------------------------------
    from .assembly import assemble_evidence

    assembly = assemble_evidence(
        ordered_facts=ordered,
        fact_results=fact_results,
        max_sentences=config.final_max_sentences,
        max_docs=config.final_max_docs,
    )

    return {
        "candidate_docs": [
            {"title": t, "roles": roles} for t, _, roles in docs
        ],
        "graph_stats": {
            "num_docs": len(graph.title_to_node),
            "num_sentences": len(graph.sentences),
            "num_entities": len(graph.entity_keys),
            "num_communities": len(set(graph.communities.values())),
        },
        "fact_results": [_fact_to_dict(r) for r in fact_results.values()],
        "selected_sentences": assembly["selected"],
        "evidence_text": assembly["evidence_text"],
        "evidence_per_fact": assembly["evidence_per_fact"],
        "assembly_summary": assembly["summary"],
    }


def _fact_to_dict(r: FactRetrieval) -> Dict[str, object]:
    return {
        "fact_id": r.fact_id,
        "fact_text": r.fact_text,
        "rely_on": r.rely_on,
        "critical": r.critical,
        "bindings_applied": r.bindings_applied,
        "seed_summary": r.seed_summary,
        "candidates": r.candidates,
        "direct_supports": r.direct_supports,
        "bridge_supports": r.bridge_supports,
        "notes": r.notes,
    }
