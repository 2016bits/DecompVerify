"""Coverage-driven evidence assembly.

Goals (lessons from the v1 run on HOVER_subset):

* **Per-fact isolation.** A 4-hop claim's f5 ("X is the mascot of
  Satellite High School") used to inherit f5-irrelevant sentences from
  pass 3 (a sentence about Boone High School) and the LLM hedged with
  ``insufficient``. We now keep ``evidence_per_fact`` strictly to
  sentences pulled by that fact's own retrieval; cross-fact sentences
  can still join the global pool but never the per-fact slice.

* **Adaptive budgets.** ``max_sentences`` / ``max_docs`` scale with the
  number of atomic facts so deeper claims aren't starved.

* **Rescue pass.** For any fact left empty after the direct + bridge
  passes we lift the best candidate from its own retrieval, even if it
  did not pass the ``direct_support`` gate – an imperfect sentence is
  better than nothing for the per-fact LLM call.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def _sentence_key(cand: dict) -> str:
    return cand["sid"]


def _redundancy(text_a: str, text_b: str) -> float:
    a = set(text_a.lower().split())
    b = set(text_b.lower().split())
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _candidate_gain(
    cand: dict,
    *,
    selected: Sequence[dict],
    selected_docs: set,
    fact_state: Dict[str, dict],
    is_critical: bool,
    bridges_doc_for: Sequence[str],
) -> float:
    base = 0.6 * cand["composite_score"] + 0.4 * cand["ce_score"]
    if not fact_state["covered"]:
        base += 0.8 if cand["direct_support"] else 0.4
    if is_critical and not fact_state["covered"]:
        base += 0.6
    if bridges_doc_for:
        base += 0.25 * len(bridges_doc_for)
    red = max((_redundancy(cand["text"], s["text"]) for s in selected), default=0.0)
    base -= 0.5 * red
    if cand["title"] not in selected_docs:
        base += 0.15
    return base


def _dynamic_budgets(
    ordered_facts: Sequence[dict],
    base_max_sentences: int,
    base_max_docs: int,
) -> tuple[int, int]:
    """Scale budgets with the claim complexity."""
    num_facts = len(ordered_facts)
    num_critical = sum(1 for q in ordered_facts if q.get("critical"))
    # Each critical fact wants 1-2 own sentences plus 1 bridge → 2.
    sent_budget = max(base_max_sentences, 2 * num_critical + num_facts)
    sent_budget = min(sent_budget, 20)
    doc_budget = max(base_max_docs, num_critical)
    doc_budget = min(doc_budget, 10)
    return sent_budget, doc_budget


def assemble_evidence(
    ordered_facts: Sequence[dict],
    fact_results: Dict[str, "FactRetrieval"],
    *,
    max_sentences: int = 8,
    max_docs: int = 4,
    direct_per_critical: int = 2,
    direct_per_normal: int = 1,
    rescue_min_score: float = 0.10,
) -> Dict[str, object]:
    """Greedy assembly with per-fact isolation."""

    fact_state = {
        fid: {"covered": False, "direct_count": 0, "needs_closure": False}
        for fid in (q.get("fact_id") for q in ordered_facts)
    }
    rely_lookup = {q.get("fact_id"): list(q.get("rely_on", []) or [])
                   for q in ordered_facts}
    critical_lookup = {q.get("fact_id"): bool(q.get("critical"))
                       for q in ordered_facts}

    max_sentences, max_docs = _dynamic_budgets(
        ordered_facts, max_sentences, max_docs
    )

    selected: List[dict] = []
    selected_keys: set = set()
    selected_docs: set = set()
    # ``per_fact`` is the "focused" slice each fact owns; pass 3 global
    # picks DO NOT get appended here even if scored under a fact_id.
    per_fact: Dict[str, List[dict]] = {fid: [] for fid in fact_state}

    def _add(cand: dict, fid: str | None, *, into_per_fact: bool) -> None:
        key = _sentence_key(cand)
        if key in selected_keys:
            return
        selected.append(cand)
        selected_keys.add(key)
        selected_docs.add(cand["title"])
        if fid is not None and into_per_fact:
            per_fact[fid].append(cand)
            fact_state[fid]["direct_count"] += 1
            fact_state[fid]["covered"] = True

    # ------------------------------------------------------------- pass 1: direct
    for qitem in ordered_facts:
        fid = qitem.get("fact_id")
        result = fact_results.get(fid)
        if result is None:
            continue
        target = direct_per_critical if critical_lookup[fid] else direct_per_normal
        for cand in result.direct_supports:
            if len(selected) >= max_sentences:
                break
            if (
                cand["title"] not in selected_docs
                and len(selected_docs) >= max_docs
                and fact_state[fid]["covered"]
            ):
                continue
            _add(cand, fid, into_per_fact=True)
            if fact_state[fid]["direct_count"] >= target:
                break

    # --------------------------------------------- pass 1b: rescue empty facts
    # For any fact that still has nothing of its own, lift the best
    # candidate from its retrieval. We allow scores below the direct
    # gate – this is the "evidence_per_fact must not be empty" rule.
    for qitem in ordered_facts:
        fid = qitem.get("fact_id")
        if per_fact[fid]:
            continue
        result = fact_results.get(fid)
        if result is None or not result.candidates:
            continue
        for cand in result.candidates:
            if cand["composite_score"] < rescue_min_score:
                break
            if len(selected) >= max_sentences:
                break
            if _sentence_key(cand) in selected_keys:
                # Already in the global pool from another fact – piggyback
                # for per-fact ownership without spending budget.
                per_fact[fid].append(cand)
                fact_state[fid]["covered"] = True
                break
            _add(cand, fid, into_per_fact=True)
            break

    # ----------------------------------------------------------- pass 2: bridge / dep closure
    for qitem in ordered_facts:
        fid = qitem.get("fact_id")
        result = fact_results.get(fid)
        if result is None:
            continue
        deps = rely_lookup.get(fid, [])
        if not deps:
            continue
        parent_docs: set = set()
        for parent_id in deps:
            for s in per_fact.get(parent_id, []):
                parent_docs.add(s["title"])
        my_docs = {s["title"] for s in per_fact[fid]}
        if parent_docs & my_docs:
            continue
        for cand in result.bridge_supports + result.direct_supports:
            if len(selected) >= max_sentences:
                break
            if _sentence_key(cand) in selected_keys:
                continue
            _add(cand, fid, into_per_fact=True)
            break

    # --------------------------------------- pass 3: global greedy gain fill
    # Pass 3 *only* expands the global pool – it does NOT add sentences
    # to ``per_fact`` so the LLM per-fact call stays focused.
    pool: List[dict] = []
    for fid, result in fact_results.items():
        for cand in result.candidates[:10]:
            cand2 = dict(cand)
            cand2["_fact_id"] = fid
            pool.append(cand2)
    pool.sort(key=lambda c: -c["composite_score"])

    while len(selected) < max_sentences:
        best, best_score = None, -1e9
        for cand in pool:
            key = _sentence_key(cand)
            if key in selected_keys:
                continue
            fid = cand["_fact_id"]
            state = fact_state[fid]
            bridges_doc_for = [
                pfid for pfid in rely_lookup.get(fid, [])
                if any(s["title"] == cand["title"] for s in per_fact.get(pfid, []))
            ]
            gain = _candidate_gain(
                cand,
                selected=selected,
                selected_docs=selected_docs,
                fact_state=state,
                is_critical=critical_lookup.get(fid, False),
                bridges_doc_for=bridges_doc_for,
            )
            if gain > best_score:
                best_score = gain
                best = cand
        if best is None or best_score <= 0.05:
            break
        fid = best.pop("_fact_id")
        _add(best, fid=None, into_per_fact=False)
        fact_state[fid]["covered"] = True

    # ----------------------------------------- emit evidence_text + per_fact
    grouped: Dict[str, List[dict]] = {}
    for cand in selected:
        grouped.setdefault(cand["title"], []).append(cand)
    for items in grouped.values():
        items.sort(key=lambda c: c["sent_idx"])

    evidence_chunks: List[str] = []
    for title in dict.fromkeys(cand["title"] for cand in selected):
        chunk = " ".join(item["text"].strip() for item in grouped[title])
        evidence_chunks.append(chunk)
    evidence_text = "\n".join(evidence_chunks)

    evidence_per_fact: Dict[str, str] = {}
    for fid, items in per_fact.items():
        if not items:
            evidence_per_fact[fid] = ""
            continue
        items_sorted = sorted(items, key=lambda c: (c["title"], c["sent_idx"]))
        evidence_per_fact[fid] = " ".join(c["text"].strip() for c in items_sorted)

    summary = {
        "num_selected": len(selected),
        "num_docs": len(selected_docs),
        "max_sentences_budget": max_sentences,
        "max_docs_budget": max_docs,
        "fact_coverage": {fid: fact_state[fid]["covered"] for fid in fact_state},
        "per_fact_sentence_counts": {
            fid: len(items) for fid, items in per_fact.items()
        },
    }

    return {
        "selected": selected,
        "evidence_text": evidence_text,
        "evidence_per_fact": evidence_per_fact,
        "summary": summary,
    }
