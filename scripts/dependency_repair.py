"""Dependency repair utilities for multi-hop question plans.

The decomposition LLM occasionally leaves ``rely_on`` empty for facts that
clearly consume a variable produced by another fact. Retrieval relies on this
structure to decide when to propagate bindings and expand the document pool, so
we repair the graph deterministically before GraphRAG runs.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")


def _clean_text(text) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _vars_in_obj(obj) -> set:
    if isinstance(obj, str):
        return set(VAR_PATTERN.findall(obj))
    if isinstance(obj, list):
        out = set()
        for item in obj:
            out.update(_vars_in_obj(item))
        return out
    if isinstance(obj, dict):
        out = set()
        for value in obj.values():
            out.update(_vars_in_obj(value))
        return out
    return set()


def _qitem_consumed_vars(qitem: dict) -> set:
    """Return variables used by a question item, excluding its own output slot."""
    vars_seen = set()
    for key in ("fact_text", "main_question", "search_hints", "constraint"):
        vars_seen.update(_vars_in_obj(qitem.get(key)))
    slot = _clean_text(qitem.get("answer_slot", ""))
    if slot:
        vars_seen.discard(slot)
    return vars_seen


def _would_create_cycle(
    child_id: str,
    parent_id: str,
    rely_map: Dict[str, List[str]],
) -> bool:
    """True if adding ``child -> parent`` would make a cycle."""
    stack = [parent_id]
    seen = set()
    while stack:
        fid = stack.pop()
        if fid == child_id:
            return True
        if fid in seen:
            continue
        seen.add(fid)
        stack.extend(rely_map.get(fid, []) or [])
    return False


def repair_question_dependencies(question_items: List[dict]) -> Tuple[List[dict], dict]:
    """Add missing ``rely_on`` edges from variable producers to consumers.

    A producer is the fact whose ``answer_slot`` is a variable. A consumer is a
    different fact whose text/question/hints/constraints mention that variable.
    """
    items = [dict(q) for q in question_items or []]
    by_id = {q.get("fact_id"): q for q in items if q.get("fact_id")}
    producers: Dict[str, str] = {}
    for q in items:
        fid = q.get("fact_id")
        slot = _clean_text(q.get("answer_slot", ""))
        if fid and slot.startswith("?"):
            producers.setdefault(slot, fid)

    rely_map: Dict[str, List[str]] = {
        fid: list(q.get("rely_on", []) or []) for fid, q in by_id.items()
    }
    added_edges = []
    consumer_vars: Dict[str, List[str]] = {}

    for q in items:
        fid = q.get("fact_id")
        if not fid:
            continue
        consumed = sorted(_qitem_consumed_vars(q))
        consumer_vars[fid] = consumed
        for var in consumed:
            parent_id = producers.get(var)
            if not parent_id or parent_id == fid:
                continue
            deps = rely_map.setdefault(fid, [])
            if parent_id in deps:
                continue
            if _would_create_cycle(fid, parent_id, rely_map):
                continue
            deps.append(parent_id)
            added_edges.append({"fact_id": fid, "depends_on": parent_id, "var": var})

    repaired = []
    for q in items:
        q2 = dict(q)
        fid = q2.get("fact_id")
        if fid in rely_map:
            q2["rely_on"] = rely_map[fid]
        repaired.append(q2)

    summary = {
        "enabled": True,
        "num_added_edges": len(added_edges),
        "added_edges": added_edges,
        "producers": producers,
        "consumer_vars": consumer_vars,
    }
    return repaired, summary


def repair_item_dependencies(item: dict) -> Tuple[dict, dict]:
    """Repair both ``question_plan`` and ``decomposition.atomic_facts`` in an item."""
    out = dict(item)
    qplan = dict(out.get("question_plan") or {})
    qitems, summary = repair_question_dependencies(qplan.get("question_items", []) or [])
    qplan["question_items"] = qitems
    qplan["dependency_repair"] = summary
    out["question_plan"] = qplan

    rely_by_fact = {
        q.get("fact_id"): list(q.get("rely_on", []) or [])
        for q in qitems
        if q.get("fact_id")
    }
    decomp = dict(out.get("decomposition") or {})
    facts = []
    for fact in decomp.get("atomic_facts", []) or []:
        f2 = dict(fact)
        fid = f2.get("id")
        if fid in rely_by_fact:
            f2["rely_on"] = rely_by_fact[fid]
        facts.append(f2)
    if facts:
        decomp["atomic_facts"] = facts
        decomp["dependency_repair"] = summary
        out["decomposition"] = decomp

    out["dependency_repair"] = summary
    return out, summary
