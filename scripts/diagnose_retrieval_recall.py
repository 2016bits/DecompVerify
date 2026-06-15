"""Diagnose retrieval recall against HOVER gold supporting_facts.

Reads a ``*_retrieved_*.json`` artefact produced by ``retrieve_evidence.py``
and the raw HOVER dev release, and computes recall at three layers:

1. ``recall@candidate_docs``     – fraction of gold (title) covered by the
   role-aware BM25 doc pool (``retrieval_meta.candidate_docs``).
2. ``recall@per_fact_top5``      – fraction of gold (title) covered by the
   union of every fact's saved top-5 candidates
   (``retrieval_meta.fact_breakdown[*].top_candidates``). NOTE: the upstream
   pipeline only persists top-5 per fact even though it scores ~25-30, so
   this is a *lower bound* on the actual per-fact pool recall.
3. ``recall@selected`` (doc / sentence) – the final evidence assembled into
   ``retrieval_meta.selected_sentences`` (what the per-fact LLM sees).

For each claim we also classify the failure mode::

    pool_miss     gold title never entered candidate_docs        (= recall bottleneck)
    ranking_miss  in candidate_docs but never made selected      (= ranking / assembly bottleneck)
    full_hit      every gold (title, sent_idx) made selected

Aggregations are reported overall and by ``num_hops``.

Example::

    python scripts/diagnose_retrieval_recall.py \\
        --retrieved data/HOVER_subset/bit_plan7.0_graphrag/dev_2_retrieved_0_200.llm_D_E.json \\
        --raw_hover /mnt/data/yangjun/data/HOVER/data/raw/hover_dev_release_v1.1.json \\
        --report_out reports/retrieval_recall_llm_D_E.md \\
        --per_claim_out data/HOVER_subset/bit_plan7.0_graphrag/recall_diag_llm_D_E.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


GoldKey = Tuple[str, int]   # (title, sent_idx)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _gold_index(raw_path: str) -> Dict[str, Dict]:
    """uid -> {gold_pairs, gold_titles, label, num_hops}."""
    raw = _load_json(raw_path)
    idx: Dict[str, Dict] = {}
    for d in raw:
        pairs: Set[GoldKey] = set()
        titles: Set[str] = set()
        for entry in d.get("supporting_facts", []) or []:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            title, sent_idx = entry[0], entry[1]
            if not isinstance(title, str) or not isinstance(sent_idx, int):
                continue
            pairs.add((title, sent_idx))
            titles.add(title)
        idx[d["uid"]] = {
            "gold_pairs": pairs,
            "gold_titles": titles,
            "label": d.get("label"),
            "num_hops": d.get("num_hops"),
        }
    return idx


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _classify(
    gold_pairs: Set[GoldKey],
    gold_titles: Set[str],
    pool_titles: Set[str],
    selected_pairs: Set[GoldKey],
    selected_titles: Set[str],
) -> str:
    if not gold_titles:
        return "no_gold"
    if not (gold_titles - pool_titles) and gold_pairs.issubset(selected_pairs):
        return "full_hit"
    missing_from_pool = gold_titles - pool_titles
    if missing_from_pool:
        return "pool_miss"
    # All gold titles were in the pool but at least one (title, sent_idx) didn't
    # make selected — that's a ranking / assembly issue.
    return "ranking_miss"


def _per_claim_metrics(item: dict, gold: dict) -> Optional[dict]:
    if gold is None or not gold["gold_titles"]:
        return None
    meta = item.get("retrieval_meta") or {}

    # Pool (doc level).
    pool_titles: Set[str] = {
        d.get("title") for d in meta.get("candidate_docs", [])
        if isinstance(d, dict) and isinstance(d.get("title"), str)
    }

    # Per-fact saved top-5 (doc level union).
    per_fact_top5_titles: Set[str] = set()
    for fb in meta.get("fact_breakdown", []) or []:
        for c in fb.get("top_candidates", []) or []:
            t = c.get("title")
            if isinstance(t, str):
                per_fact_top5_titles.add(t)

    # Selected (doc + sentence level).
    selected_titles: Set[str] = set()
    selected_pairs: Set[GoldKey] = set()
    for s in meta.get("selected_sentences", []) or []:
        title, sidx = s.get("title"), s.get("sent_idx")
        if isinstance(title, str):
            selected_titles.add(title)
            if isinstance(sidx, int):
                selected_pairs.add((title, sidx))

    gold_titles = gold["gold_titles"]
    gold_pairs = gold["gold_pairs"]

    pool_doc_recall = _safe_div(
        len(gold_titles & pool_titles), len(gold_titles))
    perfact_top5_doc_recall = _safe_div(
        len(gold_titles & per_fact_top5_titles), len(gold_titles))
    sel_doc_recall = _safe_div(
        len(gold_titles & selected_titles), len(gold_titles))
    sel_sent_recall = _safe_div(
        len(gold_pairs & selected_pairs), len(gold_pairs))

    bucket = _classify(
        gold_pairs, gold_titles, pool_titles,
        selected_pairs, selected_titles,
    )

    missing_titles = sorted(gold_titles - pool_titles)
    missing_pairs = sorted(gold_pairs - selected_pairs)

    return {
        "id": item.get("id"),
        "num_hops": gold["num_hops"] or item.get("num_hops"),
        "label": gold["label"],
        "num_gold_titles": len(gold_titles),
        "num_gold_pairs": len(gold_pairs),
        "num_candidate_docs": len(pool_titles),
        "num_selected_sentences": len(selected_pairs),
        "pool_doc_recall": pool_doc_recall,
        "perfact_top5_doc_recall": perfact_top5_doc_recall,
        "selected_doc_recall": sel_doc_recall,
        "selected_sent_recall": sel_sent_recall,
        "bucket": bucket,
        "missing_titles": missing_titles,
        "missing_pairs": [list(p) for p in missing_pairs],
    }


def _summarise(per_claim: List[dict]) -> Dict[str, dict]:
    by_hop: Dict[str, List[dict]] = defaultdict(list)
    for r in per_claim:
        by_hop[str(r["num_hops"])].append(r)
        by_hop["overall"].append(r)

    summary: Dict[str, dict] = {}
    for hop, rows in by_hop.items():
        n = len(rows)
        if not n:
            continue
        def _avg(field: str) -> float:
            return statistics.fmean(r[field] for r in rows)

        bucket_counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            bucket_counts[r["bucket"]] += 1

        # "Perfect" claims — every gold pair landed in selected.
        perfect_pairs = sum(1 for r in rows if r["selected_sent_recall"] == 1.0)
        zero_pool = sum(1 for r in rows if r["pool_doc_recall"] == 0.0)
        zero_sel = sum(1 for r in rows if r["selected_doc_recall"] == 0.0)

        summary[hop] = {
            "n": n,
            "pool_doc_recall_mean": _avg("pool_doc_recall"),
            "perfact_top5_doc_recall_mean": _avg("perfact_top5_doc_recall"),
            "selected_doc_recall_mean": _avg("selected_doc_recall"),
            "selected_sent_recall_mean": _avg("selected_sent_recall"),
            "buckets": dict(bucket_counts),
            "frac_full_hit": _safe_div(bucket_counts.get("full_hit", 0), n),
            "frac_pool_miss": _safe_div(bucket_counts.get("pool_miss", 0), n),
            "frac_ranking_miss": _safe_div(
                bucket_counts.get("ranking_miss", 0), n),
            "frac_zero_pool_recall": _safe_div(zero_pool, n),
            "frac_zero_selected_recall": _safe_div(zero_sel, n),
            "frac_perfect_sent_recall": _safe_div(perfect_pairs, n),
        }
    return summary


# ---------------------------------------------------------------- reporting

_HOP_ORDER = ["2", "3", "4", "overall"]


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def _build_markdown(summary: Dict[str, dict], meta: dict) -> str:
    lines: List[str] = []
    lines.append(f"# Retrieval recall diagnosis")
    lines.append("")
    lines.append(f"- retrieved: `{meta['retrieved_path']}`")
    lines.append(f"- raw HOVER: `{meta['raw_path']}`")
    lines.append(f"- claims analysed: {meta['n_analysed']} "
                 f"(skipped {meta['n_skipped']} without gold)")
    lines.append("")
    lines.append("## 1. Recall by hop")
    lines.append("")
    lines.append("| hop | N | pool@docs | perFact_top5@docs | selected@docs | selected@sents |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for hop in _HOP_ORDER:
        s = summary.get(hop)
        if not s:
            continue
        lines.append(
            f"| {hop} | {s['n']} "
            f"| {_fmt_pct(s['pool_doc_recall_mean'])} "
            f"| {_fmt_pct(s['perfact_top5_doc_recall_mean'])} "
            f"| {_fmt_pct(s['selected_doc_recall_mean'])} "
            f"| {_fmt_pct(s['selected_sent_recall_mean'])} |"
        )
    lines.append("")
    lines.append("**Reading the columns**")
    lines.append("")
    lines.append("- `pool@docs`: gold titles present in BM25 candidate_docs pool. "
                 "If this drops on deep hops, the BM25 stage itself is leaking — "
                 "no downstream ranker can recover.")
    lines.append("- `perFact_top5@docs`: gold titles covered by the union of "
                 "every fact's top-5 candidates. Lower bound — the upstream "
                 "pipeline only persists top-5, but the actual ranker considers "
                 "the full ~25-30 candidate set per fact.")
    lines.append("- `selected@docs` / `selected@sents`: doc- and sentence-level "
                 "recall of what the per-fact LLM actually sees.")
    lines.append("")

    lines.append("## 2. Failure-mode buckets")
    lines.append("")
    lines.append("| hop | full_hit | pool_miss | ranking_miss | zero pool | zero selected |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for hop in _HOP_ORDER:
        s = summary.get(hop)
        if not s:
            continue
        lines.append(
            f"| {hop} "
            f"| {_fmt_pct(s['frac_full_hit'])} "
            f"| {_fmt_pct(s['frac_pool_miss'])} "
            f"| {_fmt_pct(s['frac_ranking_miss'])} "
            f"| {_fmt_pct(s['frac_zero_pool_recall'])} "
            f"| {_fmt_pct(s['frac_zero_selected_recall'])} |"
        )
    lines.append("")
    lines.append("- `pool_miss`: at least one gold title never entered the "
                 "candidate pool → recall bottleneck (BM25 fix needed).")
    lines.append("- `ranking_miss`: every gold title was in the pool, but at "
                 "least one (title, sent_idx) didn't make selected → ranking / "
                 "assembly bottleneck.")
    lines.append("- `zero pool` / `zero selected`: catastrophic cases where the "
                 "pool / selected set has zero overlap with gold.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- main

def main(args):
    retrieved = _load_json(args.retrieved)
    if not isinstance(retrieved, list):
        raise SystemExit(f"expected list at top of {args.retrieved}")
    gold_idx = _gold_index(args.raw_hover)

    per_claim: List[dict] = []
    skipped = 0
    for item in retrieved:
        uid = item.get("id")
        gold = gold_idx.get(uid)
        if gold is None or not gold["gold_titles"]:
            skipped += 1
            continue
        row = _per_claim_metrics(item, gold)
        if row is None:
            skipped += 1
            continue
        per_claim.append(row)

    summary = _summarise(per_claim)

    meta = {
        "retrieved_path": args.retrieved,
        "raw_path": args.raw_hover,
        "n_analysed": len(per_claim),
        "n_skipped": skipped,
    }

    print(f"[diagnose] analysed {len(per_claim)} claims, skipped {skipped} "
          f"without gold or matching id.")
    print()
    for hop in _HOP_ORDER:
        s = summary.get(hop)
        if not s:
            continue
        print(
            f"hop={hop:7s} N={s['n']:4d}  "
            f"pool={_fmt_pct(s['pool_doc_recall_mean'])}  "
            f"perFact5={_fmt_pct(s['perfact_top5_doc_recall_mean'])}  "
            f"sel_doc={_fmt_pct(s['selected_doc_recall_mean'])}  "
            f"sel_sent={_fmt_pct(s['selected_sent_recall_mean'])}  "
            f"buckets={dict(s['buckets'])}"
        )

    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as fh:
            fh.write(_build_markdown(summary, meta))
        print(f"\n[diagnose] wrote markdown report -> {args.report_out}")

    if args.per_claim_out:
        os.makedirs(os.path.dirname(args.per_claim_out) or ".", exist_ok=True)
        payload = {"meta": meta, "summary": summary, "per_claim": per_claim}
        with open(args.per_claim_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"[diagnose] wrote per-claim details -> {args.per_claim_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieved", type=str, required=True,
        help="Path to a *_retrieved_*.json produced by retrieve_evidence.py.")
    parser.add_argument(
        "--raw_hover", type=str,
        default="/mnt/data/yangjun/data/HOVER/data/raw/hover_dev_release_v1.1.json",
        help="Raw HOVER release with supporting_facts.")
    parser.add_argument(
        "--report_out", type=str, default="",
        help="Optional markdown report path.")
    parser.add_argument(
        "--per_claim_out", type=str, default="",
        help="Optional per-claim JSON dump (full failure-mode trace).")
    main(parser.parse_args())
