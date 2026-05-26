import argparse
import copy
import hashlib
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial

from tqdm import tqdm

import aggregate_labels


SUPPORTED = aggregate_labels.SUPPORTED
CONTRADICTED = aggregate_labels.CONTRADICTED
INSUFFICIENT = aggregate_labels.INSUFFICIENT


def label_counts(verifications):
    counts = Counter(aggregate_labels.get_effective_label(item) for item in verifications)
    return {
        "support": counts.get(SUPPORTED, 0),
        "contradict": counts.get(CONTRADICTED, 0),
        "insufficient": counts.get(INSUFFICIENT, 0),
    }


def simple_fact_records(verifications):
    records = []
    for item in verifications:
        records.append({
            "fact_id": item.get("fact_id", ""),
            "critical": item.get("critical", False),
            "rely_on": item.get("rely_on", []),
            "verification_label": item.get("verification_label", INSUFFICIENT),
            "adjudicated_label": item.get("adjudicated_label", item.get("verification_label", INSUFFICIENT)),
            "effective_label": aggregate_labels.get_effective_label(item),
        })
    return records


def to_claim_label(fact_label):
    if fact_label == SUPPORTED:
        return "supports", "supports"
    if fact_label == CONTRADICTED:
        return "refutes", "refutes"
    return "not enough information", "refutes"


# ---------------------------------------------------------------------------
# Existing baselines
# ---------------------------------------------------------------------------


def aggregate_majority_vote(verifications):
    counts = label_counts(verifications)
    if not verifications:
        final_label, final_label_binary = to_claim_label(INSUFFICIENT)
        return {
            "final_label": final_label,
            "final_label_binary": final_label_binary,
            "decision_reason": "Ablation: no atomic facts were available, so the claim is treated as not enough information.",
            "aggregation_mode": "majority_vote",
            "counts": counts,
            "selected_fact_label": INSUFFICIENT,
            "tie_break_order": ["contradict", "insufficient", "support"],
            "per_fact_records": [],
        }

    tie_break_order = [CONTRADICTED, INSUFFICIENT, SUPPORTED]
    fact_label_names = {
        SUPPORTED: "support",
        CONTRADICTED: "contradict",
        INSUFFICIENT: "insufficient",
    }
    max_count = max(counts.values()) if counts else 0
    selected_fact_label = INSUFFICIENT
    for label in tie_break_order:
        if counts[fact_label_names[label]] == max_count:
            selected_fact_label = label
            break

    final_label, final_label_binary = to_claim_label(selected_fact_label)
    return {
        "final_label": final_label,
        "final_label_binary": final_label_binary,
        "decision_reason": (
            "Ablation: all atomic facts vote equally; ties are resolved as "
            "contradict > insufficient > support."
        ),
        "aggregation_mode": "majority_vote",
        "counts": counts,
        "selected_fact_label": selected_fact_label,
        "tie_break_order": ["contradict", "insufficient", "support"],
        "per_fact_records": simple_fact_records(verifications),
    }


def aggregate_no_critical_gate(verifications):
    counts = label_counts(verifications)
    if counts["contradict"] > 0:
        final_label = "refutes"
        final_label_binary = "refutes"
        reason = "Ablation: at least one fact is contradicted; critical markers are ignored."
    elif counts["insufficient"] > 0:
        final_label = "not enough information"
        final_label_binary = "refutes"
        reason = "Ablation: no fact is contradicted, but at least one fact is insufficient; critical markers are ignored."
    else:
        final_label = "supports"
        final_label_binary = "supports"
        reason = "Ablation: every fact is supported; critical markers are ignored."

    critical_counts = Counter()
    for item in verifications:
        if item.get("critical", False):
            critical_counts[aggregate_labels.get_effective_label(item)] += 1

    return {
        "final_label": final_label,
        "final_label_binary": final_label_binary,
        "decision_reason": reason,
        "aggregation_mode": "no_critical_gate",
        "counts": {
            **counts,
            "critical_support_diagnostic_only": critical_counts.get(SUPPORTED, 0),
            "critical_contradict_diagnostic_only": critical_counts.get(CONTRADICTED, 0),
            "critical_insufficient_diagnostic_only": critical_counts.get(INSUFFICIENT, 0),
        },
        "per_fact_records": simple_fact_records(verifications),
    }


# ---------------------------------------------------------------------------
# Group 1: critical-marker semantics (LLM-free aggregate-only ablations)
# ---------------------------------------------------------------------------


def _claim_seed(data):
    base = str(data.get("id") or data.get("claim") or "")
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _critical_rewrite_random(verifications, seed):
    n = len(verifications)
    target_n = sum(1 for v in verifications if v.get("critical", False))
    if n == 0:
        return [], 0, 0
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    chosen = set(indices[:target_n])
    new_verifs = []
    flipped = 0
    for idx, v in enumerate(verifications):
        new = dict(v)
        new_crit = idx in chosen
        if new_crit != v.get("critical", False):
            flipped += 1
        new["critical"] = new_crit
        new_verifs.append(new)
    return new_verifs, target_n, flipped


def _critical_rewrite_flip(verifications):
    new_verifs = []
    flipped = 0
    for v in verifications:
        new = dict(v)
        new["critical"] = not bool(v.get("critical", False))
        flipped += 1
        new_verifs.append(new)
    return new_verifs, sum(1 for x in new_verifs if x["critical"]), flipped


def _critical_rewrite_structural(verifications):
    """Define critical structurally: a fact is critical iff it is not relied
    on by any other fact (i.e. it sits at the top of the rely_on chain and is
    therefore directly referenced by the claim aggregation). All upstream
    facts in the chain are demoted to non-critical."""
    relied_on_by = set()
    for v in verifications:
        for ref in v.get("rely_on", []) or []:
            relied_on_by.add(ref)
    new_verifs = []
    flipped = 0
    for v in verifications:
        new = dict(v)
        new_crit = v.get("fact_id") not in relied_on_by
        if new_crit != v.get("critical", False):
            flipped += 1
        new["critical"] = new_crit
        new_verifs.append(new)
    return new_verifs, sum(1 for x in new_verifs if x["critical"]), flipped


def aggregate_with_rewritten_critical(verifications, mode, data):
    if mode == "random_critical":
        new_verifs, n_crit, flipped = _critical_rewrite_random(verifications, _claim_seed(data))
        reason_tag = "Ablation: critical markers were randomized while keeping the original critical count."
    elif mode == "flip_critical":
        new_verifs, n_crit, flipped = _critical_rewrite_flip(verifications)
        reason_tag = "Ablation: critical and non-critical markers were swapped."
    elif mode == "structural_critical":
        new_verifs, n_crit, flipped = _critical_rewrite_structural(verifications)
        reason_tag = "Ablation: critical markers were recomputed structurally (terminal facts only)."
    else:
        raise ValueError(f"Unknown critical-rewrite mode: {mode}")

    aggregation_result = aggregate_labels.aggregate_labels(new_verifs)
    aggregation_result["aggregation_mode"] = mode
    aggregation_result["critical_rewrite"] = {
        "mode": mode,
        "original_critical_count": sum(1 for v in verifications if v.get("critical")),
        "rewritten_critical_count": n_crit,
        "labels_flipped": flipped,
        "note": reason_tag,
    }
    base_reason = aggregation_result.get("decision_reason", "")
    aggregation_result["decision_reason"] = f"[{mode}] " + base_reason if base_reason else reason_tag
    return aggregation_result


# ---------------------------------------------------------------------------
# Group 3A: dependency removed at aggregate stage only
# ---------------------------------------------------------------------------


def aggregate_wo_rely_on(verifications):
    rewritten = []
    removed_edges = 0
    for v in verifications:
        new = dict(v)
        removed_edges += len(v.get("rely_on", []) or [])
        new["rely_on"] = []
        rewritten.append(new)
    aggregation_result = aggregate_labels.aggregate_labels(rewritten)
    aggregation_result["aggregation_mode"] = "wo_rely_on_aggregate_only"
    aggregation_result["rely_on_removed"] = {
        "removed_dependency_edges": removed_edges,
        "note": (
            "Ablation: rely_on edges are erased before aggregation. Role weight no longer "
            "distinguishes dependent vs root non-critical facts; the rest of the pipeline "
            "(decomposition, question generation, answering, verification) is unchanged."
        ),
    }
    return aggregation_result


# ---------------------------------------------------------------------------
# Group 4B: decomp + verify but with no aggregation heuristics (plain AND)
# ---------------------------------------------------------------------------


def aggregate_decomp_no_aggregation(verifications):
    """A clean alternative to majority_vote: keep decomposition and per-fact
    verification, but discard every aggregation heuristic. The claim is
    SUPPORTS iff every fact's effective label is SUPPORTED; otherwise REFUTES.
    No critical gate, no NEI tolerance, no score, no tie-breaking.
    """
    counts = label_counts(verifications)
    if not verifications:
        final_label, final_label_binary = "refutes", "refutes"
        reason = "Ablation: no atomic facts; claim defaults to refutes under plain AND."
    elif counts["support"] == len(verifications):
        final_label = "supports"
        final_label_binary = "supports"
        reason = "Ablation: plain AND aggregation -- every fact is supported."
    else:
        final_label = "refutes"
        final_label_binary = "refutes"
        reason = (
            "Ablation: plain AND aggregation -- at least one fact is not supported. "
            "No critical gate, no NEI tolerance, no scoring heuristic is applied."
        )
    return {
        "final_label": final_label,
        "final_label_binary": final_label_binary,
        "decision_reason": reason,
        "aggregation_mode": "decomp_no_aggregation",
        "counts": counts,
        "per_fact_records": simple_fact_records(verifications),
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


CRITICAL_REWRITE_MODES = {"random_critical", "flip_critical", "structural_critical"}


def run_aggregation(verifications, mode, data):
    if mode == "full":
        return aggregate_labels.aggregate_labels(verifications)
    if mode == "majority_vote":
        return aggregate_majority_vote(verifications)
    if mode == "no_critical_gate":
        return aggregate_no_critical_gate(verifications)
    if mode in CRITICAL_REWRITE_MODES:
        return aggregate_with_rewritten_critical(verifications, mode, data)
    if mode == "wo_rely_on_aggregate_only":
        return aggregate_wo_rely_on(verifications)
    if mode == "decomp_no_aggregation":
        return aggregate_decomp_no_aggregation(verifications)
    raise ValueError(f"Unsupported aggregation_mode: {mode}")


def process_data_item(data, aggregation_mode):
    verifications = aggregate_labels.normalize_verifications(data)
    aggregation_result = run_aggregation(verifications, aggregation_mode, data)
    out = dict(data)
    out["aggregation_result"] = aggregation_result
    return out


def resolve_path(path, args):
    return (
        path
        .replace("[DATA]", args.dataset)
        .replace("[PLAN]", args.plan)
        .replace("[TYPE]", args.data_type)
        .replace("[CLASS]", args.class_num)
        .replace("[T]", args.t)
        .replace("[S]", str(args.start))
        .replace("[E]", str(args.end))
    )


def main(args):
    in_path = resolve_path(args.in_path, args)

    with open(in_path, "r", encoding="utf-8") as file_obj:
        raws = json.load(file_obj)
    raws = raws[args.start:args.end]

    results = []
    partial_func = partial(process_data_item, aggregation_mode=args.aggregation_mode)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(partial_func, data) for data in raws]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Error in future: {exc}")

    results = sorted(results, key=lambda item: item["id"])

    out_path = resolve_path(args.out_path, args)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=4, ensure_ascii=False)

    print(f"Saved to {out_path}")
    print("Program finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


SUPPORTED_MODES = [
    "full",
    "majority_vote",
    "no_critical_gate",
    "random_critical",
    "flip_critical",
    "structural_critical",
    "wo_rely_on_aggregate_only",
    "decomp_no_aggregation",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation_mode", type=str, choices=SUPPORTED_MODES, default="majority_vote")
    parser.add_argument("--dataset", type=str, default="HOVER")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--class_num", type=str, default="2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=4000)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--plan", type=str, default="qc_plan6.4")
    parser.add_argument("--t", type=str, default="")
    parser.add_argument("--in_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_verify_[T][S]_[E].json")
    parser.add_argument("--out_path", type=str, default="./data/[DATA]/[PLAN]/ablations/[TYPE]_[CLASS]_majority_vote_aggregate_[T][S]_[E].json")
    main(parser.parse_args())
