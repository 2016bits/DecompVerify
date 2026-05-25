import argparse
import json
import os
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


def process_data_item(data, aggregation_mode):
    verifications = aggregate_labels.normalize_verifications(data)

    if aggregation_mode == "full":
        aggregation_result = aggregate_labels.aggregate_labels(verifications)
    elif aggregation_mode == "majority_vote":
        aggregation_result = aggregate_majority_vote(verifications)
    elif aggregation_mode == "no_critical_gate":
        aggregation_result = aggregate_no_critical_gate(verifications)
    else:
        raise ValueError(f"Unsupported aggregation_mode: {aggregation_mode}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation_mode", type=str, choices=["full", "majority_vote", "no_critical_gate"], default="majority_vote")
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
