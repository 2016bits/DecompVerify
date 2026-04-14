import os
import math
import json
import argparse
from datetime import datetime
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

SUPPORTED = "support"
CONTRADICTED = "contradict"
INSUFFICIENT = "insufficient"



def _clean_text(text):
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()



def get_effective_label(item):
    label = _clean_text(item.get("adjudicated_label", "")) or _clean_text(item.get("verification_label", ""))
    if label in {SUPPORTED, CONTRADICTED, INSUFFICIENT}:
        return label
    return INSUFFICIENT



def get_role_weight(item):
    if item.get("critical", False):
        return 1.0, "critical"
    if item.get("rely_on", []):
        return 0.9, "noncritical_dependent"
    return 0.95, "noncritical_root"



def get_source_weight(item):
    evidence_span = _clean_text(item.get("evidence_span", ""))
    return 1.0 if evidence_span else 0.75



def get_confidence(item):
    label = get_effective_label(item)
    evidence_span = _clean_text(item.get("evidence_span", ""))
    answer_status = item.get("answer_status", "insufficient")

    base = 0.85 if evidence_span else 0.60
    if label == INSUFFICIENT:
        base -= 0.15
    if answer_status in {"insufficient", "api_error"}:
        base -= 0.10
    if (item.get("adjudication", {}) or {}).get("triggered"):
        base -= 0.05
    return max(0.30, min(1.0, base))



def fact_score(item):
    label = get_effective_label(item)
    role_weight, role_name = get_role_weight(item)
    source_weight = get_source_weight(item)
    confidence = get_confidence(item)

    if label == SUPPORTED:
        base = 1.0
    elif label == CONTRADICTED:
        base = -1.5 if item.get("critical", False) else -1.1
    else:
        base = -0.4 if item.get("critical", False) else -0.15

    score = base * role_weight * source_weight * confidence
    return {
        "fact_id": item.get("fact_id", ""),
        "critical": item.get("critical", False),
        "verification_label": item.get("verification_label", INSUFFICIENT),
        "adjudicated_label": item.get("adjudicated_label", item.get("verification_label", INSUFFICIENT)),
        "effective_label": label,
        "role": role_name,
        "role_weight": role_weight,
        "source_weight": source_weight,
        "confidence": confidence,
        "score": round(score, 4),
    }



def normalize_verifications(data):
    fact_verification = data.get("fact_verification", {})
    items = fact_verification.get("verifications", []) if isinstance(fact_verification, dict) else []

    normalized = []
    for item in items:
        normalized.append({
            "fact_id": item.get("fact_id", ""),
            "fact_text": item.get("fact_text", ""),
            "bound_fact_text": item.get("bound_fact_text", ""),
            "question": item.get("question", ""),
            "question_type": item.get("question_type", ""),
            "answer": item.get("answer", ""),
            "answer_status": item.get("answer_status", "insufficient"),
            "verification_label": item.get("verification_label", INSUFFICIENT),
            "adjudicated_label": item.get("adjudicated_label", item.get("verification_label", INSUFFICIENT)),
            "evidence_span": item.get("evidence_span", ""),
            "constraint": item.get("constraint", {}),
            "rely_on": item.get("rely_on", []),
            "critical": item.get("critical", False),
            "adjudication": item.get("adjudication", {}),
        })
    return normalized



def aggregate_labels(verifications):
    critical_items = [item for item in verifications if item.get("critical", False)]
    noncritical_items = [item for item in verifications if not item.get("critical", False)]

    support_count = sum(1 for item in verifications if get_effective_label(item) == SUPPORTED)
    contradict_count = sum(1 for item in verifications if get_effective_label(item) == CONTRADICTED)
    insufficient_count = sum(1 for item in verifications if get_effective_label(item) == INSUFFICIENT)

    critical_support = sum(1 for item in critical_items if get_effective_label(item) == SUPPORTED)
    critical_contradict = sum(1 for item in critical_items if get_effective_label(item) == CONTRADICTED)
    critical_insufficient = sum(1 for item in critical_items if get_effective_label(item) == INSUFFICIENT or item.get("answer_status") == "api_error")

    noncritical_insufficient = sum(1 for item in noncritical_items if get_effective_label(item) == INSUFFICIENT)
    allow_noncritical_insufficient = max(1, math.floor(len(noncritical_items) * 0.3)) if noncritical_items else 0
    adjudicated_conflicts = sum(1 for item in verifications if (item.get("adjudication", {}) or {}).get("triggered"))

    per_fact_scores = [fact_score(item) for item in verifications]
    total_score = round(sum(item["score"] for item in per_fact_scores), 4)

    if critical_contradict > 0:
        return {
            "final_label": "refutes",
            "final_label_binary": "refutes",
            "decision_reason": "At least one critical fact is contradicted after adjudication.",
            "counts": {
                "support": support_count,
                "contradict": contradict_count,
                "insufficient": insufficient_count,
                "critical_support": critical_support,
                "critical_contradict": critical_contradict,
                "critical_insufficient": critical_insufficient,
                "adjudicated_conflicts": adjudicated_conflicts,
            },
            "noncritical_insufficient_allowance": allow_noncritical_insufficient,
            "total_score": total_score,
            "per_fact_scores": per_fact_scores,
        }

    if critical_insufficient > 0:
        return {
            "final_label": "not enough information",
            "final_label_binary": "refutes",
            "decision_reason": "At least one critical fact is still insufficient after adjudication.",
            "counts": {
                "support": support_count,
                "contradict": contradict_count,
                "insufficient": insufficient_count,
                "critical_support": critical_support,
                "critical_contradict": critical_contradict,
                "critical_insufficient": critical_insufficient,
                "adjudicated_conflicts": adjudicated_conflicts,
            },
            "noncritical_insufficient_allowance": allow_noncritical_insufficient,
            "total_score": total_score,
            "per_fact_scores": per_fact_scores,
        }

    if contradict_count > 0:
        return {
            "final_label": "refutes",
            "final_label_binary": "refutes",
            "decision_reason": "There is at least one contradicted fact after adjudication.",
            "counts": {
                "support": support_count,
                "contradict": contradict_count,
                "insufficient": insufficient_count,
                "critical_support": critical_support,
                "critical_contradict": critical_contradict,
                "critical_insufficient": critical_insufficient,
                "adjudicated_conflicts": adjudicated_conflicts,
            },
            "noncritical_insufficient_allowance": allow_noncritical_insufficient,
            "total_score": total_score,
            "per_fact_scores": per_fact_scores,
        }

    all_critical_supported = critical_support == len(critical_items)
    few_noncritical_insufficient = noncritical_insufficient <= allow_noncritical_insufficient
    if all_critical_supported and few_noncritical_insufficient:
        return {
            "final_label": "supports",
            "final_label_binary": "supports",
            "decision_reason": "All critical facts are supported after adjudication; non-critical insufficient facts are within tolerance.",
            "counts": {
                "support": support_count,
                "contradict": contradict_count,
                "insufficient": insufficient_count,
                "critical_support": critical_support,
                "critical_contradict": critical_contradict,
                "critical_insufficient": critical_insufficient,
                "adjudicated_conflicts": adjudicated_conflicts,
            },
            "noncritical_insufficient_allowance": allow_noncritical_insufficient,
            "total_score": total_score,
            "per_fact_scores": per_fact_scores,
        }

    return {
        "final_label": "not enough information",
        "final_label_binary": "refutes",
        "decision_reason": "Critical facts are supported, but too many non-critical facts remain insufficient after adjudication.",
        "counts": {
            "support": support_count,
            "contradict": contradict_count,
            "insufficient": insufficient_count,
            "critical_support": critical_support,
            "critical_contradict": critical_contradict,
            "critical_insufficient": critical_insufficient,
            "adjudicated_conflicts": adjudicated_conflicts,
        },
        "noncritical_insufficient_allowance": allow_noncritical_insufficient,
        "total_score": total_score,
        "per_fact_scores": per_fact_scores,
    }



def process_data_item(data):
    verifications = normalize_verifications(data)
    aggregation_result = aggregate_labels(verifications)
    out = dict(data)
    out["aggregation_result"] = aggregation_result
    return out



def main(args):
    in_path = (
        args.in_path
        .replace("[DATA]", args.dataset)
        .replace("[PLAN]", args.plan)
        .replace("[TYPE]", args.data_type)
        .replace("[CLASS]", args.class_num)
        .replace("[T]", args.t)
        .replace("[S]", str(args.start))
        .replace("[E]", str(args.end))
    )

    with open(in_path, "r", encoding="utf-8") as file_obj:
        raws = json.load(file_obj)
    raws = raws[args.start:args.end]

    results = []
    partial_func = partial(process_data_item)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(partial_func, data) for data in raws]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Error in future: {exc}")

    results = sorted(results, key=lambda item: item["id"])

    out_path = (
        args.out_path
        .replace("[DATA]", args.dataset)
        .replace("[PLAN]", args.plan)
        .replace("[TYPE]", args.data_type)
        .replace("[CLASS]", args.class_num)
        .replace("[T]", args.t)
        .replace("[S]", str(args.start))
        .replace("[E]", str(args.end))
    )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=4, ensure_ascii=False)

    print(f"Saved to {out_path}")
    print("Program finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HOVER")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--class_num", type=str, default="2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--plan", type=str, default="qc")
    parser.add_argument("--t", type=str, default="")
    parser.add_argument("--in_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_verify_[T][S]_[E].json")
    parser.add_argument("--out_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_aggregate_[T][S]_[E].json")
    main(parser.parse_args())
