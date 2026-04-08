import os
import math
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from datetime import datetime

SUPPORTED = "support"
CONTRADICTED = "contradict"
INSUFFICIENT = "insufficient"


def _clean_text(text):
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


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
    label = item.get("verification_label", INSUFFICIENT)
    evidence_span = _clean_text(item.get("evidence_span", ""))
    answer_status = item.get("answer_status", "insufficient")

    base = 0.85 if evidence_span else 0.60
    if label == INSUFFICIENT:
        base -= 0.15
    if answer_status in {"insufficient", "api_error"}:
        base -= 0.10
    return max(0.30, min(1.0, base))


def fact_score(item):
    label = item.get("verification_label", INSUFFICIENT)
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
        "verification_label": label,
        "role": role_name,
        "role_weight": role_weight,
        "source_weight": source_weight,
        "confidence": confidence,
        "score": round(score, 4),
    }


def normalize_verifications(data):
    fv = data.get("fact_verification", {})
    items = fv.get("verifications", []) if isinstance(fv, dict) else []

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
            "evidence_span": item.get("evidence_span", ""),
            "constraint": item.get("constraint", {}),
            "rely_on": item.get("rely_on", []),
            "critical": item.get("critical", False),
        })
    return normalized


def aggregate_labels(verifications):
    critical_items = [v for v in verifications if v.get("critical", False)]
    noncritical_items = [v for v in verifications if not v.get("critical", False)]

    support_count = sum(1 for v in verifications if v.get("verification_label") == SUPPORTED)
    contradict_count = sum(1 for v in verifications if v.get("verification_label") == CONTRADICTED)
    insufficient_count = sum(1 for v in verifications if v.get("verification_label") == INSUFFICIENT)

    critical_support = sum(1 for v in critical_items if v.get("verification_label") == SUPPORTED)
    critical_contradict = sum(1 for v in critical_items if v.get("verification_label") == CONTRADICTED)
    critical_insufficient = sum(
        1 for v in critical_items if v.get("verification_label") in {INSUFFICIENT} or v.get("answer_status") == "api_error"
    )

    noncritical_insufficient = sum(1 for v in noncritical_items if v.get("verification_label") == INSUFFICIENT)
    allow_noncritical_insufficient = max(1, math.floor(len(noncritical_items) * 0.3)) if noncritical_items else 0

    per_fact_scores = [fact_score(v) for v in verifications]
    total_score = round(sum(x["score"] for x in per_fact_scores), 4)

    # Rule 1: any critical contradiction => REFUTES
    if critical_contradict > 0:
        return {
            "final_label": "refutes",
            "final_label_binary": "refutes",
            "decision_reason": "At least one critical fact is contradicted.",
            "counts": {
                "support": support_count,
                "contradict": contradict_count,
                "insufficient": insufficient_count,
                "critical_support": critical_support,
                "critical_contradict": critical_contradict,
                "critical_insufficient": critical_insufficient,
            },
            "noncritical_insufficient_allowance": allow_noncritical_insufficient,
            "total_score": total_score,
            "per_fact_scores": per_fact_scores,
        }

    # Rule 2: any critical insufficient => NEI (binary => REFUTES)
    if critical_insufficient > 0:
        return {
            "final_label": "not enough information",
            "final_label_binary": "refutes",
            "decision_reason": "At least one critical fact is insufficient.",
            "counts": {
                "support": support_count,
                "contradict": contradict_count,
                "insufficient": insufficient_count,
                "critical_support": critical_support,
                "critical_contradict": critical_contradict,
                "critical_insufficient": critical_insufficient,
            },
            "noncritical_insufficient_allowance": allow_noncritical_insufficient,
            "total_score": total_score,
            "per_fact_scores": per_fact_scores,
        }

    # Rule 3: any contradiction anywhere => REFUTES
    if contradict_count > 0:
        return {
            "final_label": "refutes",
            "final_label_binary": "refutes",
            "decision_reason": "There is at least one contradicted fact.",
            "counts": {
                "support": support_count,
                "contradict": contradict_count,
                "insufficient": insufficient_count,
                "critical_support": critical_support,
                "critical_contradict": critical_contradict,
                "critical_insufficient": critical_insufficient,
            },
            "noncritical_insufficient_allowance": allow_noncritical_insufficient,
            "total_score": total_score,
            "per_fact_scores": per_fact_scores,
        }

    # Rule 4: SUPPORTS conditions
    all_critical_supported = (critical_support == len(critical_items))
    few_noncritical_insufficient = (noncritical_insufficient <= allow_noncritical_insufficient)

    if all_critical_supported and few_noncritical_insufficient:
        return {
            "final_label": "supports",
            "final_label_binary": "supports",
            "decision_reason": "All critical facts are supported; no contradictions; non-critical insufficient facts are within tolerance.",
            "counts": {
                "support": support_count,
                "contradict": contradict_count,
                "insufficient": insufficient_count,
                "critical_support": critical_support,
                "critical_contradict": critical_contradict,
                "critical_insufficient": critical_insufficient,
            },
            "noncritical_insufficient_allowance": allow_noncritical_insufficient,
            "total_score": total_score,
            "per_fact_scores": per_fact_scores,
        }

    # Rule 5: otherwise NEI
    return {
        "final_label": "not enough information",
        "final_label_binary": "refutes",
        "decision_reason": "Critical facts are supported, but too many non-critical facts remain insufficient.",
        "counts": {
            "support": support_count,
            "contradict": contradict_count,
            "insufficient": insufficient_count,
            "critical_support": critical_support,
            "critical_contradict": critical_contradict,
            "critical_insufficient": critical_insufficient,
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

    with open(in_path, "r", encoding="utf-8") as f:
        raws = json.load(f)

    results = []
    partial_func = partial(process_data_item)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(partial_func, data) for data in raws]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error in future: {e}")

    results = sorted(results, key=lambda x: x["id"])

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

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved to {out_path}")
    print("程序结束时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


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

    parser.add_argument(
        "--in_path",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_verify_[T][S]_[E].json"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_aggregate_[T][S]_[E].json"
    )

    args = parser.parse_args()
    main(args)