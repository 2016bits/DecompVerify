
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

SUPPORTED = "support"
CONTRADICTED = "contradict"
INSUFFICIENT = "insufficient"


def _clean_text(text):
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


def get_role_weight(fact_text: str, rely_on):
    """
    Lightweight role approximation for the current minimal schema:
    - root facts / endpoint factual claims: verify-like
    - intermediate bound/entity bridge facts: bridge-like
    """
    rely_on = rely_on or []
    fact_text = _clean_text(fact_text).lower()

    bridge_markers = [
        "has a director",
        "has a capacity",
        "was taught by",
        "is named after",
        "has a home ground",
        "there exists an author",
        "there exists a person",
        "there is a film",
        "there is a documentary",
        "there is an acting academy",
        "a person is an ambassador",
        "that person",
        "that actress",
        "that animal",
        "that author",
        "that composer",
        "that football club",
        "that naturalist",
        "that village",
    ]

    if any(m in fact_text for m in bridge_markers):
        return 0.85, "bridge"
    if not rely_on:
        return 1.0, "verify"
    return 0.95, "verify"


def get_source_weight(item):
    evidence_span = _clean_text(item.get("evidence_span", ""))
    return 1.0 if evidence_span else 0.85


def get_confidence(item):
    label = item.get("verification_label", INSUFFICIENT)
    evidence_span = _clean_text(item.get("evidence_span", ""))
    answer_status = item.get("answer_status", "insufficient")

    base = 0.85 if evidence_span else 0.65
    if label == INSUFFICIENT:
        base -= 0.15
    if answer_status == "insufficient":
        base -= 0.10
    return max(0.3, min(1.0, base))


def fact_score(item):
    label = item.get("verification_label", INSUFFICIENT)
    role_weight, role_name = get_role_weight(item.get("fact_text", ""), item.get("rely_on", []))
    source_weight = get_source_weight(item)
    confidence = get_confidence(item)

    if label == SUPPORTED:
        base = 1.0
    elif label == CONTRADICTED:
        base = -1.25
    else:
        base = -0.20

    score = base * role_weight * source_weight * confidence
    return {
        "fact_id": item.get("fact_id", ""),
        "verification_label": label,
        "role": role_name,
        "role_weight": role_weight,
        "source_weight": source_weight,
        "confidence": confidence,
        "score": round(score, 4),
    }


def is_constraint_fact(item):
    constraint = item.get("constraint", {}) or {}
    return bool(constraint.get("negation") is True or constraint.get("time") or constraint.get("quantity"))


def constraint_score(item):
    if not is_constraint_fact(item):
        return None

    label = item.get("verification_label", INSUFFICIENT)
    if label == SUPPORTED:
        score = 0.7
    elif label == CONTRADICTED:
        score = -1.2
    else:
        score = -0.15

    return {
        "fact_id": item.get("fact_id", ""),
        "verification_label": label,
        "constraint_score": score,
    }


def has_hard_contradiction(verifications):
    for item in verifications:
        label = item.get("verification_label", INSUFFICIENT)
        evidence_span = _clean_text(item.get("evidence_span", ""))
        rely_on = item.get("rely_on", []) or []
        _, role_name = get_role_weight(item.get("fact_text", ""), rely_on)
        if label == CONTRADICTED and evidence_span and role_name == "verify":
            return True, item.get("fact_id", "")
    return False, None


def has_constraint_contradiction(verifications):
    for item in verifications:
        if is_constraint_fact(item) and item.get("verification_label") == CONTRADICTED:
            return True, item.get("fact_id", "")
    return False, None


def aggregate_labels(verifications):
    per_fact_scores = [fact_score(v) for v in verifications]
    per_constraint_scores = [constraint_score(v) for v in verifications]
    per_constraint_scores = [x for x in per_constraint_scores if x is not None]

    fact_total = sum(x["score"] for x in per_fact_scores)
    constraint_total = sum(x["constraint_score"] for x in per_constraint_scores)
    total = fact_total + constraint_total

    support_count = sum(1 for v in verifications if v.get("verification_label") == SUPPORTED)
    contradict_count = sum(1 for v in verifications if v.get("verification_label") == CONTRADICTED)
    insufficient_count = sum(1 for v in verifications if v.get("verification_label") == INSUFFICIENT)

    hard_flag, hard_fact = has_hard_contradiction(verifications)
    if hard_flag:
        return {
            "final_label": "refutes",
            "decision_reason": f"Hard contradiction triggered by {hard_fact}.",
            "fact_total_score": round(fact_total, 4),
            "constraint_total_score": round(constraint_total, 4),
            "total_score": round(total, 4),
            "counts": {
                "support": support_count,
                "contradict": contradict_count,
                "insufficient": insufficient_count,
            },
            "per_fact_scores": per_fact_scores,
            "per_constraint_scores": per_constraint_scores,
        }

    cflag, c_fact = has_constraint_contradiction(verifications)
    if cflag:
        return {
            "final_label": "refutes",
            "decision_reason": f"Constraint contradiction triggered by {c_fact}.",
            "fact_total_score": round(fact_total, 4),
            "constraint_total_score": round(constraint_total, 4),
            "total_score": round(total, 4),
            "counts": {
                "support": support_count,
                "contradict": contradict_count,
                "insufficient": insufficient_count,
            },
            "per_fact_scores": per_fact_scores,
            "per_constraint_scores": per_constraint_scores,
        }

    if total <= -0.8:
        final = "refutes"
        reason = "Total score below contradiction threshold."
    elif total >= 1.0 and support_count >= max(1, len(verifications) // 2) and contradict_count == 0:
        final = "supports"
        reason = "Total score and support coverage meet support condition."
    else:
        final = "not enough information"
        reason = "Evidence does not satisfy support or contradiction conditions."

    return {
        "final_label": final,
        "decision_reason": reason,
        "fact_total_score": round(fact_total, 4),
        "constraint_total_score": round(constraint_total, 4),
        "total_score": round(total, 4),
        "counts": {
            "support": support_count,
            "contradict": contradict_count,
            "insufficient": insufficient_count,
        },
        "per_fact_scores": per_fact_scores,
        "per_constraint_scores": per_constraint_scores,
    }


def normalize_verifications(data):
    fv = data.get("fact_verification", {})
    if isinstance(fv, dict) and "verifications" in fv:
        items = fv.get("verifications", []) or []
    else:
        vr = data.get("verification_result", {}) or {}
        items = vr.get("fact_verifications", []) or []

    normalized = []
    for item in items:
        normalized.append({
            "fact_id": item.get("fact_id", ""),
            "fact_text": item.get("fact_text", item.get("atomic_fact", {}).get("text", "")),
            "bound_fact_text": item.get("bound_fact_text", ""),
            "question": item.get("question", item.get("question_used", "")),
            "question_type": item.get("question_type", ""),
            "answer": item.get("answer", ""),
            "answer_status": item.get("answer_status", item.get("previous_status", "insufficient")),
            "verification_label": item.get("verification_label", INSUFFICIENT),
            "evidence_span": item.get("evidence_span", ""),
            "constraint": item.get("constraint", item.get("atomic_fact", {}).get("constraint", {})),
            "rely_on": item.get("rely_on", item.get("atomic_fact", {}).get("rely_on", [])),
        })
    return normalized


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
                continue

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
