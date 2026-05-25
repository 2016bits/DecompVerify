import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial

from tqdm import tqdm

import verify_atomic_facts
import verify_helpers


SUPPORTED_STATUS = "supported"
CONTRADICTED_STATUS = "contradicted"
INSUFFICIENT_STATUS = "insufficient"
API_ERROR_STATUS = "api_error"


def status_to_fact_label(answer_status):
    answer_status = verify_helpers.normalize_status(answer_status)
    if answer_status == SUPPORTED_STATUS:
        return "support"
    if answer_status == CONTRADICTED_STATUS:
        return "contradict"
    if answer_status in {INSUFFICIENT_STATUS, API_ERROR_STATUS}:
        return "insufficient"
    return "insufficient"


def verify_one_fact_no_constraint(fact, qitem, aitem, final_bindings, gold_evidence):
    raw_fact_text = verify_atomic_facts._clean_text(fact.get("text", ""))
    bound_fact_text = verify_helpers.replace_placeholders(raw_fact_text, final_bindings)

    question = verify_atomic_facts._clean_text((aitem or {}).get("question", "") or (qitem or {}).get("main_question", ""))
    bound_question = verify_helpers.replace_placeholders(question, final_bindings)

    answer = verify_atomic_facts._clean_text((aitem or {}).get("answer", "insufficient"))
    answer_status = verify_helpers.normalize_status((aitem or {}).get("status", "insufficient"))
    evidence_span = verify_atomic_facts._clean_text((aitem or {}).get("evidence_span", ""))
    extracted_values = (aitem or {}).get("extracted_values", {}) or {}

    question_type = verify_atomic_facts._clean_text((qitem or {}).get("question_type", ""))
    question_polarity = verify_atomic_facts._clean_text((qitem or {}).get("question_polarity", "literal_fact")) or "literal_fact"
    constraint = fact.get("constraint", {}) or {}
    critical = fact.get("critical", False)

    verification_label = status_to_fact_label(answer_status)
    reason = (
        "Ablation: fact label is mapped directly from answer.status; "
        "constraint fields, negation polarity, temporal/numeric/entity checks, "
        "comparison checks, and second-pass adjudication are not used."
    )

    adjudication = {
        "triggered": False,
        "final_label": verification_label,
        "reason": "Ablation disables constraint-aware adjudication.",
    }

    return {
        "fact_id": fact.get("id", ""),
        "fact_text": raw_fact_text,
        "bound_fact_text": bound_fact_text,
        "question": question,
        "bound_question": bound_question,
        "question_type": question_type,
        "question_polarity": question_polarity,
        "answer": answer,
        "answer_status": answer_status,
        "verification_label": verification_label,
        "adjudicated_label": verification_label,
        "reason": reason,
        "evidence_span": evidence_span,
        "constraint": constraint,
        "rely_on": fact.get("rely_on", []),
        "critical": critical,
        "critical_reasons": fact.get("critical_reasons", []),
        "extracted_values": extracted_values,
        "value_checks": {},
        "adjudication": adjudication,
        "ablation": {
            "mode": "wo_constraint_aware_verification",
            "used_constraint_fields": False,
            "used_answer_status_only": True,
        },
    }


def process_data_item_no_constraint(data):
    fact_map, q_map, a_map, final_bindings = verify_atomic_facts.build_maps(data)
    ordered_fact_ids = [fact.get("id") for fact in (data.get("decomposition", {}) or {}).get("atomic_facts", []) or []]

    fact_verification = []
    gold_evidence = data.get("gold_evidence", data.get("evidence", ""))
    for fact_id in ordered_fact_ids:
        fact = fact_map.get(fact_id, {})
        qitem = q_map.get(fact_id, {})
        aitem = a_map.get(fact_id, {})
        fact_verification.append(
            verify_one_fact_no_constraint(fact, qitem, aitem, final_bindings, gold_evidence)
        )

    return {
        "id": data["id"],
        "claim": data["claim"],
        "gold_evidence": gold_evidence,
        "num_hops": data.get("num_hops", None),
        "label": data.get("label", None),
        "decomposition": data.get("decomposition", {}),
        "question_plan": data.get("question_plan", {}),
        "answer_result": data.get("answer_result", {}),
        "fact_verification": {
            "claim": data["claim"],
            "final_bindings": final_bindings,
            "verifications": fact_verification,
        },
        "verification_ablation": {
            "mode": "wo_constraint_aware_verification",
            "answer_status_mapping": {
                "supported": "support",
                "contradicted": "contradict",
                "insufficient": "insufficient",
                "api_error": "insufficient",
            },
        },
    }


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

    if args.mode == "full":
        partial_func = partial(verify_atomic_facts.process_data_item)
    elif args.mode == "no_constraint":
        partial_func = partial(process_data_item_no_constraint)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    results = []
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
    parser.add_argument("--mode", type=str, choices=["full", "no_constraint"], default="no_constraint")
    parser.add_argument("--dataset", type=str, default="HOVER")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--class_num", type=str, default="2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=4000)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--plan", type=str, default="qc_plan6.4")
    parser.add_argument("--t", type=str, default="")
    parser.add_argument("--in_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_answers_[T][S]_[E].json")
    parser.add_argument("--out_path", type=str, default="./data/[DATA]/[PLAN]/ablations/[TYPE]_[CLASS]_wo_constraint_verify_[T][S]_[E].json")
    main(parser.parse_args())
