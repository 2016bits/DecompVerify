import argparse
import contextlib
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


# ---------------------------------------------------------------------------
# Existing baseline: wo_constraint_full (drop the whole constraint stack)
# ---------------------------------------------------------------------------


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
            "mode": "wo_constraint_full",
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
            "mode": "wo_constraint_full",
            "answer_status_mapping": {
                "supported": "support",
                "contradicted": "contradict",
                "insufficient": "insufficient",
                "api_error": "insufficient",
            },
        },
    }


# ---------------------------------------------------------------------------
# Group 2: sub-mechanism ablations of the constraint stack
# ---------------------------------------------------------------------------
#
# These three ablations all reuse verify_atomic_facts.verify_one_fact via
# monkey-patching its building blocks. This way every other piece of logic
# (question_type dispatch, evidence grounding, etc.) stays in sync with the
# canonical implementation.


def _empty_constraint_eval(constraint, answer, evidence_span, extracted_values, fact_text):
    """Stand-in for evaluate_constraint_checks that produces a neutral shape.

    Every check is reported as `match=None` with no targets, so downstream
    code that consults `constraint_eval["checks"][...]` will see "nothing to
    verify". `explicit_conflict` is False so the second-pass adjudication
    cannot use value-level conflicts."""
    neutral_check = {"match": None, "targets": [], "details": {}}
    return {
        "checks": {
            "time": dict(neutral_check),
            "quantity": dict(neutral_check),
            "location": dict(neutral_check),
            "comparison": dict(neutral_check),
            "unique_modifier": dict(neutral_check),
            "title_role": dict(neutral_check),
            "exact_name": {"match": None, "match_all": None, "targets": [], "details": {}},
        },
        "required_keys": [],
        "all_required_satisfied": True,
        "explicit_conflict": False,
        "textual_required_keys": [],
        "all_textual_satisfied": True,
        "any_textual_missed": False,
        "exact_name_match": None,
        "exact_name_all_match": None,
    }


def _noop_second_pass(fact, qitem, answer_status, initial_label, evidence_span,
                      gold_evidence, constraint_eval, verifier_metadata):
    return {
        "triggered": False,
        "final_label": initial_label,
        "reason": "Ablation: second-pass adjudication disabled (wo_adjudication).",
    }


@contextlib.contextmanager
def _patch(target_module, attr, replacement):
    sentinel = object()
    original = getattr(target_module, attr, sentinel)
    setattr(target_module, attr, replacement)
    try:
        yield
    finally:
        if original is sentinel:
            delattr(target_module, attr)
        else:
            setattr(target_module, attr, original)


def _force_literal_polarity(fact, qitem, aitem, final_bindings, gold_evidence):
    """Wrap verify_one_fact while forcing question_polarity='literal_fact'.

    Negated facts/questions still parse normally via verify_helpers, but the
    polarity tag itself stops biasing relation_yesno mode inference."""
    patched_qitem = dict(qitem or {})
    patched_qitem["question_polarity"] = "literal_fact"
    return verify_atomic_facts.verify_one_fact(fact, patched_qitem, aitem, final_bindings, gold_evidence)


def _verify_with_disabled_value_checks(fact, qitem, aitem, final_bindings, gold_evidence):
    with _patch(verify_atomic_facts, "evaluate_constraint_checks", _empty_constraint_eval):
        return verify_atomic_facts.verify_one_fact(fact, qitem, aitem, final_bindings, gold_evidence)


def _verify_with_disabled_adjudication(fact, qitem, aitem, final_bindings, gold_evidence):
    with _patch(verify_atomic_facts, "run_second_pass_adjudication", _noop_second_pass):
        return verify_atomic_facts.verify_one_fact(fact, qitem, aitem, final_bindings, gold_evidence)


def _verify_with_no_runtime_binding(fact, qitem, aitem, final_bindings, gold_evidence):
    """Group 3B: keep rely_on structure but stop substituting ?slot tokens at
    verify time. The verifier sees the un-bound fact / question text."""
    return verify_atomic_facts.verify_one_fact(fact, qitem, aitem, {}, gold_evidence)


# ---------------------------------------------------------------------------
# Generic process_data_item that swaps verify_one_fact for a given variant
# ---------------------------------------------------------------------------


VARIANT_TO_VERIFIER = {
    "wo_adjudication": _verify_with_disabled_adjudication,
    "wo_value_checks": _verify_with_disabled_value_checks,
    "wo_polarity": _force_literal_polarity,
    "wo_runtime_binding": _verify_with_no_runtime_binding,
}


def process_data_item_variant(data, mode):
    if mode == "wo_constraint_full":
        return process_data_item_no_constraint(data)

    verifier = VARIANT_TO_VERIFIER[mode]
    fact_map, q_map, a_map, final_bindings = verify_atomic_facts.build_maps(data)
    ordered_fact_ids = [fact.get("id") for fact in (data.get("decomposition", {}) or {}).get("atomic_facts", []) or []]

    if mode == "wo_runtime_binding":
        # Aggregator will use the same view of bindings; emit empty to avoid
        # leaking placeholder substitutions through downstream consumers.
        emitted_bindings = {}
    else:
        emitted_bindings = final_bindings

    fact_verification = []
    gold_evidence = data.get("gold_evidence", data.get("evidence", ""))
    for fact_id in ordered_fact_ids:
        fact = fact_map.get(fact_id, {})
        qitem = q_map.get(fact_id, {})
        aitem = a_map.get(fact_id, {})
        v = verifier(fact, qitem, aitem, final_bindings, gold_evidence)
        v.setdefault("ablation", {})
        v["ablation"]["mode"] = mode
        fact_verification.append(v)

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
            "final_bindings": emitted_bindings,
            "verifications": fact_verification,
        },
        "verification_ablation": {
            "mode": mode,
            "description": _MODE_DESCRIPTIONS[mode],
        },
    }


_MODE_DESCRIPTIONS = {
    "wo_constraint_full": "Drop the entire constraint stack; map answer.status directly to fact label.",
    "wo_adjudication": "Skip the second-pass adjudication; adjudicated_label := verification_label.",
    "wo_value_checks": "Replace evaluate_constraint_checks with a no-op; type-specific verifiers run on empty checks.",
    "wo_polarity": "Force question_polarity='literal_fact' so the relation_yesno polarity branch never fires.",
    "wo_runtime_binding": "Keep rely_on structure but do not substitute ?slot placeholders before verification.",
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


SUPPORTED_MODES = [
    "full",
    "wo_constraint_full",
    "wo_adjudication",
    "wo_value_checks",
    "wo_polarity",
    "wo_runtime_binding",
    # Back-compat alias for the old shell scripts.
    "no_constraint",
]


def main(args):
    in_path = resolve_path(args.in_path, args)

    with open(in_path, "r", encoding="utf-8") as file_obj:
        raws = json.load(file_obj)
    raws = raws[args.start:args.end]

    mode = args.mode
    if mode == "no_constraint":
        mode = "wo_constraint_full"

    if mode == "full":
        partial_func = partial(verify_atomic_facts.process_data_item)
    else:
        partial_func = partial(process_data_item_variant, mode=mode)

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
    parser.add_argument("--mode", type=str, choices=SUPPORTED_MODES, default="wo_constraint_full")
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
