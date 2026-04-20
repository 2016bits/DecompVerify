import os
import re
import json
import argparse
import importlib.util
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm


_HELPER_PATH = Path(__file__).with_name("2.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("verify_helpers", _HELPER_PATH)
verify_helpers = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(verify_helpers)



def _clean_text(text):
    return verify_helpers._clean_text(text)



def _clean_list(values):
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    out = []
    for value in values:
        value = _clean_text(value)
        if value and value not in out:
            out.append(value)
    return out



def merge_bindings(base_bindings, extra_bindings):
    merged = dict(base_bindings or {})
    for key, value in (extra_bindings or {}).items():
        key = _clean_text(key)
        value = _clean_text(value)
        if key.startswith("?") and value:
            merged[key] = value
    return merged



def extract_initial_bindings(decomposition):
    entity_slots = (decomposition or {}).get("entity_slots", {}) or {}
    bindings = {}
    for slot, raw in entity_slots.items():
        slot = _clean_text(slot)
        if not slot.startswith("?"):
            continue
        if isinstance(raw, dict):
            value = _clean_text(raw.get("value", ""))
        else:
            value = _clean_text(raw)
        if value:
            bindings[slot] = value
    return bindings



def collect_observed_texts(answer, evidence_span, extracted_values, key):
    texts = []
    for value in _clean_list((extracted_values or {}).get(key, [])):
        if value not in texts:
            texts.append(value)
    for value in [_clean_text(answer), _clean_text(evidence_span)]:
        if value and value not in texts:
            texts.append(value)
    return texts



def build_time_check(answer, evidence_span, extracted_values, targets):
    observed_texts = collect_observed_texts(answer, evidence_span, extracted_values, "time")
    match = None
    for text in observed_texts:
        current = verify_helpers.time_match(text, targets)
        if current is True:
            match = True
            break
        if current is False:
            match = False if match is not True else True
    return {"targets": _clean_list(targets), "observed": observed_texts, "match": match}



def build_quantity_check(answer, evidence_span, extracted_values, targets, fact_text):
    observed_texts = collect_observed_texts(answer, evidence_span, extracted_values, "quantity")
    match = None
    for text in observed_texts:
        current = verify_helpers.quantity_match(text, targets, fact_text)
        if current is True:
            match = True
            break
        if current is False:
            match = False if match is not True else True
    return {"targets": _clean_list(targets), "observed": observed_texts, "match": match}



def build_location_check(answer, evidence_span, extracted_values, targets):
    observed_texts = collect_observed_texts(answer, evidence_span, extracted_values, "location")
    match = None
    if targets:
        saw_conflict = False
        for target in _clean_list(targets):
            if any(verify_helpers.text_match_loose(candidate, target) for candidate in observed_texts):
                match = True
                break
            if observed_texts:
                saw_conflict = True
        if match is not True and saw_conflict:
            match = False
    return {"targets": _clean_list(targets), "observed": observed_texts, "match": match}



def evaluate_constraint_checks(constraint, answer, evidence_span, extracted_values, fact_text):
    checks = {
        "time": build_time_check(answer, evidence_span, extracted_values, (constraint or {}).get("time", [])),
        "quantity": build_quantity_check(answer, evidence_span, extracted_values, (constraint or {}).get("quantity", []), fact_text),
        "location": build_location_check(answer, evidence_span, extracted_values, (constraint or {}).get("location", [])),
    }
    required_keys = [key for key, payload in checks.items() if payload.get("targets")]
    all_required_satisfied = all(checks[key].get("match") is True for key in required_keys) if required_keys else True
    explicit_conflict = any(checks[key].get("match") is False for key in checks)
    return {
        "checks": checks,
        "required_keys": required_keys,
        "all_required_satisfied": all_required_satisfied,
        "explicit_conflict": explicit_conflict,
    }



def infer_relation_yesno_mode(fact, qitem, question_text):
    negated_fact = verify_helpers.contains_negation(fact)
    question_polarity = _clean_text((qitem or {}).get("question_polarity", "literal_fact")) or "literal_fact"
    question_has_negation = bool(
        re.search(r"\b(?:no|not|never|none|without)\b", _clean_text(question_text).lower())
    )

    # Some questions are tagged as positive probes even though the surface form
    # still contains a negation cue like "no", "never", or "none".
    if question_polarity == "positive_probe_for_negation" and not question_has_negation:
        return "positive_probe_for_negation"
    if negated_fact:
        return "literal_negated_fact"
    return "literal_fact"


def resolve_relation_yesno_label(mode, yesno, grounded, constraint_eval):
    if not grounded:
        return "insufficient", "The yes/no answer is not grounded enough in the provided evidence.", {"yesno": yesno, "grounded": grounded, "explicit_conflict": False}

    if mode == "positive_probe_for_negation":
        if yesno == "yes":
            return "contradict", "The positive yes/no probe is answered yes, so the negated fact is refuted.", {"yesno": yesno, "grounded": grounded, "explicit_conflict": True}
        if yesno == "no" and constraint_eval.get("all_required_satisfied", False):
            return "support", "The positive yes/no probe is answered no and the required constraints are grounded.", {"yesno": yesno, "grounded": grounded, "explicit_conflict": False}
        return "insufficient", "The negated fact cannot be resolved with a grounded no-answer unless the required constraints are satisfied.", {"yesno": yesno, "grounded": grounded, "explicit_conflict": constraint_eval.get("explicit_conflict", False)}

    if mode == "literal_negated_fact":
        if yesno == "yes" and constraint_eval.get("all_required_satisfied", False):
            return "support", "The answer directly affirms the negated fact and the required constraints are grounded.", {"yesno": yesno, "grounded": grounded, "explicit_conflict": False}
        if yesno == "no":
            return "contradict", "The answer denies the negated fact.", {"yesno": yesno, "grounded": grounded, "explicit_conflict": True}
        return "insufficient", "The negated fact cannot be confirmed because the required constraints are not fully satisfied.", {"yesno": yesno, "grounded": grounded, "explicit_conflict": constraint_eval.get("explicit_conflict", False)}

    if yesno == "yes":
        return "support", "The answer affirms the fact.", {"yesno": yesno, "grounded": grounded, "explicit_conflict": False}
    return "contradict", "The answer denies the fact.", {"yesno": yesno, "grounded": grounded, "explicit_conflict": True}


def recover_entity_wh_from_constraints(answer_status, evidence_span, gold_evidence, constraint_eval):
    if answer_status not in {"insufficient", "api_error"}:
        return None
    if not verify_helpers.span_grounded_in_evidence(evidence_span, gold_evidence):
        return None
    if constraint_eval.get("explicit_conflict", False):
        return None

    matched_keys = [
        key for key in ("time", "quantity")
        if (constraint_eval.get("checks", {}).get(key, {}) or {}).get("match") is True
    ]
    if not matched_keys:
        return None

    joined = ", ".join(matched_keys)
    return (
        "support",
        f"The direct entity answer was missing, but the grounded evidence span satisfies the {joined} constraint(s).",
        {"explicit_conflict": False},
    )


def verify_relation_yesno(fact, qitem, answer, answer_status, evidence_span, gold_evidence, extracted_values, constraint_eval):
    grounded = verify_helpers.span_grounded_in_evidence(evidence_span, gold_evidence)
    yesno = (extracted_values or {}).get("yesno") or verify_helpers.normalize_yesno(answer)

    if answer_status in {"insufficient", "api_error"} or yesno is None:
        return "insufficient", "The answer does not clearly determine the fact.", {"yesno": yesno, "grounded": grounded, "explicit_conflict": False}

    question_text = _clean_text((qitem or {}).get("main_question", ""))
    mode = infer_relation_yesno_mode(fact, qitem, question_text)
    return resolve_relation_yesno_label(mode, yesno, grounded, constraint_eval)



def verify_time_wh(fact, answer_status, constraint_eval):
    negated = verify_helpers.contains_negation(fact)
    matched = constraint_eval["checks"]["time"].get("match")
    if matched is True:
        return ("contradict" if negated else "support"), "The extracted time value matches the time constraint.", {"explicit_conflict": False}
    if matched is False and (fact.get("constraint", {}) or {}).get("time"):
        return ("support" if negated else "contradict"), "The extracted time value conflicts with the time constraint.", {"explicit_conflict": True}
    if answer_status in {"insufficient", "api_error"}:
        return "insufficient", "The answer does not provide enough temporal information.", {"explicit_conflict": False}
    return "insufficient", "The answer does not determine the time constraint.", {"explicit_conflict": False}



def verify_quantity_wh(fact, answer_status, constraint_eval):
    negated = verify_helpers.contains_negation(fact)
    matched = constraint_eval["checks"]["quantity"].get("match")
    if matched is True:
        return ("contradict" if negated else "support"), "The extracted quantity value matches the quantity constraint.", {"explicit_conflict": False}
    if matched is False:
        return ("support" if negated else "contradict"), "The extracted quantity value conflicts with the quantity constraint.", {"explicit_conflict": True}
    if answer_status in {"insufficient", "api_error"}:
        return "insufficient", "The answer does not provide enough quantity information.", {"explicit_conflict": False}
    return "insufficient", "The answer does not determine the quantity constraint.", {"explicit_conflict": False}



def verify_fallback(fact, answer, answer_status, evidence_span, gold_evidence):
    negated = verify_helpers.contains_negation(fact)
    if answer_status == "contradicted":
        return ("support" if negated else "contradict"), "The answer result contradicts the fact.", {"explicit_conflict": True}
    if answer_status in {"insufficient", "api_error"} or _clean_text(answer).lower() == "insufficient":
        return "insufficient", "The answer does not provide enough information.", {"explicit_conflict": False}
    if verify_helpers.span_grounded_in_evidence(evidence_span, gold_evidence):
        return ("contradict" if negated else "support"), "The answer provides a grounded value for the fact.", {"explicit_conflict": False}
    return "insufficient", "The answer is not sufficiently grounded in the evidence.", {"explicit_conflict": False}



def run_second_pass_adjudication(fact, qitem, answer_status, initial_label, evidence_span, gold_evidence, constraint_eval, verifier_metadata):
    if answer_status != "supported" or initial_label == "support":
        return {"triggered": False, "final_label": initial_label, "reason": "Second-pass adjudication not needed."}

    grounded = verify_helpers.span_grounded_in_evidence(evidence_span, gold_evidence)
    question_type = _clean_text((qitem or {}).get("question_type", ""))
    yesno = verifier_metadata.get("yesno")
    explicit_conflict = verifier_metadata.get("explicit_conflict") or constraint_eval.get("explicit_conflict", False)

    if explicit_conflict:
        final_label = "contradict" if initial_label == "contradict" else "insufficient"
        reason = "Second pass keeps the verifier rejection because the extracted values show an explicit conflict."
    elif not grounded:
        final_label = "insufficient"
        reason = "Second pass downgrades to insufficient because the supported answer is not grounded in the provided evidence span."
    elif question_type == "relation_yesno" and yesno in {"yes", "no"}:
        mode = infer_relation_yesno_mode(fact, qitem, _clean_text((qitem or {}).get("main_question", "")))
        final_label, _, _ = resolve_relation_yesno_label(mode, yesno, grounded, constraint_eval)
        reason = "Second pass resolves the answer/verifier conflict using the grounded yes/no answer and the constraint checks."
    elif question_type == "time_wh" and constraint_eval["checks"]["time"].get("match") is True:
        final_label = "support"
        reason = "Second pass prefers the grounded supported answer because the extracted time values do match."
    elif question_type == "quantity_wh" and constraint_eval["checks"]["quantity"].get("match") is True:
        final_label = "support"
        reason = "Second pass prefers the grounded supported answer because the extracted quantity values do match."
    elif question_type == "entity_wh" and grounded:
        final_label = "support"
        reason = "Second pass prefers the grounded supported answer because no explicit entity conflict remains."
    else:
        final_label = "insufficient"
        reason = "Second pass could not safely reconcile the supported answer with the verifier rejection."

    return {"triggered": True, "final_label": final_label, "reason": reason}



def build_maps(data):
    decomposition = data.get("decomposition", {}) or {}
    question_plan = data.get("question_plan", {}) or {}
    answer_result = data.get("answer_result", {}) or {}

    facts = decomposition.get("atomic_facts", []) or []
    question_items = question_plan.get("question_items", []) or []
    answers = answer_result.get("answers", []) or []

    initial_bindings = answer_result.get("initial_bindings", {}) or extract_initial_bindings(decomposition)
    final_bindings = merge_bindings(initial_bindings, answer_result.get("final_bindings", {}) or {})

    return (
        {fact.get("id"): fact for fact in facts},
        {item.get("fact_id"): item for item in question_items},
        {answer.get("fact_id"): answer for answer in answers},
        final_bindings,
    )



def verify_one_fact(fact, qitem, aitem, final_bindings, gold_evidence):
    raw_fact_text = _clean_text(fact.get("text", ""))
    bound_fact_text = verify_helpers.replace_placeholders(raw_fact_text, final_bindings)

    question = _clean_text((aitem or {}).get("question", "") or (qitem or {}).get("main_question", ""))
    bound_question = verify_helpers.replace_placeholders(question, final_bindings)

    answer = _clean_text((aitem or {}).get("answer", "insufficient"))
    answer_status = verify_helpers.normalize_status((aitem or {}).get("status", "insufficient"))
    evidence_span = _clean_text((aitem or {}).get("evidence_span", ""))
    extracted_values = (aitem or {}).get("extracted_values", {}) or {}

    question_type = _clean_text((qitem or {}).get("question_type", ""))
    question_polarity = _clean_text((qitem or {}).get("question_polarity", "literal_fact")) or "literal_fact"
    constraint = fact.get("constraint", {}) or {}
    critical = fact.get("critical", False)

    still_unresolved = verify_helpers.unresolved_vars(bound_fact_text) or verify_helpers.unresolved_vars(bound_question)
    if still_unresolved and answer_status in {"insufficient", "api_error"}:
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
            "verification_label": "insufficient",
            "adjudicated_label": "insufficient",
            "reason": f"Unresolved variables remain: {still_unresolved}",
            "evidence_span": evidence_span,
            "constraint": constraint,
            "rely_on": fact.get("rely_on", []),
            "critical": critical,
            "critical_reasons": fact.get("critical_reasons", []),
            "extracted_values": extracted_values,
            "value_checks": {},
            "adjudication": {"triggered": False, "final_label": "insufficient", "reason": "Unresolved variables remained before adjudication."},
        }

    constraint_eval = evaluate_constraint_checks(constraint, answer, evidence_span, extracted_values, bound_fact_text)

    if question_type == "relation_yesno":
        verification_label, reason, verifier_metadata = verify_relation_yesno(fact, qitem, answer, answer_status, evidence_span, gold_evidence, extracted_values, constraint_eval)
    elif question_type == "time_wh":
        verification_label, reason, verifier_metadata = verify_time_wh(fact, answer_status, constraint_eval)
    elif question_type == "quantity_wh":
        verification_label, reason, verifier_metadata = verify_quantity_wh(fact, answer_status, constraint_eval)
    elif question_type == "entity_wh":
        verification_label, reason = verify_helpers.verify_entity_wh(
            fact={"text": bound_fact_text},
            answer=answer,
            answer_status=answer_status,
            evidence_span=evidence_span,
            gold_evidence=gold_evidence,
        )
        verifier_metadata = {"explicit_conflict": verification_label == "contradict"}
        recovered = recover_entity_wh_from_constraints(answer_status, evidence_span, gold_evidence, constraint_eval)
        if verification_label == "insufficient" and recovered is not None:
            verification_label, reason, verifier_metadata = recovered
    else:
        verification_label, reason, verifier_metadata = verify_fallback(fact, answer, answer_status, evidence_span, gold_evidence)

    adjudication = run_second_pass_adjudication(
        fact=fact,
        qitem=qitem,
        answer_status=answer_status,
        initial_label=verification_label,
        evidence_span=evidence_span,
        gold_evidence=gold_evidence,
        constraint_eval=constraint_eval,
        verifier_metadata=verifier_metadata,
    )
    adjudicated_label = adjudication.get("final_label", verification_label)

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
        "adjudicated_label": adjudicated_label,
        "reason": reason,
        "evidence_span": evidence_span,
        "constraint": constraint,
        "rely_on": fact.get("rely_on", []),
        "critical": critical,
        "critical_reasons": fact.get("critical_reasons", []),
        "extracted_values": extracted_values,
        "value_checks": constraint_eval,
        "adjudication": adjudication,
    }



def process_data_item(data):
    fact_map, q_map, a_map, final_bindings = build_maps(data)
    ordered_fact_ids = [fact.get("id") for fact in (data.get("decomposition", {}) or {}).get("atomic_facts", []) or []]

    fact_verification = []
    gold_evidence = data.get("gold_evidence", data.get("evidence", ""))
    for fact_id in ordered_fact_ids:
        fact = fact_map.get(fact_id, {})
        qitem = q_map.get(fact_id, {})
        aitem = a_map.get(fact_id, {})
        fact_verification.append(verify_one_fact(fact, qitem, aitem, final_bindings, gold_evidence))

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
    }



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
    parser.add_argument("--in_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_answers_[T][S]_[E].json")
    parser.add_argument("--out_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_verify_[T][S]_[E].json")
    main(parser.parse_args())
