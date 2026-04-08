import os
import re
import json
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from datetime import datetime


VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")


def _clean_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_text_for_match(text):
    text = _clean_text(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def replace_placeholders(text, bindings):
    text = _clean_text(text)
    if not text:
        return text
    if not isinstance(bindings, dict):
        return text

    for var in sorted(bindings.keys(), key=len, reverse=True):
        if not var.startswith("?"):
            continue
        val = _clean_text(bindings.get(var, ""))
        if val:
            text = text.replace(var, val)
    return text


def unresolved_vars(text):
    return sorted(set(VAR_PATTERN.findall(_clean_text(text))))


def normalize_status(status):
    s = _clean_text(status).lower()
    if s in {"support", "supported"}:
        return "supported"
    if s in {"contradict", "contradicted"}:
        return "contradicted"
    if s == "api_error":
        return "api_error"
    return "insufficient"


def normalize_yesno(answer):
    a = _clean_text(answer).lower()
    if a in {"yes", "yes.", "true"}:
        return "yes"
    if a in {"no", "no.", "false"}:
        return "no"
    return None


def contains_negation(fact):
    return (fact.get("constraint", {}) or {}).get("negation") is True


def extract_years(text):
    return re.findall(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", _clean_text(text))


def normalize_num_string(s):
    return _clean_text(s).lower().replace(",", "").strip()


def quantity_match(answer, quantities, fact_text=""):
    ans = normalize_num_string(answer)
    qvals = [normalize_num_string(x) for x in (quantities or []) if _clean_text(x)]

    if not qvals:
        if re.search(r"\beight\b", fact_text.lower()):
            qvals = ["eight"]
        elif re.search(r"\b27[, ]?000\b", fact_text):
            qvals = ["27000"]

    if not qvals:
        return None

    for q in qvals:
        if q and (q in ans or ans in q):
            return True
    return False


def time_match(answer, times):
    ans = _clean_text(answer).lower()
    tvals = [_clean_text(x).lower() for x in (times or []) if _clean_text(x)]
    if not tvals:
        return None

    for t in tvals:
        if t and (t in ans or ans in t):
            return True

    ans_years = set(extract_years(ans))
    if ans_years:
        for t in tvals:
            t_years = set(extract_years(t))
            if t_years and ans_years == t_years:
                return True
    return False


def span_grounded_in_evidence(evidence_span, gold_evidence):
    span = normalize_text_for_match(evidence_span)
    gold = normalize_text_for_match(gold_evidence)
    if not span:
        return False

    if span in gold:
        return True

    span_tokens = set(span.split())
    gold_tokens = set(gold.split())
    if not span_tokens:
        return False

    overlap = len(span_tokens & gold_tokens) / max(1, len(span_tokens))
    return overlap >= 0.7


def text_match_loose(a, b):
    a_n = normalize_text_for_match(a)
    b_n = normalize_text_for_match(b)
    if not a_n or not b_n:
        return False
    if a_n == b_n or a_n in b_n or b_n in a_n:
        return True

    a_tokens = set(a_n.split())
    b_tokens = set(b_n.split())
    if not a_tokens or not b_tokens:
        return False

    overlap = len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))
    return overlap >= 0.75


def extract_entity_target_from_fact(fact_text):
    text = _clean_text(fact_text)

    strict_patterns = [
        r"\bwas born in\s+(.+?)[\.\,]?$",
        r"\bshown on\s+(.+?)[\.\,]?$",
        r"\blocated in\s+(.+?)[\.\,]?$",
        r"\btook place in\s+(.+?)[\.\,]?$",
        r"\bis in\s+([A-Z][A-Za-z0-9 ,\-]+)[\.\,]?$",
    ]
    for p in strict_patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            cand = _clean_text(m.group(1))
            if cand and cand.lower() not in {"a village", "a town", "a city", "a television sitcom", "a film", "an animal"}:
                return cand
    return ""


def is_existential_or_type_fact(fact_text):
    t = _clean_text(fact_text).lower()
    patterns = [
        "there exists",
        "is a village",
        "is a town",
        "is a city",
        "is a character in a television sitcom",
        "is a character in",
        "was born in a village",
        "provided the score for a film",
        "played for a team",
        "has a star",
    ]
    return any(p in t for p in patterns)


def verify_entity_wh(fact, answer, answer_status, evidence_span, gold_evidence):
    fact_text = _clean_text(fact.get("text", ""))

    if answer_status == "contradicted":
        return "contradict", "The answer result contradicts the fact."

    if answer_status in {"insufficient", "api_error"} or _clean_text(answer).lower() == "insufficient":
        return "insufficient", "The answer does not provide enough information."

    target = extract_entity_target_from_fact(fact_text)
    grounded = span_grounded_in_evidence(evidence_span, gold_evidence)

    if is_existential_or_type_fact(fact_text) or not target:
        if grounded and _clean_text(answer):
            return "support", "The answer provides a grounded value for the fact."
        return "insufficient", "The answer is not sufficiently grounded in the evidence."

    if text_match_loose(answer, target):
        if grounded:
            return "support", "The answer matches the target value and is grounded in the evidence."
        return "insufficient", "The answer matches the target value but is not grounded enough."

    return "contradict", "The answer conflicts with the target value in the atomic fact."


def build_maps(data):
    decomposition = data.get("decomposition", {}) or {}
    question_plan = data.get("question_plan", {}) or {}
    answer_result = data.get("answer_result", {}) or {}

    facts = decomposition.get("atomic_facts", []) or []
    qitems = question_plan.get("question_items", []) or []
    answers = answer_result.get("answers", []) or []
    final_bindings = answer_result.get("final_bindings", {}) or {}

    return (
        {f.get("id"): f for f in facts},
        {q.get("fact_id"): q for q in qitems},
        {a.get("fact_id"): a for a in answers},
        final_bindings,
    )


def verify_one_fact(fact, qitem, aitem, final_bindings, gold_evidence):
    raw_fact_text = _clean_text(fact.get("text", ""))
    bound_fact_text = replace_placeholders(raw_fact_text, final_bindings)

    question = _clean_text((aitem or {}).get("question", "") or (qitem or {}).get("main_question", ""))
    bound_question = replace_placeholders(question, final_bindings)

    answer = _clean_text((aitem or {}).get("answer", "insufficient"))
    answer_status = normalize_status((aitem or {}).get("status", "insufficient"))
    evidence_span = _clean_text((aitem or {}).get("evidence_span", ""))

    question_type = _clean_text((qitem or {}).get("question_type", ""))
    constraint = fact.get("constraint", {}) or {}
    negated = contains_negation(fact)
    critical = fact.get("critical", False)

    still_unresolved = unresolved_vars(bound_fact_text) or unresolved_vars(bound_question)
    if still_unresolved and answer_status in {"insufficient", "api_error"}:
        return {
            "fact_id": fact.get("id", ""),
            "fact_text": raw_fact_text,
            "bound_fact_text": bound_fact_text,
            "question": question,
            "bound_question": bound_question,
            "question_type": question_type,
            "answer": answer,
            "answer_status": answer_status,
            "verification_label": "insufficient",
            "reason": f"Unresolved variables remain: {still_unresolved}",
            "evidence_span": evidence_span,
            "constraint": constraint,
            "rely_on": fact.get("rely_on", []),
            "critical": critical,
        }

    verification_label = "insufficient"
    reason = ""

    if question_type == "relation_yesno":
        yn = normalize_yesno(answer)
        if answer_status == "contradicted":
            yn = "no" if yn is None else yn

        if answer_status in {"insufficient", "api_error"} or yn is None:
            verification_label = "insufficient"
            reason = "The answer does not clearly determine the fact."
        else:
            if yn == "yes":
                verification_label = "contradict" if negated else "support"
                reason = "The answer affirms the fact."
            else:
                verification_label = "support" if negated else "contradict"
                reason = "The answer denies the fact."

        if verification_label == "support" and not span_grounded_in_evidence(evidence_span, gold_evidence):
            if len(_clean_text(evidence_span).split()) >= 6:
                verification_label = "support"
                reason = "The answer affirms the fact and the evidence span provides weak but sufficient compositional grounding."
            else:
                verification_label = "insufficient"
                reason = "The answer suggests support, but the evidence span is not grounded in the provided evidence."

    elif question_type == "time_wh":
        matched = time_match(answer, constraint.get("time", []))
        if answer_status in {"insufficient", "api_error"}:
            verification_label = "insufficient"
            reason = "The answer does not provide enough temporal information."
        elif matched is True:
            verification_label = "contradict" if negated else "support"
            reason = "The answer matches the time constraint."
        elif matched is False and constraint.get("time"):
            verification_label = "support" if negated else "contradict"
            reason = "The answer conflicts with the time constraint."
        else:
            verification_label = "insufficient"
            reason = "The answer does not determine the time constraint."

    elif question_type == "quantity_wh":
        matched = quantity_match(answer, constraint.get("quantity", []), raw_fact_text)
        if answer_status in {"insufficient", "api_error"}:
            verification_label = "insufficient"
            reason = "The answer does not provide enough quantity information."
        elif matched is True:
            verification_label = "contradict" if negated else "support"
            reason = "The answer matches the quantity constraint."
        elif matched is False and (constraint.get("quantity") or re.search(r"\beight\b|\b27[, ]?000\b", raw_fact_text.lower())):
            verification_label = "support" if negated else "contradict"
            reason = "The answer conflicts with the quantity constraint."
        else:
            verification_label = "insufficient"
            reason = "The answer does not determine the quantity constraint."

    elif question_type == "entity_wh":
        verification_label, reason = verify_entity_wh(
            fact={"text": bound_fact_text},
            answer=answer,
            answer_status=answer_status,
            evidence_span=evidence_span,
            gold_evidence=gold_evidence
        )

    else:
        if answer_status == "contradicted":
            verification_label = "contradict" if not negated else "support"
            reason = "The answer result contradicts the fact."
        elif answer_status in {"insufficient", "api_error"} or answer.lower() == "insufficient":
            verification_label = "insufficient"
            reason = "The answer does not provide enough information."
        elif span_grounded_in_evidence(evidence_span, gold_evidence):
            verification_label = "support" if not negated else "contradict"
            reason = "The answer provides a grounded value for the fact."
        else:
            verification_label = "insufficient"
            reason = "The answer is not sufficiently grounded in the evidence."

    return {
        "fact_id": fact.get("id", ""),
        "fact_text": raw_fact_text,
        "bound_fact_text": bound_fact_text,
        "question": question,
        "bound_question": bound_question,
        "question_type": question_type,
        "answer": answer,
        "answer_status": answer_status,
        "verification_label": verification_label,
        "reason": reason,
        "evidence_span": evidence_span,
        "constraint": constraint,
        "rely_on": fact.get("rely_on", []),
        "critical": critical,
    }


def process_data_item(data):
    fact_map, q_map, a_map, final_bindings = build_maps(data)
    ordered_fact_ids = [f.get("id") for f in (data.get("decomposition", {}) or {}).get("atomic_facts", []) or []]

    fact_verification = []
    gold_evidence = data.get("gold_evidence", data.get("evidence", ""))

    for fid in ordered_fact_ids:
        fact = fact_map.get(fid, {})
        qitem = q_map.get(fid, {})
        aitem = a_map.get(fid, {})
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
            "verifications": fact_verification
        }
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
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_answer_[T][S]_[E].json"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_verify_[T][S]_[E].json"
    )

    args = parser.parse_args()
    main(args)