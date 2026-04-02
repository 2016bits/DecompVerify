import os
import re
import json
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")


def _clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def normalize_text_for_match(text):
    text = _clean_text(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return [x for x in normalize_text_for_match(text).split() if x]


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


def replace_placeholders_in_obj(obj, bindings):
    if isinstance(obj, str):
        return replace_placeholders(obj, bindings)
    if isinstance(obj, list):
        return [replace_placeholders_in_obj(x, bindings) for x in obj]
    if isinstance(obj, dict):
        return {k: replace_placeholders_in_obj(v, bindings) for k, v in obj.items()}
    return obj


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
    constraint = fact.get("constraint", {}) or {}
    return constraint.get("negation") is True


def extract_years(text):
    text = _clean_text(text)
    return re.findall(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", text)


def normalize_num_string(s):
    s = _clean_text(s).lower()
    s = s.replace(",", "")
    return s.strip()


def quantity_match(answer, quantities):
    ans = normalize_num_string(answer)
    qvals = [normalize_num_string(x) for x in (quantities or []) if _clean_text(x)]
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
    return span in gold


def text_match_loose(a, b):
    a_n = normalize_text_for_match(a)
    b_n = normalize_text_for_match(b)
    if not a_n or not b_n:
        return False
    if a_n == b_n:
        return True
    if a_n in b_n or b_n in a_n:
        return True
    return False


def text_clearly_different(a, b):
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens or not b_tokens:
        return False
    if text_match_loose(a, b):
        return False
    overlap = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    jaccard = overlap / union if union else 0.0
    return jaccard < 0.35


def extract_entity_target_from_fact(fact_text):
    text = _clean_text(fact_text)

    patterns = [
        r"\bwas born in\s+(.+?)[\.\,]?$",
        r"\bis born in\s+(.+?)[\.\,]?$",
        r"\bshown on\s+(.+?)[\.\,]?$",
        r"\bshown in\s+(.+?)[\.\,]?$",
        r"\blocated in\s+(.+?)[\.\,]?$",
        r"\btook place in\s+(.+?)[\.\,]?$",
        r"\bdied at a battle in\s+(.+?)[\.\,]?$",
        r"\bwas in\s+(.+?)[\.\,]?$",
        r"\bin\s+([A-Z][A-Za-z0-9 ,\-]+)[\.\,]?$",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            cand = _clean_text(m.group(1))
            bad = {
                "a football club", "that football club", "a battle", "the battle",
                "an award", "the award", "a director", "the director", "a series",
                "the series", "a person", "that person"
            }
            if cand.lower() not in bad:
                return cand
    return ""


def build_maps(data):
    decomposition = data.get("decomposition", {}) or {}
    question_plan = data.get("question_plan", {}) or {}
    answer_result = data.get("answer_result", {}) or {}

    facts = decomposition.get("atomic_facts", []) or []
    qitems = question_plan.get("question_items", []) or []
    answers = answer_result.get("answers", []) or []
    final_bindings = answer_result.get("final_bindings", {}) or {}

    fact_map = {f.get("id"): f for f in facts}
    q_map = {q.get("fact_id"): q for q in qitems}
    a_map = {a.get("fact_id"): a for a in answers}

    return fact_map, q_map, a_map, final_bindings


def verify_entity_wh_with_target(fact, answer, answer_status, evidence_span, gold_evidence):
    target = extract_entity_target_from_fact(fact.get("text", ""))

    if answer_status in {"insufficient", "api_error"} or _clean_text(answer).lower() == "insufficient":
        return "insufficient", "The answer does not provide enough information."

    if not target:
        # fallback: old behavior but require grounding
        if span_grounded_in_evidence(evidence_span, gold_evidence) and _clean_text(answer):
            return "support", "The answer provides a grounded value for the fact."
        return "insufficient", "The answer is not sufficiently grounded in the evidence."

    grounded = span_grounded_in_evidence(evidence_span, gold_evidence)
    target_supported_by_gold = text_match_loose(target, gold_evidence) or text_match_loose(answer, gold_evidence)
    target_supported_by_span = text_match_loose(target, evidence_span) or text_match_loose(answer, evidence_span)

    if text_match_loose(answer, target):
        if grounded and (target_supported_by_gold or target_supported_by_span):
            return "support", "The answer matches the target value and is grounded in the evidence."
        return "insufficient", "The answer matches the target value but the evidence span is not grounded enough."

    if text_clearly_different(answer, target):
        return "contradict", "The answer conflicts with the target value in the atomic fact."

    return "insufficient", "The answer cannot be confidently aligned with the target value."


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

    verification_label = "insufficient"
    reason = ""

    if still_unresolved:
        verification_label = "insufficient"
        reason = f"Unresolved variables remain: {still_unresolved}"

    elif question_type == "relation_yesno":
        yn = normalize_yesno(answer)

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
        matched = quantity_match(answer, constraint.get("quantity", []))
        if answer_status in {"insufficient", "api_error"}:
            verification_label = "insufficient"
            reason = "The answer does not provide enough quantity information."
        elif matched is True:
            verification_label = "contradict" if negated else "support"
            reason = "The answer matches the quantity constraint."
        elif matched is False and constraint.get("quantity"):
            verification_label = "support" if negated else "contradict"
            reason = "The answer conflicts with the quantity constraint."
        else:
            verification_label = "insufficient"
            reason = "The answer does not determine the quantity constraint."

    elif question_type == "entity_wh":
        verification_label, reason = verify_entity_wh_with_target(
            fact={"text": bound_fact_text},
            answer=answer,
            answer_status=answer_status,
            evidence_span=evidence_span,
            gold_evidence=gold_evidence
        )

    else:
        if answer_status in {"insufficient", "api_error"} or answer.lower() == "insufficient":
            verification_label = "insufficient"
            reason = "The answer does not provide enough information."
        elif answer_status == "contradicted":
            verification_label = "contradict" if not negated else "support"
            reason = "The answer result contradicts the fact."
        else:
            if span_grounded_in_evidence(evidence_span, gold_evidence):
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
        res = verify_one_fact(fact, qitem, aitem, final_bindings, gold_evidence)
        fact_verification.append(res)

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