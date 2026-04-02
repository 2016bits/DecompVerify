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
    neg = constraint.get("negation", None)
    return neg is True


def extract_years(text):
    text = _clean_text(text)
    return re.findall(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", text)


def normalize_num_string(s):
    s = _clean_text(s).lower()
    s = s.replace(",", "")
    s = s.replace(" spectators", "")
    s = s.replace(" people", "")
    s = s.replace(" students", "")
    return s.strip()


def quantity_match(answer, quantities):
    ans = normalize_num_string(answer)
    qvals = [normalize_num_string(x) for x in (quantities or []) if _clean_text(x)]
    if not qvals:
        return None

    for q in qvals:
        if q and q in ans:
            return True

    # 序数 / 排名类
    ordinal_map = {
        "first": ["1st", "first"],
        "second": ["2nd", "second"],
        "third": ["3rd", "third"],
        "fourth": ["4th", "fourth"],
        "fifth": ["5th", "fifth"],
        "sixth": ["6th", "sixth"],
        "seventh": ["7th", "seventh"],
        "eighth": ["8th", "eighth"],
        "ninth": ["9th", "ninth"],
        "tenth": ["10th", "tenth"],
    }

    for q in qvals:
        for key, variants in ordinal_map.items():
            if key in q:
                if any(v in ans for v in variants):
                    return True

    return False


def time_match(answer, times):
    ans = _clean_text(answer).lower()
    tvals = [_clean_text(x).lower() for x in (times or []) if _clean_text(x)]
    if not tvals:
        return None

    # 先直接字符串包含
    for t in tvals:
        if t and (t in ans or ans in t):
            return True

    # 再退化到年份比较
    ans_years = set(extract_years(ans))
    if ans_years:
        for t in tvals:
            t_years = set(extract_years(t))
            if t_years and ans_years == t_years:
                return True

    return False


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


def verify_one_fact(fact, qitem, aitem, final_bindings):
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

    still_unresolved = unresolved_vars(bound_fact_text) or unresolved_vars(bound_question)

    # 默认结果
    verification_label = "insufficient"
    reason = ""

    # 1) 未解析变量，优先降为 insufficient
    if still_unresolved:
        verification_label = "insufficient"
        reason = f"Unresolved variables remain: {still_unresolved}"

    # 2) yes/no 类型
    elif question_type == "relation_yesno":
        yn = normalize_yesno(answer)

        if answer_status == "insufficient" or yn is None:
            verification_label = "insufficient"
            reason = "The answer does not clearly determine the fact."
        else:
            if yn == "yes":
                verification_label = "contradict" if negated else "support"
                reason = "The answer affirms the fact."
            elif yn == "no":
                verification_label = "support" if negated else "contradict"
                reason = "The answer denies the fact."

        if not evidence_span and verification_label == "support":
            verification_label = "insufficient"
            reason = "The answer suggests support, but no grounded evidence span is provided."

    # 3) 时间类
    elif question_type == "time_wh":
        matched = time_match(answer, constraint.get("time", []))
        if answer_status == "insufficient":
            verification_label = "insufficient"
            reason = "The answer does not provide enough temporal information."
        elif matched is True:
            verification_label = "contradict" if negated else "support"
            reason = "The answer matches the time constraint."
        elif matched is False and constraint.get("time"):
            verification_label = "support" if negated else "contradict"
            reason = "The answer conflicts with the time constraint."
        else:
            verification_label = "support" if answer_status == "supported" else "insufficient"
            reason = "The answer provides time information but no explicit time constraint is available."

    # 4) 数值类
    elif question_type == "quantity_wh":
        matched = quantity_match(answer, constraint.get("quantity", []))
        if answer_status == "insufficient":
            verification_label = "insufficient"
            reason = "The answer does not provide enough quantity information."
        elif matched is True:
            verification_label = "contradict" if negated else "support"
            reason = "The answer matches the quantity constraint."
        elif matched is False and constraint.get("quantity"):
            verification_label = "support" if negated else "contradict"
            reason = "The answer conflicts with the quantity constraint."
        else:
            verification_label = "support" if answer_status == "supported" else "insufficient"
            reason = "The answer provides quantity information but no explicit quantity constraint is available."

    # 5) 实体/描述类
    else:
        if answer_status == "insufficient" or answer.lower() == "insufficient":
            verification_label = "insufficient"
            reason = "The answer does not provide enough information."
        elif answer_status == "contradicted":
            verification_label = "contradict" if not negated else "support"
            reason = "The answer result contradicts the fact."
        else:
            verification_label = "support" if not negated else "contradict"
            reason = "The answer provides a grounded value for the fact."

        if not evidence_span and verification_label == "support":
            verification_label = "insufficient"
            reason = "The answer suggests support, but no grounded evidence span is provided."

    return {
        "fact_id": fact.get("id", ""),
        "fact_text": raw_fact_text,
        "bound_fact_text": bound_fact_text,
        "question": question,
        "bound_question": bound_question,
        "question_type": question_type,
        "answer": answer,
        "answer_status": answer_status,
        "verification_label": verification_label,   # support / contradict / insufficient
        "reason": reason,
        "evidence_span": evidence_span,
        "constraint": constraint,
        "rely_on": fact.get("rely_on", []),
    }


def process_data_item(data):
    fact_map, q_map, a_map, final_bindings = build_maps(data)

    ordered_fact_ids = [f.get("id") for f in (data.get("decomposition", {}) or {}).get("atomic_facts", []) or []]
    fact_verification = []

    for fid in ordered_fact_ids:
        fact = fact_map.get(fid, {})
        qitem = q_map.get(fid, {})
        aitem = a_map.get(fid, {})
        res = verify_one_fact(fact, qitem, aitem, final_bindings)
        fact_verification.append(res)

    return {
        "id": data["id"],
        "claim": data["claim"],
        "gold_evidence": data["gold_evidence"],
        "num_hops": data.get("num_hops", None),
        "label": data.get("label", None),
        "decomposition": data.get("decomposition", {}),
        "question_plan": data.get("question_plan", {}),
        "answer_result": data.get("answer_result", {}),
        "fact_verification": {
            "claim": data["claim"],
            "final_bindings": (data.get("answer_result", {}) or {}).get("final_bindings", {}),
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
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_answer_[T][S]_[E].json"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_verify_[T][S]_[E].json"
    )

    args = parser.parse_args()
    main(args)