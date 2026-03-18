import json
import argparse
import os
import re
from datetime import datetime


MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}

ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
    "twenty-first": 21, "twenty-second": 22, "twenty-third": 23, "twenty-fourth": 24, "twenty-fifth": 25,
    "twenty-sixth": 26, "twenty-seventh": 27, "twenty-eighth": 28, "twenty-ninth": 29, "thirtieth": 30,
    "thirty-first": 31, "fortieth": 40, "fiftieth": 50, "sixtieth": 60, "seventieth": 70,
    "eightieth": 80, "ninetieth": 90, "hundredth": 100
}

TRUE_WORDS = {"yes", "true", "correct", "supported"}
FALSE_WORDS = {"no", "false", "incorrect", "contradicted"}


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_label_3way(label):
    label = normalize_text(label).lower()
    if label in {"supports", "support", "supported"}:
        return "supports"
    if label in {"refutes", "refute", "refuted", "contradict", "contradicted"}:
        return "refutes"
    return "not enough info"


def safe_float(text):
    try:
        s = str(text).replace(",", "").strip()
        s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
        return float(s)
    except Exception:
        return None


def extract_year(text):
    if text is None:
        return None
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", str(text))
    if m:
        return int(m.group(1))
    return None


def ordinal_to_number(text):
    if text is None:
        return None
    s = normalize_text(text).lower()

    m = re.search(r"\b(\d+)(st|nd|rd|th)\b", s)
    if m:
        return int(m.group(1))

    for k, v in sorted(ORDINAL_WORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"\b{re.escape(k)}\b", s):
            return v
    return None


def normalize_bool(text):
    if text is None:
        return None
    s = normalize_text(text).lower()
    if s in TRUE_WORDS:
        return True
    if s in FALSE_WORDS:
        return False
    return None


def parse_date_like(text):
    """
    尽量把 answer 解析成可比较的时间值。
    优先返回：
    - 完整 datetime
    - 年份 int
    """
    if text is None:
        return None

    s = normalize_text(text).lower()

    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass

    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),\s*(\d{4})\b",
        s
    )
    if m:
        month = MONTHS[m.group(1)]
        day = int(m.group(2))
        year = int(m.group(3))
        try:
            return datetime(year, month, day)
        except Exception:
            pass

    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b",
        s
    )
    if m:
        month = MONTHS[m.group(1)]
        year = int(m.group(2))
        try:
            return datetime(year, month, 1)
        except Exception:
            pass

    year = extract_year(s)
    if year is not None:
        return year

    return None


def clean_entity_text(text):
    s = normalize_text(text).lower()
    s = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"[.,;:!?(){}\[\]]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_likely_truncated_or_noisy_value(value):
    s = normalize_text(value).lower()
    if not s:
        return True
    if re.fullmatch(r"\d", s):
        return True
    if re.fullmatch(r"[\W_]+", s):
        return True
    if len(s) <= 2 and not re.fullmatch(r"\d{4}", s):
        return True
    return False


def canonicalize_value(text):
    """
    返回 typed value:
    {
        "type": "number"/"ordinal"/"date"/"year"/"bool"/"text",
        "value": ...
    }
    """
    if text is None:
        return {"type": "text", "value": ""}

    b = normalize_bool(text)
    if b is not None:
        return {"type": "bool", "value": b}

    d = parse_date_like(text)
    if isinstance(d, datetime):
        return {"type": "date", "value": d}
    if isinstance(d, int):
        return {"type": "year", "value": d}

    f = safe_float(text)
    if f is not None:
        return {"type": "number", "value": f}

    o = ordinal_to_number(text)
    if o is not None:
        return {"type": "ordinal", "value": float(o)}

    return {"type": "text", "value": clean_entity_text(text)}


def compare_typed_values(left, right, operator):
    op = normalize_text(operator).lower()
    numeric_types = {"number", "ordinal"}

    if left["type"] in numeric_types and right["type"] in numeric_types:
        lv = float(left["value"])
        rv = float(right["value"])
        if op in {"=", "==", "same_as"}:
            return lv == rv
        if op in {"!=", "different_from"}:
            return lv != rv
        if op == ">":
            return lv > rv
        if op == "<":
            return lv < rv
        if op == ">=":
            return lv >= rv
        if op == "<=":
            return lv <= rv
        if op in {"before", "earlier_than"}:
            return lv < rv
        if op in {"after", "later_than"}:
            return lv > rv
        return None

    if left["type"] in {"date", "year"} and right["type"] in {"date", "year"}:
        lv = left["value"]
        rv = right["value"]

        if isinstance(lv, int) and isinstance(rv, datetime):
            lv = datetime(lv, 1, 1)
        if isinstance(lv, datetime) and isinstance(rv, int):
            rv = datetime(rv, 1, 1)

        if isinstance(lv, int) and isinstance(rv, int):
            if op in {"=", "==", "same_as"}:
                return lv == rv
            if op in {"!=", "different_from"}:
                return lv != rv
            if op == ">":
                return lv > rv
            if op == "<":
                return lv < rv
            if op == ">=":
                return lv >= rv
            if op == "<=":
                return lv <= rv
            if op in {"before", "earlier_than"}:
                return lv < rv
            if op in {"after", "later_than"}:
                return lv > rv
            return None

        if isinstance(lv, datetime) and isinstance(rv, datetime):
            if op in {"=", "==", "same_as"}:
                return lv == rv
            if op in {"!=", "different_from"}:
                return lv != rv
            if op == ">":
                return lv > rv
            if op == "<":
                return lv < rv
            if op == ">=":
                return lv >= rv
            if op == "<=":
                return lv <= rv
            if op in {"before", "earlier_than"}:
                return lv < rv
            if op in {"after", "later_than"}:
                return lv > rv
            return None

    if left["type"] == "bool" and right["type"] == "bool":
        if op in {"=", "==", "same_as"}:
            return left["value"] == right["value"]
        if op in {"!=", "different_from"}:
            return left["value"] != right["value"]
        return None

    if left["type"] == "text" and right["type"] == "text":
        lv = left["value"]
        rv = right["value"]
        if op in {"=", "==", "same_as"}:
            return lv == rv
        if op in {"!=", "different_from"}:
            return lv != rv
        if op == "contains":
            return rv in lv
        if op == "contained_by":
            return lv in rv
        return None

    lf = safe_float(left["value"]) if left["type"] == "text" else None
    rf = safe_float(right["value"]) if right["type"] == "text" else None

    if left["type"] in numeric_types and rf is not None:
        return compare_typed_values(left, {"type": "number", "value": rf}, op)
    if right["type"] in numeric_types and lf is not None:
        return compare_typed_values({"type": "number", "value": lf}, right, op)

    return None


def build_fact_maps(answer_plan, fact_verification):
    answer_map = {item["fact_id"]: item for item in answer_plan}
    verify_map = {item["fact_id"]: item for item in fact_verification}
    return answer_map, verify_map


def collect_global_bindings(answer_plan, final_bindings=None):
    bindings = {}
    for item in answer_plan:
        upd = item.get("bindings_update", {}) or {}
        for k, v in upd.items():
            bindings[k] = v
    if final_bindings:
        for k, v in final_bindings.items():
            bindings[k] = v
    return bindings


def answer_is_effectively_resolved(answer_item, global_bindings):
    status = normalize_text(answer_item.get("status", "")).lower()
    if status not in {"insufficient", "unknown", ""}:
        return True

    filled_fact = answer_item.get("filled_fact", {}) or {}
    subject = normalize_text(filled_fact.get("subject", ""))
    obj = normalize_text(filled_fact.get("object", ""))

    vars_in_fact = re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", subject + " " + obj)
    if vars_in_fact and all(v in global_bindings for v in vars_in_fact):
        ans = normalize_text(answer_item.get("answer", ""))
        if ans and ans.lower() != "insufficient":
            return True
    return False

def get_fact_role_weight(fact_item):
    role = normalize_text(fact_item.get("role", "verify")).lower()
    if role == "verify":
        return 1.0
    if role == "bridge":
        return 0.6
    if role == "anchor":
        return 0.45
    return 0.7


def get_fact_source_weight(fact_item):
    source = normalize_text(fact_item.get("source", "verified")).lower()
    if source == "verified":
        return 1.0
    if source == "answer_fallback":
        return 0.7
    if source == "parse_fallback":
        return 0.4
    if source == "timeout_fallback":
        return 0.3
    return 0.8


def get_fact_confidence(fact_item):
    conf = fact_item.get("confidence", None)
    try:
        conf = float(conf)
        return max(0.0, min(1.0, conf))
    except Exception:
        return 1.0


def get_fact_importance(fact_item):
    role = normalize_text(fact_item.get("role", "verify")).lower()
    predicate = normalize_text(fact_item.get("predicate", "")).lower()
    obj = normalize_text(fact_item.get("object", "")).strip()

    if role in {"bridge", "anchor"}:
        return "optional"
    if predicate in {"named", "called", "titled", "aka", "also known as"}:
        return "optional"
    if "?" in obj:
        return "optional"
    return "critical"


def get_effective_verification_label(fact_item, answer_item, global_bindings):
    label = normalize_text(fact_item.get("verification_label", "")).lower()
    role = normalize_text(fact_item.get("role", "verify")).lower()

    if label in {"support", "contradict"}:
        return label

    if role in {"bridge", "anchor"} and answer_item is not None:
        if answer_is_effectively_resolved(answer_item, global_bindings):
            return "support"

    return "insufficient"


def score_fact(fact_item, answer_item=None, global_bindings=None):
    """
    返回:
    {
        "label_3way": "...",
        "effective_label": "...",
        "score": float,
        "hard_contradict": bool,
        "importance": "...",
        "trace": {...}
    }
    """
    if global_bindings is None:
        global_bindings = {}

    raw_label = fact_item.get("verification_label", "")
    effective_label = get_effective_verification_label(fact_item, answer_item, global_bindings)

    role_w = get_fact_role_weight(fact_item)
    src_w = get_fact_source_weight(fact_item)
    conf_w = get_fact_confidence(fact_item)
    weight = role_w * src_w * conf_w

    role = normalize_text(fact_item.get("role", "verify")).lower()
    source = normalize_text(fact_item.get("source", "verified")).lower()
    importance = get_fact_importance(fact_item)

    if effective_label == "support":
        base = 0.9 if role == "verify" else 0.35
        return {
            "label_3way": "supports",
            "effective_label": effective_label,
            "score": base * weight,
            "hard_contradict": False,
            "importance": importance,
            "trace": {
                "raw_label": raw_label,
                "role_weight": role_w,
                "source_weight": src_w,
                "confidence": conf_w,
                "final_weight": weight
            }
        }

    if effective_label == "contradict":
        base = -1.35 if role == "verify" else -0.45
        hard_contradict = (
            source in {"verified", "answer_fallback"}
            and conf_w >= 0.6
            and role == "verify"
            and importance == "critical"
        )
        return {
            "label_3way": "refutes",
            "effective_label": effective_label,
            "score": base * weight,
            "hard_contradict": hard_contradict,
            "importance": importance,
            "trace": {
                "raw_label": raw_label,
                "role_weight": role_w,
                "source_weight": src_w,
                "confidence": conf_w,
                "final_weight": weight
            }
        }

    base = -0.45 if (role == "verify" and importance == "critical") else -0.06
    return {
        "label_3way": "not enough info",
        "effective_label": effective_label,
        "score": base * weight,
        "hard_contradict": False,
        "importance": importance,
        "trace": {
            "raw_label": raw_label,
            "role_weight": role_w,
            "source_weight": src_w,
            "confidence": conf_w,
            "final_weight": weight
        }
    }


def evaluate_single_constraint(constraint, answer_map, verify_map):
    """
    返回:
    {
        "result": "support"/"contradict"/"insufficient",
        "confidence": "high"/"low",
        "reason": str
    }
    """
    ctype = normalize_text(constraint.get("type", "")).lower()
    target_facts = constraint.get("target_facts", []) or []
    operator = normalize_text(constraint.get("operator", ""))
    value = constraint.get("value", "")

    if not target_facts:
        return {
            "result": "insufficient",
            "confidence": "low",
            "reason": "No target facts."
        }

    for fid in target_facts:
        if fid not in verify_map or fid not in answer_map:
            return {
                "result": "insufficient",
                "confidence": "low",
                "reason": f"Missing fact {fid}."
            }

    for fid in target_facts:
        vlabel = normalize_text(verify_map[fid].get("verification_label", "")).lower()
        if vlabel == "insufficient":
            return {
                "result": "insufficient",
                "confidence": "low",
                "reason": f"Target fact {fid} is insufficient."
            }
        if vlabel == "contradict":
            return {
                "result": "contradict",
                "confidence": "high",
                "reason": f"Target fact {fid} is contradicted."
            }

    if ctype == "negation":
        if len(target_facts) != 1:
            return {
                "result": "insufficient",
                "confidence": "low",
                "reason": "Negation expects exactly 1 target fact."
            }
        fid = target_facts[0]
        base = normalize_text(verify_map[fid].get("verification_label", "")).lower()
        if base == "support":
            return {
                "result": "contradict",
                "confidence": "high",
                "reason": "Negation over a supported fact."
            }
        if base == "contradict":
            return {
                "result": "support",
                "confidence": "high",
                "reason": "Negation over a contradicted fact."
            }
        return {
            "result": "insufficient",
            "confidence": "low",
            "reason": "Negation target fact is insufficient."
        }

    answers = [answer_map[fid].get("answer", "") for fid in target_facts]

    if ctype == "comparison":
        if len(answers) != 2:
            return {
                "result": "insufficient",
                "confidence": "low",
                "reason": "Comparison expects exactly 2 answers."
            }

        left = canonicalize_value(answers[0])
        right = canonicalize_value(answers[1])

        ok = compare_typed_values(left, right, operator)
        if ok is None:
            return {
                "result": "insufficient",
                "confidence": "low",
                "reason": f"Cannot compare {left} and {right}."
            }
        return {
            "result": "support" if ok else "contradict",
            "confidence": "high",
            "reason": f"Comparison evaluated on {left} {operator} {right}."
        }

    if ctype in {"time", "quantity"}:
        if len(answers) != 1:
            return {
                "result": "insufficient",
                "confidence": "low",
                "reason": f"{ctype} expects exactly 1 answer."
            }

        if is_likely_truncated_or_noisy_value(value):
            return {
                "result": "insufficient",
                "confidence": "low",
                "reason": f"Constraint value looks noisy/truncated: {value}"
            }

        left = canonicalize_value(answers[0])
        right = canonicalize_value(value)

        ok = compare_typed_values(left, right, operator)
        if ok is None:
            return {
                "result": "insufficient",
                "confidence": "low",
                "reason": f"Cannot compare {left} and {right}."
            }
        return {
            "result": "support" if ok else "contradict",
            "confidence": "high",
            "reason": f"{ctype} constraint evaluated on {left} {operator} {right}."
        }

    if ctype in {"equality", "entity", "relation"}:
        if len(answers) != 1:
            return {
                "result": "insufficient",
                "confidence": "low",
                "reason": f"{ctype} expects exactly 1 answer."
            }

        if is_likely_truncated_or_noisy_value(value):
            return {
                "result": "insufficient",
                "confidence": "low",
                "reason": f"Constraint value looks noisy/truncated: {value}"
            }

        left = canonicalize_value(answers[0])
        right = canonicalize_value(value)
        op = operator if operator else "="

        ok = compare_typed_values(left, right, op)
        if ok is None:
            return {
                "result": "insufficient",
                "confidence": "low",
                "reason": f"Cannot compare {left} and {right}."
            }
        return {
            "result": "support" if ok else "contradict",
            "confidence": "high",
            "reason": f"Fallback equality evaluated on {left} {op} {right}."
        }

    return {
        "result": "insufficient",
        "confidence": "low",
        "reason": f"Unknown constraint type: {ctype}"
    }


def score_constraint_result(result, constraint, confidence="high"):
    ctype = normalize_text(constraint.get("type", "")).lower()

    if ctype in {"comparison", "time", "quantity", "negation"}:
        support_w = 0.8
        refute_w = -1.3
    else:
        support_w = 0.6
        refute_w = -0.9

    if confidence == "low":
        support_w *= 0.6
        refute_w *= 0.45

    if result == "support":
        return support_w
    if result == "contradict":
        return refute_w
    return -0.18 if confidence == "high" else -0.08


def map_3way_to_2way(label, nei_to="refutes"):
    label = normalize_label_3way(label)
    if label == "supports":
        return "supports"
    if label == "refutes":
        return "refutes"
    return "supports" if nei_to == "supports" else "refutes"


def aggregate_labels(
    decomposition,
    fact_verification,
    answer_plan,
    final_bindings=None,
    class_num="2",
    support_threshold=1.3,
    refute_threshold=-0.9,
    nei_to="refutes"
):
    """
    新版聚合：
    1. 先做 fact critical/optional 区分
    2. constraint 带 confidence
    3. bridge/anchor 不再轻易 hard contradiction
    4. critical verify insufficient 不能轻易 support
    """
    constraints = decomposition.get("constraints", []) if decomposition else []
    answer_map, verify_map = build_fact_maps(answer_plan, fact_verification)
    global_bindings = collect_global_bindings(answer_plan, final_bindings)

    fact_details = []
    fact_score_total = 0.0
    hard_contradict = False
    num_support = 0
    num_refute = 0
    num_nei = 0

    critical_verify_support = 0
    critical_verify_refute = 0
    critical_verify_nei = 0
    optional_support = 0
    optional_refute = 0
    optional_nei = 0

    for item in fact_verification:
        fid = item.get("fact_id")
        answer_item = answer_map.get(fid)
        scored = score_fact(item, answer_item=answer_item, global_bindings=global_bindings)
        fact_score_total += scored["score"]
        hard_contradict = hard_contradict or scored["hard_contradict"]

        if scored["label_3way"] == "supports":
            num_support += 1
        elif scored["label_3way"] == "refutes":
            num_refute += 1
        else:
            num_nei += 1

        role = normalize_text(item.get("role", "verify")).lower()
        importance = scored["importance"]
        effective_label = scored["effective_label"]

        if role == "verify" and importance == "critical":
            if effective_label == "support":
                critical_verify_support += 1
            elif effective_label == "contradict":
                critical_verify_refute += 1
            else:
                critical_verify_nei += 1
        else:
            if effective_label == "support":
                optional_support += 1
            elif effective_label == "contradict":
                optional_refute += 1
            else:
                optional_nei += 1

        fact_details.append({
            "fact_id": item.get("fact_id"),
            "verification_label": item.get("verification_label"),
            "effective_label": effective_label,
            "importance": importance,
            "role": role,
            "score": scored["score"],
            "hard_contradict": scored["hard_contradict"],
            "trace": scored["trace"]
        })

    constraint_results = []
    constraint_score_total = 0.0
    constraint_refute_count = 0
    constraint_support_count = 0
    constraint_nei_count = 0
    high_conf_constraint_refute_count = 0

    for constraint in constraints:
        cres = evaluate_single_constraint(constraint, answer_map, verify_map)
        cscore = score_constraint_result(
            cres["result"],
            constraint,
            confidence=cres.get("confidence", "high")
        )
        constraint_score_total += cscore

        if cres["result"] == "support":
            constraint_support_count += 1
        elif cres["result"] == "contradict":
            constraint_refute_count += 1
            if cres.get("confidence", "high") == "high":
                high_conf_constraint_refute_count += 1
        else:
            constraint_nei_count += 1

        constraint_results.append({
            "constraint": constraint,
            "result": cres["result"],
            "confidence": cres.get("confidence", "high"),
            "reason": cres.get("reason", ""),
            "score": cscore
        })

    total_score = fact_score_total + constraint_score_total

    if hard_contradict:
        pred_3way = "refutes"
        decision = "Hard contradiction from critical verify fact."
    elif critical_verify_refute > 0:
        pred_3way = "refutes"
        decision = "At least one critical verify fact is contradicted."
    elif high_conf_constraint_refute_count > 0:
        pred_3way = "refutes"
        decision = "At least one high-confidence constraint is contradicted."
    elif total_score <= refute_threshold:
        pred_3way = "refutes"
        decision = "Total score falls below refute threshold."
    elif critical_verify_nei > 0:
        pred_3way = "not enough info"
        decision = "Critical verify facts remain insufficient."
    elif constraint_nei_count > 0 and constraint_support_count == 0:
        pred_3way = "not enough info"
        decision = "Constraints remain unresolved."
    elif (
        total_score >= support_threshold
        and critical_verify_refute == 0
        and high_conf_constraint_refute_count == 0
        and critical_verify_support >= 1
    ):
        pred_3way = "supports"
        decision = "Critical verify facts are supported and no high-confidence contradiction exists."
    else:
        pred_3way = "not enough info"
        decision = "Evidence is mixed or insufficient under current thresholds."

    pred_label = (
        map_3way_to_2way(pred_3way, nei_to=nei_to)
        if str(class_num) == "2"
        else pred_3way
    )

    return {
        "predicted_label_3way": pred_3way,
        "predicted_label": pred_label,
        "aggregation_trace": {
            "fact_details": fact_details,
            "constraint_results": constraint_results,
            "fact_score_total": fact_score_total,
            "constraint_score_total": constraint_score_total,
            "total_score": total_score,
            "num_support_facts": num_support,
            "num_refute_facts": num_refute,
            "num_nei_facts": num_nei,
            "num_support_constraints": constraint_support_count,
            "num_refute_constraints": constraint_refute_count,
            "num_high_conf_refute_constraints": high_conf_constraint_refute_count,
            "num_nei_constraints": constraint_nei_count,
            "hard_contradict": hard_contradict,
            "support_threshold": support_threshold,
            "refute_threshold": refute_threshold,
            "critical_verify_support_facts": critical_verify_support,
            "critical_verify_refute_facts": critical_verify_refute,
            "critical_verify_nei_facts": critical_verify_nei,
            "optional_support_facts": optional_support,
            "optional_refute_facts": optional_refute,
            "optional_nei_facts": optional_nei,
            "nei_to": nei_to,
            "decision": decision
        }
    }


def process_data_item(data, class_num, support_threshold=1.3, refute_threshold=-0.9, nei_to="refutes"):
    claim = data["claim"]
    evidence = data.get("gold_evidence", data.get("evidence", ""))
    decomposition = data.get("decomposition", data.get("atomic_facts", None))
    if decomposition is None:
        decomposition = {
            "claim": claim,
            "atomic_facts": [],
            "constraints": []
        }

    question_plan = data.get("question_plan", [])
    answer_plan = data.get("answer_plan", [])
    final_bindings = data.get("final_bindings", {})
    fact_verification = data.get("fact_verification", [])

    aggregation_result = aggregate_labels(
        decomposition=decomposition,
        fact_verification=fact_verification,
        answer_plan=answer_plan,
        final_bindings=final_bindings,
        class_num=class_num,
        support_threshold=support_threshold,
        refute_threshold=refute_threshold,
        nei_to=nei_to,
    )

    return {
        "id": data["id"],
        "claim": claim,
        "gold_evidence": evidence,
        "num_hops": data.get("num_hops", None),
        "label": data.get("label", None),
        "decomposition": decomposition,
        "question_plan": question_plan,
        "answer_plan": answer_plan,
        "final_bindings": final_bindings,
        "fact_verification": fact_verification,
        "predicted_label": aggregation_result["predicted_label"],
        "predicted_label_3way": aggregation_result["predicted_label_3way"],
        "aggregation_trace": aggregation_result["aggregation_trace"]
    }


def main(args):
    in_path = (
        args.in_path
        .replace("[DATA]", args.dataset)
        .replace("[TYPE]", args.data_type)
        .replace("[CLASS]", args.class_num)
        .replace("[T]", args.t)
        .replace("[S]", str(args.start))
        .replace("[E]", str(args.end))
        .replace("[PLAN]", args.plan)
    )

    with open(in_path, "r", encoding="utf-8") as f:
        raws = json.load(f)

    results = []
    for data in raws:
        try:
            res = process_data_item(
                data,
                class_num=args.class_num,
                support_threshold=args.support_threshold,
                refute_threshold=args.refute_threshold,
                nei_to=args.nei_to
            )
            results.append(res)
        except Exception as e:
            print(f"Error processing {data.get('id', 'unknown')}: {e}")
            continue

    results = sorted(results, key=lambda x: x["id"])

    out_path = (
        args.out_path
        .replace("[DATA]", args.dataset)
        .replace("[TYPE]", args.data_type)
        .replace("[CLASS]", args.class_num)
        .replace("[T]", args.t)
        .replace("[S]", str(args.start))
        .replace("[E]", str(args.end))
        .replace("[PLAN]", args.plan)
    )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HOVER', help='Dataset name')
    parser.add_argument('--data_type', type=str, default='dev', help='Data type: train/dev/test')
    parser.add_argument('--class_num', type=str, default='2', help='Number of classes: 2/3')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=200, help='End index')

    parser.add_argument(
        '--in_path',
        type=str,
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_fact_verify_[T][S]_[E].json',
        help='Input path template'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_final_[T][S]_[E].json',
        help='Output path template'
    )

    parser.add_argument('--plan', type=str, default='plan2.1', help='Which plan version to use for aggregation logic')
    parser.add_argument('--t', type=str, default='')
    parser.add_argument('--support_threshold', type=float, default=1.3)
    parser.add_argument('--refute_threshold', type=float, default=-0.9)
    parser.add_argument('--nei_to', type=str, default="refutes")

    args = parser.parse_args()
    main(args)