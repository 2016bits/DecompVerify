import os
import re
import json
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from datetime import datetime


VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")

NEGATION_CUE_PATTERNS = [
    r"\bnot\b",
    r"\bno\b",
    r"\bno longer\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bfail(?:ed|s|ing)?\b",
    r"\bunsuccessful(?:ly)?\b",
    r"\bunable\b",
    r"\bunaware\b",
    r"\black(?:s|ed|ing)?\b",
    r"\babsence\b",
]

MONTH_TO_NUM = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

STOPWORD_TOKENS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from", "by", "with",
}

GENERIC_LOCATION_TOKENS = {
    "country", "region", "area", "province", "state", "city", "town",
    "village", "county", "district", "place",
}

DIRECTION_TOKENS = {
    "north", "south", "east", "west",
    "northern", "southern", "eastern", "western",
    "northeast", "northwest", "southeast", "southwest",
    "northeastern", "northwestern", "southeastern", "southwestern",
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

SCALE_WORDS = {
    "hundred": 100,
    "thousand": 1000,
}

ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}



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



def normalize_semantic_text(text):
    text = normalize_text_for_match(text)
    text = text.replace("all year round", "year round")
    text = text.replace("all year-round", "year round")
    text = text.replace("all-year round", "year round")
    text = text.replace("all-year-round", "year round")
    text = text.replace("year-round", "year round")
    return text



def contains_explicit_negation(text):
    text_l = _clean_text(text).lower()
    if not text_l:
        return False
    return any(re.search(pattern, text_l) for pattern in NEGATION_CUE_PATTERNS)



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
    return re.findall(r"\b([1-9][0-9]{2,3})\b", _clean_text(text))



def extract_dates(text):
    month_regex = "|".join(MONTH_TO_NUM.keys())
    text = _clean_text(text)
    patterns = [
        re.compile(rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{month_regex})\s+(?P<year>\d{{3,4}})\b", re.IGNORECASE),
        re.compile(rf"\b(?P<month>{month_regex})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*|\s+)(?P<year>\d{{3,4}})\b", re.IGNORECASE),
    ]

    dates = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            month = match.group("month").lower()
            dates.append((int(match.group("year")), MONTH_TO_NUM[month], int(match.group("day"))))
    return dates



def extract_month_numbers(text):
    text_l = _clean_text(text).lower()
    return sorted({num for month, num in MONTH_TO_NUM.items() if re.search(rf"\b{month}\b", text_l)})



def is_year_only_text(text):
    text_l = _clean_text(text).lower()
    text_l = re.sub(r"\b[1-9][0-9]{2,3}\b", " ", text_l)
    text_l = re.sub(r"\b(ad|ce|bc|bce)\b", " ", text_l)
    text_l = re.sub(r"[^a-z]", " ", text_l)
    return not re.sub(r"\s+", "", text_l)



def extract_decade_range(text):
    text_l = _clean_text(text).lower()
    match = re.search(r"\b(?:(early|mid|late)\s+)?([1-9][0-9]{2,3})s\b", text_l)
    if not match:
        return None

    modifier = match.group(1)
    start = int(match.group(2))
    span = 100 if start % 100 == 0 else 10
    end = start + span - 1

    if modifier == "early":
        end = start + (39 if span == 100 else 3)
    elif modifier == "mid":
        start = start + (30 if span == 100 else 3)
        end = start + (39 if span == 100 else 3)
    elif modifier == "late":
        start = start + (70 if span == 100 else 7)

    return start, end



def time_match(answer, times):
    ans = normalize_semantic_text(answer)
    tvals = [_clean_text(x) for x in (times or []) if _clean_text(x)]
    if not tvals:
        return None

    answer_dates = extract_dates(answer)
    answer_months = extract_month_numbers(answer)
    answer_years = [int(x) for x in extract_years(answer)]
    saw_conflict = False

    for t in tvals:
        t_norm = normalize_semantic_text(t)
        if t_norm and (t_norm in ans or ans in t_norm):
            return True

        t_dates = extract_dates(t)
        if t_dates:
            if answer_dates:
                if any(ad == td for ad in answer_dates for td in t_dates):
                    return True
                saw_conflict = True
            continue

        t_months = extract_month_numbers(t)
        t_years = [int(x) for x in extract_years(t)]

        if t_months and t_years:
            if answer_months and answer_years:
                if any(m in t_months for m in answer_months) and any(y in t_years for y in answer_years):
                    return True
                saw_conflict = True
            continue

        decade_range = extract_decade_range(t)
        if decade_range and answer_years:
            start, end = decade_range
            if any(start <= year <= end for year in answer_years):
                return True
            saw_conflict = True
            continue

        if t_years and answer_years:
            if is_year_only_text(t):
                if any(year in t_years for year in answer_years):
                    return True
                saw_conflict = True
                continue

            if set(answer_years) == set(t_years):
                return True
            saw_conflict = True

    return False if saw_conflict else None



def normalize_num_string(s):
    return _clean_text(s).lower().replace(",", "").replace("-", " ").strip()



def tokenize_number_words(text):
    text = normalize_num_string(text)
    text = re.sub(r"[^\w\s]", " ", text)
    return [tok for tok in text.split() if tok]



def consume_number_tokens(tokens, start):
    total = 0
    current = 0
    used = 0
    found = False

    while start + used < len(tokens):
        tok = tokens[start + used]
        if tok in NUMBER_WORDS:
            current += NUMBER_WORDS[tok]
            found = True
        elif tok in SCALE_WORDS:
            current = max(1, current) * SCALE_WORDS[tok]
            if SCALE_WORDS[tok] >= 1000:
                total += current
                current = 0
            found = True
        elif tok in ORDINAL_WORDS:
            current += ORDINAL_WORDS[tok]
            found = True
            used += 1
            break
        elif tok == "and" and found:
            used += 1
            continue
        else:
            break
        used += 1

    if not found:
        return None, 0
    return total + current, used



def extract_numeric_values(text):
    text = normalize_num_string(text)
    values = []

    for match in re.findall(r"\b(\d+)(?:st|nd|rd|th)?\b", text.lower()):
        values.append(int(match))

    tokens = tokenize_number_words(text)
    idx = 0
    while idx < len(tokens):
        value, used = consume_number_tokens(tokens, idx)
        if used:
            values.append(value)
            idx += used
        else:
            idx += 1

    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped



def parse_quantity_comparator(text):
    text = normalize_num_string(text)
    number_token = r"(?:\d+(?:st|nd|rd|th)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)"
    number_phrase = rf"(?P<num>{number_token}(?:\s+(?:and\s+)?{number_token}){{0,4}})"
    patterns = [
        (rf"\b(?:more than|over|greater than|above)\s+{number_phrase}\b", ">"),
        (rf"\b(?:at least|no fewer than|minimum of)\s+{number_phrase}\b", ">="),
        (rf"\b(?:at most|no more than|up to|maximum of)\s+{number_phrase}\b", "<="),
        (rf"\b(?:less than|under|below|fewer than)\s+{number_phrase}\b", "<"),
    ]
    for pattern, op in patterns:
        match = re.search(pattern, text)
        if match:
            nums = extract_numeric_values(match.group("num"))
            if nums:
                return op, nums[0]
    return None



def compare_numeric(value, op, target):
    if op == ">":
        return value > target
    if op == ">=":
        return value >= target
    if op == "<":
        return value < target
    if op == "<=":
        return value <= target
    return value == target



def infer_quantity_targets_from_fact(fact_text):
    fact_text = _clean_text(fact_text)
    fact_l = fact_text.lower()
    if re.search(
        r"\b(multiple|several|many|numerous|various|different|majority|more popular|less popular|grew|grown|growing|increase|increased|decrease|decreased|over|under|at least|at most|less than|more than|fewer than)\b",
        fact_l,
    ):
        return [fact_text]
    if re.search(r"\b\d+(?:st|nd|rd|th)?\b", fact_l):
        return [fact_text]
    if any(word in fact_l for word in list(NUMBER_WORDS.keys()) + list(ORDINAL_WORDS.keys())):
        return [fact_text]
    return []



def qualitative_quantity_match(fact_norm, answer_norm, answer_numbers):
    plural_synonyms = {"multiple", "several", "many", "numerous", "various", "different"}

    if "multiple" in fact_norm:
        if any(num >= 2 for num in answer_numbers) or any(word in answer_norm for word in plural_synonyms):
            return True
        if any(num == 1 for num in answer_numbers):
            return False

    if "several" in fact_norm:
        if any(num >= 3 for num in answer_numbers) or any(word in answer_norm for word in {"several", "many", "numerous"}):
            return True
        if answer_numbers and all(num < 3 for num in answer_numbers):
            return False

    if re.search(r"\b(many|numerous)\b", fact_norm):
        if any(num >= 3 for num in answer_numbers) or any(word in answer_norm for word in {"many", "numerous", "several"}):
            return True

    if re.search(r"\b(various|different)\b", fact_norm):
        if any(num >= 2 for num in answer_numbers) or any(word in answer_norm for word in {"various", "different", "several", "multiple"}):
            return True

    if "majority" in fact_norm:
        if "majority" in answer_norm:
            return True
        if "minority" in answer_norm:
            return False

    if re.search(r"\b(more popular|grew|grown|growing|increase|increased|higher|greater)\b", fact_norm):
        if re.search(r"\b(grow|grew|grown|growing|increase|increased|increasing|rose|risen|higher|greater|more)\b", answer_norm):
            return True
        if re.search(r"\b(decrease|decreased|decline|declined|less|lower)\b", answer_norm):
            return False

    if re.search(r"\b(less popular|decrease|decreased|lower|fewer|smaller)\b", fact_norm):
        if re.search(r"\b(decrease|decreased|decline|declined|less|lower|fewer|smaller)\b", answer_norm):
            return True
        if re.search(r"\b(grow|grew|grown|increase|increased|more|higher|greater)\b", answer_norm):
            return False

    return None



def quantity_match(answer, quantities, fact_text=""):
    ans = normalize_num_string(answer)
    qvals = [normalize_num_string(x) for x in (quantities or []) if _clean_text(x)]
    qvals.extend(normalize_num_string(x) for x in infer_quantity_targets_from_fact(fact_text))

    if not qvals:
        return None

    deduped_qvals = []
    for q in qvals:
        if q and q not in deduped_qvals:
            deduped_qvals.append(q)
    qvals = deduped_qvals

    answer_numbers = extract_numeric_values(answer)
    answer_comparator = parse_quantity_comparator(answer)
    saw_conflict = False

    for q in qvals:
        comparator = parse_quantity_comparator(q)
        if comparator:
            if answer_comparator == comparator:
                return True
            if answer_numbers:
                op, target = comparator
                if any(compare_numeric(num, op, target) for num in answer_numbers):
                    return True
                saw_conflict = True
                continue

        q_numbers = extract_numeric_values(q)
        if q_numbers and answer_numbers:
            if any(a == qn for a in answer_numbers for qn in q_numbers):
                return True
            saw_conflict = True
            continue

        if ans and (ans in q or q in ans):
            return True

        qualitative = qualitative_quantity_match(q, ans, answer_numbers)
        if qualitative is True:
            return True
        if qualitative is False:
            saw_conflict = True

    return False if saw_conflict else None



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



def content_tokens(text):
    return [tok for tok in normalize_text_for_match(text).split() if tok and tok not in STOPWORD_TOKENS]



def directional_location_match(a, b):
    a_tokens = set(content_tokens(a))
    b_tokens = set(content_tokens(b))
    if not a_tokens or not b_tokens:
        return False

    shared_directions = (a_tokens & b_tokens) & DIRECTION_TOKENS
    if not shared_directions:
        return False

    a_core = a_tokens - GENERIC_LOCATION_TOKENS
    b_core = b_tokens - GENERIC_LOCATION_TOKENS
    if a_core <= shared_directions or b_core <= shared_directions:
        return True

    if "country" in a_tokens or "country" in b_tokens:
        return True

    return False



def text_match_loose(a, b):
    a_n = normalize_text_for_match(a)
    b_n = normalize_text_for_match(b)
    if not a_n or not b_n:
        return False
    if a_n == b_n or a_n in b_n or b_n in a_n:
        return True
    if directional_location_match(a, b):
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
        r"\blocated in the\s+(.+?)[\.\,]?$",
        r"\blocated in\s+(.+?)[\.\,]?$",
        r"\bwas located in\s+(.+?)[\.\,]?$",
        r"\btook place in\s+(.+?)[\.\,]?$",
        r"\bcan only be used in\s+(.+?)[\.\,]?$",
        r"\bknown by\s+(.+?)[\.\,]?$",
        r"\bis in\s+([A-Z][A-Za-z0-9 ,\-/]+)[\.\,]?$",
    ]
    for pattern in strict_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            cand = _clean_text(match.group(1))
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
    return any(pattern in t for pattern in patterns)



def verify_entity_wh(fact, answer, answer_status, evidence_span, gold_evidence):
    fact_text = _clean_text(fact.get("text", ""))
    grounded = span_grounded_in_evidence(evidence_span, gold_evidence)
    target = extract_entity_target_from_fact(fact_text)

    answer_text = _clean_text(answer)
    answer_missing = answer_status in {"insufficient", "api_error"} or answer_text.lower() == "insufficient" or not answer_text
    candidate = answer_text

    if answer_missing and grounded and target and text_match_loose(evidence_span, target):
        candidate = _clean_text(evidence_span)
        answer_missing = False

    if answer_missing:
        return "insufficient", "The answer does not provide enough information."

    if is_existential_or_type_fact(fact_text) or not target:
        if answer_status == "contradicted":
            return "contradict", "The answer result contradicts the fact."
        if grounded and candidate:
            return "support", "The answer provides a grounded value for the fact."
        return "insufficient", "The answer is not sufficiently grounded in the evidence."

    if text_match_loose(candidate, target):
        if grounded or span_grounded_in_evidence(candidate, gold_evidence):
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
        if answer_status == "contradicted" and yn is None:
            yn = "no"

        if answer_status in {"insufficient", "api_error"} or yn is None:
            verification_label = "insufficient"
            reason = "The answer does not clearly determine the fact."
        else:
            reverse_polarity = negated and not contains_explicit_negation(bound_question or question or raw_fact_text)
            if yn == "yes":
                verification_label = "contradict" if reverse_polarity else "support"
                reason = "The answer affirms the fact."
            else:
                verification_label = "support" if reverse_polarity else "contradict"
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
        if matched is None and evidence_span:
            matched = time_match(evidence_span, constraint.get("time", []))

        if matched is True:
            verification_label = "contradict" if negated else "support"
            reason = "The answer matches the time constraint."
        elif matched is False and constraint.get("time"):
            verification_label = "support" if negated else "contradict"
            reason = "The answer conflicts with the time constraint."
        elif answer_status in {"insufficient", "api_error"}:
            verification_label = "insufficient"
            reason = "The answer does not provide enough temporal information."
        else:
            verification_label = "insufficient"
            reason = "The answer does not determine the time constraint."

    elif question_type == "quantity_wh":
        matched = quantity_match(answer, constraint.get("quantity", []), raw_fact_text)
        if matched is None and evidence_span:
            matched = quantity_match(evidence_span, constraint.get("quantity", []), raw_fact_text)

        if matched is True:
            verification_label = "contradict" if negated else "support"
            reason = "The answer matches the quantity constraint."
        elif matched is False:
            verification_label = "support" if negated else "contradict"
            reason = "The answer conflicts with the quantity constraint."
        elif answer_status in {"insufficient", "api_error"}:
            verification_label = "insufficient"
            reason = "The answer does not provide enough quantity information."
        else:
            verification_label = "insufficient"
            reason = "The answer does not determine the quantity constraint."

    elif question_type == "entity_wh":
        verification_label, reason = verify_entity_wh(
            fact={"text": bound_fact_text},
            answer=answer,
            answer_status=answer_status,
            evidence_span=evidence_span,
            gold_evidence=gold_evidence,
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

    with open(in_path, "r", encoding="utf-8") as f:
        raws = json.load(f)
    raws = raws[args.start:args.end]

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

    parser.add_argument(
        "--in_path",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_answer_[T][S]_[E].json",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_verify_[T][S]_[E].json",
    )

    args = parser.parse_args()
    main(args)
