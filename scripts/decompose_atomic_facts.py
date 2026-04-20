import os
import re
import json
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm
from openai import OpenAI

from prompts.decompose_prompts import get_decompose_prompt


CONSTRAINT_LIST_KEYS = [
    "time",
    "quantity",
    "comparison",
    "location",
    "title_role",
    "exact_name",
    "unique_modifier",
]

NEGATION_CUE_PATTERNS = [
    r"\bnot\b",
    r"\bno\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bno longer\b",
    r"\bneither\b",
    r"\bnor\b",
    r"\bfail(?:ed|s|ing)?\b",
]

COMPARISON_PATTERNS = [
    r"\bsame\b",
    r"\bdifferent\b",
    r"\bmore than\b",
    r"\bless than\b",
    r"\bat least\b",
    r"\bat most\b",
    r"\bgreater than\b",
    r"\blower than\b",
    r"\bhigher than\b",
    r"\bearlier than\b",
    r"\blater than\b",
    r"\blargest\b",
    r"\bsmallest\b",
    r"\bbiggest\b",
    r"\bhighest\b",
    r"\blowest\b",
    r"\boldest\b",
    r"\byoungest\b",
    r"\bfirst\b",
    r"\blast\b",
    r"\bordered\b",
    r"\branked\b",
]

ROLE_KEYWORDS = [
    "director", "producer", "host", "ambassador", "president", "prime minister",
    "actor", "actress", "writer", "author", "coach", "captain", "governor",
    "mayor", "minister", "king", "queen", "commander", "chairman", "chairperson",
    "ceo", "founder", "inventor", "composer", "singer", "drummer", "guitarist",
    "role", "title", "character",
]

UNIQUE_MODIFIER_PATTERNS = [
    r"\bonly\b",
    r"\bexactly\b",
    r"\bformer\b",
    r"\blatter\b",
    r"\bcurrent\b",
    r"\bprevious\b",
    r"\boriginal\b",
    r"\bsole\b",
    r"\bunique\b",
    r"\bbest known\b",
    r"\baward-winning\b",
    r"\bco-produced by\b",
]

TEMPORAL_PATTERNS = [
    re.compile(r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{1,2})?(?:,\s*|\s+)?\d{3,4}\b", re.IGNORECASE),
    re.compile(r"\b[1-9][0-9]{2,3}\b"),
    re.compile(r"\b(?:early|mid|late)\s+[1-9][0-9]{2,3}s\b", re.IGNORECASE),
    re.compile(r"\b[1-9][0-9]{2,3}s\b"),
]

QUANTITY_PATTERNS = [
    re.compile(r"\b(?:more than|less than|at least|at most|over|under|exactly|about|around|approximately|roughly)\s+[^,.;]+", re.IGNORECASE),
    re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?(?:st|nd|rd|th)?(?:\s+[A-Za-z%]+)?\b"),
]

PROPER_NOUN_PATTERN = re.compile(r"\b(?:[A-Z][A-Za-z0-9'&\-\.]*)(?:\s+(?:[A-Z][A-Za-z0-9'&\-\.]*(?:\s+[A-Z][A-Za-z0-9'&\-\.]*)*))?\b")
VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")
COREf_PRONOUN_PATTERN = re.compile(r"\b(he|she|they|them|his|her|hers|their|theirs|its|it|this|that|these|those|former|latter)\b", re.IGNORECASE)

MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

NON_ENTITY_CAPITALIZED = {
    "The", "A", "An", "In", "On", "At", "By", "From", "To", "And", "Or",
    "He", "She", "It", "They", "This", "That", "These", "Those",
    "Both", "Yes", "No", "Former", "Latter",
}

QUANTITY_TRAILING_STOPWORDS = {
    "in", "on", "at", "of", "for", "to", "from", "by", "with",
    "is", "are", "was", "were", "be", "been", "being",
}



def build_client_and_model(plan, port='8370'):
    plan = (plan or "").lower()

    if plan in {"local", "localhost", "vllm"} or plan.startswith("plan"):
        client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        model = "Meta-Llama-3-70B-Instruct-AutoAWQ-4bit"
        return client, model, None, None, None, 0.0

    if plan.startswith("azure"):
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set in environment variables.")
        client = OpenAI(api_key=api_key, base_url="https://open1027.openai.azure.com/openai/v1/")
        return client, "gpt-4o", None, {}, 512, 0.7

    if plan.startswith("qwen_plan"):
        client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        return client, "qwen2.5-72b-awq", "You are a helpful assistant.", None, None, 0.7

    if plan.startswith("qc_plan"):
        api_key = os.getenv("AIPING_API_KEY")
        if not api_key:
            raise ValueError("AIPING_API_KEY is not set in environment variables.")
        client = OpenAI(api_key=api_key, base_url="https://www.aiping.cn/api/v1")
        return client, "DeepSeek-V3.2", None, {}, 2048, 0.7

    raise ValueError(
        f"Unsupported plan: {plan}. Supported plans are: local, azure, azure_gpt4o, qc_plan, qwen_plan"
    )



def llm(input_text, plan='local', port='8370'):
    client, model, system_prompt, extra_body, max_tokens, temperature = build_client_and_model(plan, port)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": input_text})

    kwargs = {"model": model, "messages": messages}
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    if not response or not getattr(response, "choices", None):
        raise ValueError(f"Empty API response: {response}")

    choice = response.choices[0]
    if not choice or not getattr(choice, "message", None):
        raise ValueError(f"Missing message in API response: {response}")

    content = getattr(choice.message, "content", None)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        content = "\n".join(text_parts).strip() if text_parts else None

    if content is None:
        raise ValueError(f"Model returned empty content. Full choice: {choice}")
    if not isinstance(content, str):
        content = str(content)
    content = content.strip()
    if not content:
        raise ValueError(f"Model returned blank content. Full choice: {choice}")
    return content



def extract_json_block(text):
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    first_brace = text.find("{")
    if first_brace == -1:
        raise ValueError("No JSON found.")

    stack = 0
    in_string = False
    escape = False
    for idx in range(first_brace, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0:
                return text[first_brace:idx + 1]
    raise ValueError("Incomplete JSON block.")



def parse_decompose_output(output_text):
    return json.loads(extract_json_block(output_text))



def _clean_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()



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



def normalize_match_text(text):
    text = _clean_text(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()



def normalize_constraint(constraint):
    if not isinstance(constraint, dict):
        constraint = {}

    normalized = {"negation": True if constraint.get("negation") is True else None}
    for key in CONSTRAINT_LIST_KEYS:
        normalized[key] = _clean_list(constraint.get(key, []))
    return normalized



def normalize_coverage(coverage):
    normalized = []
    if not isinstance(coverage, list):
        coverage = []

    seen = set()
    for item in coverage:
        if isinstance(item, dict):
            ctype = _clean_text(item.get("type", "")).lower().replace("-", "_")
            value = _clean_text(item.get("value", ""))
        else:
            ctype = "other"
            value = _clean_text(item)
        if not ctype or not value:
            continue
        key = (ctype, normalize_match_text(value))
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"type": ctype, "value": value})
    return normalized



def normalize_entity_slots(entity_slots):
    if not isinstance(entity_slots, dict):
        return {}

    normalized = {}
    for slot, raw in entity_slots.items():
        slot = _clean_text(slot)
        if not slot.startswith("?"):
            continue

        if isinstance(raw, dict):
            value = _clean_text(raw.get("value", ""))
            mentions = _clean_list(raw.get("mentions", []))
        else:
            value = _clean_text(raw)
            mentions = [value] if value else []

        if not value:
            continue
        if value not in mentions:
            mentions.insert(0, value)
        normalized[slot] = {"value": value, "mentions": mentions}
    return normalized



def contains_explicit_negation(text):
    text_l = _clean_text(text).lower()
    return any(re.search(pattern, text_l) for pattern in NEGATION_CUE_PATTERNS)



def text_covers(target, candidate):
    target_n = normalize_match_text(target)
    candidate_n = normalize_match_text(candidate)
    if not target_n or not candidate_n:
        return False
    if target_n == candidate_n or target_n in candidate_n or candidate_n in target_n:
        return True

    target_tokens = set(target_n.split())
    candidate_tokens = set(candidate_n.split())
    if not target_tokens or not candidate_tokens:
        return False

    overlap = len(target_tokens & candidate_tokens) / max(1, len(target_tokens))
    return overlap >= 0.8



def extract_temporal_targets(text):
    targets = []
    for pattern in TEMPORAL_PATTERNS:
        for match in pattern.finditer(text):
            value = _clean_text(match.group(0))
            if value and value not in targets:
                targets.append(value)
    return targets



def extract_quantity_targets(text):
    targets = []
    for pattern in QUANTITY_PATTERNS:
        for match in pattern.finditer(text):
            value = _clean_text(match.group(0))
            if not value:
                continue
            parts = value.split()
            while len(parts) > 1 and parts[-1].lower() in QUANTITY_TRAILING_STOPWORDS:
                parts.pop()
            value = _clean_text(" ".join(parts)).strip(" ,;:.")
            if not value:
                continue
            if re.fullmatch(r"\d{3,4}", value):
                continue
            if value not in targets:
                targets.append(value)
    return targets



def extract_comparison_targets(text):
    text_l = _clean_text(text).lower()
    targets = []
    for pattern in COMPARISON_PATTERNS:
        match = re.search(pattern, text_l)
        if match:
            value = _clean_text(match.group(0))
            if value and value not in targets:
                targets.append(value)
    return targets



def extract_location_targets(text):
    targets = []
    patterns = [
        re.compile(r"\b(?:in|at|from|on|to)\s+([A-Z][A-Za-z0-9'&\-\.]+(?:\s+[A-Z][A-Za-z0-9'&\-\.]*)*)"),
        re.compile(r"\b(?:located in|born in|shown on|shown in|took place in)\s+([A-Z][A-Za-z0-9'&\-\.]+(?:\s+[A-Z][A-Za-z0-9'&\-\.]*)*)", re.IGNORECASE),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = _clean_text(match.group(1))
            if value and value.lower() not in MONTH_WORDS and value not in targets:
                targets.append(value)
    return targets



def extract_role_title_targets(text):
    text_l = _clean_text(text).lower()
    targets = []
    for keyword in ROLE_KEYWORDS:
        if keyword in text_l and keyword not in targets:
            targets.append(keyword)

    phrase_patterns = [
        re.compile(r"\b(?:same|different)\s+(director|producer|host|ambassador|president|title|role)\b", re.IGNORECASE),
        re.compile(r"\bbest known(?:\s+for)?\s+[^,.;]+", re.IGNORECASE),
        re.compile(r"\bhost(?:ed)?\s+[^,.;]+", re.IGNORECASE),
    ]
    for pattern in phrase_patterns:
        for match in pattern.finditer(text):
            value = _clean_text(match.group(0))
            if value and value not in targets:
                targets.append(value)
    return targets



def extract_exact_name_targets(text):
    targets = []
    for match in PROPER_NOUN_PATTERN.finditer(text):
        raw_value = _clean_text(match.group(0))
        if not raw_value:
            continue
        parts = re.split(r"(?<=[.!?])\s+", raw_value)
        for part in parts:
            value = _clean_text(part).strip(" ,;:.")
            if not value or value in NON_ENTITY_CAPITALIZED:
                continue
            lower = value.lower()
            if lower in MONTH_WORDS:
                continue
            if value not in targets:
                targets.append(value)
    return targets



def extract_unique_modifier_targets(text):
    text_l = _clean_text(text).lower()
    targets = []
    for pattern in UNIQUE_MODIFIER_PATTERNS:
        match = re.search(pattern, text_l)
        if match:
            value = _clean_text(match.group(0))
            if value and value not in targets:
                targets.append(value)
    return targets



def collect_claim_coverage_targets(claim):
    claim = _clean_text(claim)
    units = []

    def add_units(unit_type, values):
        for value in values:
            value = _clean_text(value)
            if value:
                units.append({"type": unit_type, "value": value})

    if contains_explicit_negation(claim):
        units.append({"type": "negation", "value": "negation"})

    add_units("time", extract_temporal_targets(claim))
    add_units("quantity", extract_quantity_targets(claim))
    add_units("comparison", extract_comparison_targets(claim))
    add_units("location", extract_location_targets(claim))
    add_units("title_role", extract_role_title_targets(claim))
    add_units("exact_name", extract_exact_name_targets(claim))
    add_units("unique_modifier", extract_unique_modifier_targets(claim))

    deduped = []
    seen = set()
    for unit in units:
        key = (unit["type"], normalize_match_text(unit["value"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(unit)
    return deduped



def coverage_values_for_fact(fact, entity_slots):
    values = [_clean_text(fact.get("text", ""))]
    constraint = fact.get("constraint", {}) or {}
    coverage = fact.get("coverage", []) or []

    if constraint.get("negation") is True:
        values.append("negation")

    for key in CONSTRAINT_LIST_KEYS:
        values.extend(_clean_list(constraint.get(key, [])))

    for item in coverage:
        values.append(item.get("value", ""))

    for slot in VAR_PATTERN.findall(fact.get("text", "")):
        slot_info = entity_slots.get(slot, {})
        if isinstance(slot_info, dict):
            values.append(slot_info.get("value", ""))
            values.extend(slot_info.get("mentions", []))
    return [value for value in values if _clean_text(value)]



def coverage_unit_covered(unit, fact, entity_slots):
    unit_type = unit.get("type", "")
    unit_value = unit.get("value", "")
    constraint = fact.get("constraint", {}) or {}
    coverage = fact.get("coverage", []) or []

    if unit_type == "negation":
        return constraint.get("negation") is True or any(item.get("type") == "negation" for item in coverage)

    typed_values = []
    if unit_type in CONSTRAINT_LIST_KEYS:
        typed_values.extend(_clean_list(constraint.get(unit_type, [])))
    typed_values.extend(item.get("value", "") for item in coverage if item.get("type") == unit_type)

    for candidate in typed_values:
        if text_covers(unit_value, candidate):
            return True

    for candidate in coverage_values_for_fact(fact, entity_slots):
        if text_covers(unit_value, candidate):
            return True
    return False



def find_unresolved_coref_mentions(facts):
    issues = []
    for fact in facts:
        text = _clean_text(fact.get("text", ""))
        if not text:
            continue
        if "?" in text:
            stripped = re.sub(VAR_PATTERN, " ", text)
        else:
            stripped = text
        matches = []
        for match in COREf_PRONOUN_PATTERN.finditer(stripped):
            token = match.group(1)
            if token.lower() in {"it"} and re.search(r"Is it true that", stripped, flags=re.IGNORECASE):
                continue
            if token.lower() == "that" and token != "That":
                continue
            matches.append(token)
        if matches:
            issues.append({"fact_id": fact.get("id", ""), "mentions": sorted(set(matches))})
    return issues



def looks_coordinated_claim(claim):
    claim_l = f" {_clean_text(claim).lower()} "
    patterns = [
        r"\band\b",
        r"\bas well as\b",
        r"\balong with\b",
        r"\bplus\b",
        r"\bboth\b.*\band\b",
        r",\s*(?:and|while|but)\s+",
    ]
    return any(re.search(pattern, claim_l) for pattern in patterns)



def infer_critical_reasons(fact, claim, decomposition_type, coordinated_root_ids):
    reasons = []
    if fact.get("_critical_raw") is True:
        reasons.append("model_marked_critical")

    constraint = fact.get("constraint", {}) or {}
    coverage = fact.get("coverage", []) or []
    coverage_types = {item.get("type") for item in coverage}
    text_l = _clean_text(fact.get("text", "")).lower()

    if decomposition_type == "comparison":
        reasons.append("comparison_claim")
    if fact.get("id") in coordinated_root_ids:
        reasons.append("coordinated_conjunct")
    if constraint.get("negation") is True:
        reasons.append("negation")

    for key in CONSTRAINT_LIST_KEYS:
        if constraint.get(key):
            reasons.append(key)
    for ctype in coverage_types:
        if ctype in {"comparison", "title_role", "unique_modifier", "location"}:
            reasons.append(ctype)

    if any(re.search(pattern, text_l) for pattern in UNIQUE_MODIFIER_PATTERNS):
        reasons.append("unique_modifier")
    if any(re.search(pattern, text_l) for pattern in COMPARISON_PATTERNS):
        reasons.append("comparison")
    if re.search(r"\b(?:born in|shown on|shown in|located in|took place in)\b", text_l):
        reasons.append("location")

    deduped = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped



def normalize_fact(fact, idx):
    if not isinstance(fact, dict):
        fact = {}

    rely_on = _clean_list(fact.get("rely_on", []))
    return {
        "id": _clean_text(fact.get("id", f"f{idx + 1}")) or f"f{idx + 1}",
        "text": _clean_text(fact.get("text", "")),
        "rely_on": rely_on,
        "constraint": normalize_constraint(fact.get("constraint", {})),
        "coverage": normalize_coverage(fact.get("coverage", [])),
        "critical": False,
        "critical_reasons": [],
        "_critical_raw": True if fact.get("critical") is True else None,
    }



def build_coverage_report(claim, atomic_facts, entity_slots):
    required_units = collect_claim_coverage_targets(claim)
    missing_units = []
    covered_units = []

    for unit in required_units:
        if any(coverage_unit_covered(unit, fact, entity_slots) for fact in atomic_facts):
            covered_units.append(unit)
        else:
            missing_units.append(unit)

    unresolved_coref = find_unresolved_coref_mentions(atomic_facts)
    return {
        "required_units": required_units,
        "covered_units": covered_units,
        "missing_units": missing_units,
        "unresolved_coref_mentions": unresolved_coref,
        "passed": not missing_units and not unresolved_coref,
    }



def normalize_decomposition_result(parsed, claim):
    dtype = _clean_text(parsed.get("decomposition_type", "simple")).lower()
    if dtype not in {"simple", "coordinated", "nested", "comparison"}:
        dtype = "simple"

    atomic_facts = parsed.get("atomic_facts", [])
    if not isinstance(atomic_facts, list):
        atomic_facts = []

    normalized_facts = [normalize_fact(fact, idx) for idx, fact in enumerate(atomic_facts)]
    entity_slots = normalize_entity_slots(parsed.get("entity_slots", {}))

    root_fact_ids = [fact["id"] for fact in normalized_facts if not fact.get("rely_on")]
    coordinated_root_ids = set()
    if len(root_fact_ids) > 1 and (dtype == "coordinated" or looks_coordinated_claim(claim)):
        coordinated_root_ids.update(root_fact_ids)

    for fact in normalized_facts:
        reasons = infer_critical_reasons(fact, claim, dtype, coordinated_root_ids)
        fact["critical_reasons"] = reasons
        fact["critical"] = bool(reasons)
        fact.pop("_critical_raw", None)

    coverage_check = build_coverage_report(claim, normalized_facts, entity_slots)
    return {
        "claim": claim,
        "decomposition_type": dtype,
        "entity_slots": entity_slots,
        "coverage_check": coverage_check,
        "atomic_facts": normalized_facts,
    }



def validate_decomposition_result(result):
    if "claim" not in result or "atomic_facts" not in result:
        raise ValueError("Invalid decomposition result.")
    if "entity_slots" not in result or "coverage_check" not in result:
        raise ValueError("Missing entity_slots or coverage_check.")

    for fact in result["atomic_facts"]:
        required = {"id", "text", "rely_on", "constraint", "coverage", "critical", "critical_reasons"}
        if not required.issubset(fact.keys()):
            raise ValueError("Invalid atomic fact.")

    coverage_check = result.get("coverage_check", {}) or {}
    missing_units = coverage_check.get("missing_units", []) or []
    unresolved_coref = coverage_check.get("unresolved_coref_mentions", []) or []
    problems = []
    if missing_units:
        pretty = ", ".join(f"{item['type']}={item['value']}" for item in missing_units[:10])
        problems.append(f"Missing coverage for: {pretty}")
    if unresolved_coref:
        pretty = ", ".join(
            f"{item.get('fact_id', '?')} -> {'/'.join(item.get('mentions', []))}" for item in unresolved_coref[:10]
        )
        problems.append(f"Unresolved coreference mentions: {pretty}")
    if problems:
        raise ValueError("Coverage mismatch. " + " | ".join(problems))
    return True



def fallback_decomposition(claim):
    atomic_fact = normalize_fact({
        "id": "f1",
        "text": claim,
        "rely_on": [],
        "constraint": {"negation": True if contains_explicit_negation(claim) else None},
        "coverage": [{"type": "claim", "value": claim}],
        "critical": True,
    }, 0)
    atomic_fact["critical"] = True
    atomic_fact["critical_reasons"] = ["fallback_claim_fact"]

    coverage_check = build_coverage_report(claim, [atomic_fact], {})
    return {
        "claim": claim,
        "decomposition_type": "simple",
        "entity_slots": {},
        "coverage_check": coverage_check,
        "atomic_facts": [atomic_fact],
    }



def process_data_item(data, plan, port):
    claim = data["claim"]
    infer_count = 0
    last_error = None
    feedback = ""

    while infer_count < 3:
        try:
            prompt = get_decompose_prompt(claim, feedback=feedback)
            output = llm(prompt, plan=plan, port=port)
            parsed = parse_decompose_output(output)
            result = normalize_decomposition_result(parsed, claim)
            validate_decomposition_result(result)

            out = dict(data)
            out["decomposition"] = result
            return out
        except Exception as exc:
            last_error = exc
            feedback = _clean_text(exc)
            print(f"[Retry {infer_count + 1}/3] Error during decomposition: {exc}")
            infer_count += 1

    print(f"[Fallback] decomposition failed: {last_error}")
    out = dict(data)
    out["decomposition"] = fallback_decomposition(claim)
    return out



def main(args):
    in_path = args.in_path.replace("[DATA]", args.dataset).replace("[TYPE]", args.data_type)

    with open(in_path, "r", encoding="utf-8") as file_obj:
        raws = json.load(file_obj)
    raws = raws[args.start:args.end]

    partial_func = partial(process_data_item, plan=args.plan, port=args.port)
    results = []

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

    with open(out_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=4, ensure_ascii=False)

    print(f"Saved to {out_path}")
    print("Program finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HOVER_subset")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--class_num", type=str, default="2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument("--port", type=str, default="8270")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--plan", type=str, default="local")
    parser.add_argument("--t", type=str, default="")
    parser.add_argument("--in_path", type=str, default="./data/[DATA]/converted_data/[TYPE].json", help="Input path template")
    parser.add_argument("--out_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_decomposed_[T][S]_[E].json", help="Output path template")
    main(parser.parse_args())
