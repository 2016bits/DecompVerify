import os
import re
import json
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher

from tqdm import tqdm

import verify_helpers



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


def _constraint_tokens(text):
    return [
        tok for tok in verify_helpers.normalize_text_for_match(text).split()
        if tok and tok not in verify_helpers.STOPWORD_TOKENS
    ]


def _common_prefix_len(a, b):
    limit = min(len(a), len(b))
    matched = 0
    while matched < limit and a[matched] == b[matched]:
        matched += 1
    return matched


def _soft_token_match(a, b):
    if a == b:
        return True

    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 5 and longer.startswith(shorter):
        return True

    if len(shorter) >= 6 and _common_prefix_len(a, b) >= 5:
        return True

    if len(shorter) >= 4 and abs(len(a) - len(b)) <= 1 and SequenceMatcher(None, a, b).ratio() >= 0.85:
        return True

    if len(shorter) >= 7 and SequenceMatcher(None, a, b).ratio() >= 0.9:
        return True

    return False


def soft_phrase_match(candidate, target):
    candidate_tokens = _constraint_tokens(candidate)
    target_tokens = _constraint_tokens(target)
    if not candidate_tokens or not target_tokens:
        return False

    matched = 0
    for target_token in target_tokens:
        if any(_soft_token_match(target_token, candidate_token) for candidate_token in candidate_tokens):
            matched += 1

    return matched / max(1, len(target_tokens)) >= 0.75


def build_textual_constraint_check(answer, evidence_span, targets, require_all=True):
    observed_texts = [_clean_text(answer), _clean_text(evidence_span)]
    normalized_targets = _clean_list(targets)
    if not normalized_targets:
        return {"targets": [], "observed": observed_texts, "match": None, "matched_targets": [], "unmatched_targets": []}

    matched_targets = []
    unmatched_targets = []
    for target in normalized_targets:
        matched = any(soft_phrase_match(observed, target) for observed in observed_texts if observed)
        if matched:
            matched_targets.append(target)
        else:
            unmatched_targets.append(target)

    match = len(unmatched_targets) == 0 if require_all else len(matched_targets) > 0
    return {
        "targets": normalized_targets,
        "observed": observed_texts,
        "match": match,
        "match_all": len(unmatched_targets) == 0,
        "matched_targets": matched_targets,
        "unmatched_targets": unmatched_targets,
    }



def build_time_check(answer, evidence_span, extracted_values, targets):
    observed_texts = collect_observed_texts(answer, evidence_span, extracted_values, "time")
    normalized_targets = _clean_list(targets)
    if len(normalized_targets) > 1:
        per_target = []
        for target in normalized_targets:
            target_match = None
            for text in observed_texts:
                current = verify_helpers.time_match(text, [target])
                if current is True:
                    target_match = True
                    break
                if current is False:
                    target_match = False if target_match is not True else True
            per_target.append(target_match)

        if all(value is True for value in per_target):
            return {"targets": normalized_targets, "observed": observed_texts, "match": True, "per_target_match": per_target}
        if any(value is False for value in per_target):
            return {"targets": normalized_targets, "observed": observed_texts, "match": False, "per_target_match": per_target}
        return {"targets": normalized_targets, "observed": observed_texts, "match": None, "per_target_match": per_target}

    match = None
    for text in observed_texts:
        current = verify_helpers.time_match(text, normalized_targets)
        if current is True:
            match = True
            break
        if current is False:
            match = False if match is not True else True
    return {"targets": normalized_targets, "observed": observed_texts, "match": match}



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
        "comparison": build_textual_constraint_check(answer, evidence_span, (constraint or {}).get("comparison", [])),
        "unique_modifier": build_textual_constraint_check(answer, evidence_span, (constraint or {}).get("unique_modifier", [])),
        "title_role": build_textual_constraint_check(answer, evidence_span, (constraint or {}).get("title_role", [])),
        "exact_name": build_textual_constraint_check(answer, evidence_span, (constraint or {}).get("exact_name", []), require_all=False),
    }
    required_keys = [key for key in ("time", "quantity", "location") if checks[key].get("targets")]
    all_required_satisfied = all(checks[key].get("match") is True for key in required_keys) if required_keys else True
    explicit_conflict = any(checks[key].get("match") is False for key in ("time", "quantity", "location"))
    textual_required_keys = [key for key in ("comparison", "unique_modifier", "title_role") if checks[key].get("targets")]
    all_textual_satisfied = all(checks[key].get("match") is True for key in textual_required_keys) if textual_required_keys else True
    any_textual_missed = any(checks[key].get("match") is False for key in textual_required_keys)
    return {
        "checks": checks,
        "required_keys": required_keys,
        "all_required_satisfied": all_required_satisfied,
        "explicit_conflict": explicit_conflict,
        "textual_required_keys": textual_required_keys,
        "all_textual_satisfied": all_textual_satisfied,
        "any_textual_missed": any_textual_missed,
        "exact_name_match": checks["exact_name"].get("match"),
        "exact_name_all_match": checks["exact_name"].get("match_all"),
    }



def infer_relation_yesno_mode(fact, qitem, question_text):
    fact_text = _clean_text((fact or {}).get("text", ""))
    negated_fact = verify_helpers.contains_negation(fact) or span_has_effective_negation(fact_text, fact_text)
    question_polarity = _clean_text((qitem or {}).get("question_polarity", "literal_fact")) or "literal_fact"
    question_has_negation = (
        verify_helpers.contains_explicit_negation(question_text)
        or span_has_effective_negation(question_text, fact_text)
    )

    # Trust the positive-probe tag only when the fact itself is negated and
    # the generated question asks the corresponding positive form.
    if question_polarity == "positive_probe_for_negation" and negated_fact and not question_has_negation:
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


def span_has_effective_negation(text, fact_text=""):
    text_l = _clean_text(text).lower()
    if not text_l:
        return False

    text_l = re.sub(r"\bnot\s+(?:only|just|merely|solely)\b", " ", text_l)
    fact_l = _clean_text(fact_text).lower()
    if not re.search(r"\b(?:fail(?:ed|s|ing)?|unsuccessful(?:ly)?)\b", fact_l):
        text_l = re.sub(r"\bunsuccessful(?:ly)?\b", " ", text_l)
    if re.search(r"\b(?:isn't|aren't|wasn't|weren't|doesn't|don't|didn't|hasn't|haven't|hadn't|can't|cannot|couldn't|won't|wouldn't|shouldn't)\b", text_l):
        return True
    return verify_helpers.contains_explicit_negation(text_l)


def has_strong_unique_modifier_match(constraint_eval):
    unique_check = ((constraint_eval.get("checks", {}) or {}).get("unique_modifier", {}) or {})
    if unique_check.get("match") is not True:
        return False
    targets = unique_check.get("targets", []) or []
    if not targets:
        return False
    return all(len(_constraint_tokens(target)) >= 3 for target in targets)


RELATION_COVERAGE_STOPWORDS = verify_helpers.STOPWORD_TOKENS | {
    "and", "or", "that", "which", "who", "whose", "whom", "this", "that",
    "these", "those", "there", "their", "his", "her", "its", "as", "it",
    "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "had", "do", "does", "did", "true", "fact", "occurred", "occur",
}


def relation_content_tokens(text):
    return [
        tok for tok in verify_helpers.normalize_text_for_match(text).split()
        if len(tok) > 2 and tok not in RELATION_COVERAGE_STOPWORDS
    ]


def fact_span_coverage_match(fact_text, evidence_span):
    return fact_span_coverage_ratio(fact_text, evidence_span) >= 0.78


def fact_span_coverage_ratio(fact_text, evidence_span):
    target_tokens = []
    for token in relation_content_tokens(fact_text):
        if token not in target_tokens:
            target_tokens.append(token)
    candidate_tokens = relation_content_tokens(evidence_span)

    if len(target_tokens) < 4 or not candidate_tokens:
        return 0.0

    matched = 0
    for target_token in target_tokens:
        if any(_soft_token_match(target_token, candidate_token) for candidate_token in candidate_tokens):
            matched += 1

    return matched / max(1, len(target_tokens))


def exact_names_recoverable_from_eval(constraint_eval):
    exact_name_check = ((constraint_eval.get("checks", {}) or {}).get("exact_name", {}) or {})
    exact_name_targets = exact_name_check.get("targets", []) or []
    matched_exact_names = exact_name_check.get("matched_targets", []) or []
    unmatched_exact_names = exact_name_check.get("unmatched_targets", []) or []
    near_exact_name_match = bool(matched_exact_names) and len(unmatched_exact_names) <= 1
    return not exact_name_targets or exact_name_check.get("match_all") is True or near_exact_name_match


def contradicted_answer_can_be_recovered(fact_text, evidence_span, constraint_eval):
    exact_name_check = (constraint_eval.get("checks", {}) or {}).get("exact_name", {}) or {}
    exact_name_targets = exact_name_check.get("targets", []) or []
    has_structured_target = bool(
        exact_name_targets
        or constraint_eval.get("required_keys")
        or constraint_eval.get("textual_required_keys")
    )
    if not has_structured_target:
        return False
    if exact_name_targets and exact_name_check.get("match_all") is not True:
        return False
    return fact_span_coverage_ratio(fact_text, evidence_span) >= 0.9


def adjust_relation_yesno_by_textual_constraints(label, answer_status, mode, grounded, constraint_eval, metadata, fact_text="", evidence_span=""):
    has_textual_targets = bool(constraint_eval.get("textual_required_keys"))
    all_textual_satisfied = constraint_eval.get("all_textual_satisfied", True)
    textual_keys = set(constraint_eval.get("textual_required_keys", []))
    all_required_satisfied = constraint_eval.get("all_required_satisfied", False)
    explicit_conflict = constraint_eval.get("explicit_conflict", False)
    exact_name_check = (constraint_eval.get("checks", {}) or {}).get("exact_name", {}) or {}
    exact_names_recoverable = exact_names_recoverable_from_eval(constraint_eval)
    matched_exact_names = exact_name_check.get("matched_targets", []) or []
    unmatched_exact_names = exact_name_check.get("unmatched_targets", []) or []
    near_exact_name_match = bool(matched_exact_names) and len(unmatched_exact_names) <= 1
    unique_without_role = "unique_modifier" in textual_keys and "title_role" not in textual_keys
    strong_unique_match = has_strong_unique_modifier_match(constraint_eval)
    allow_textual_upgrade = (
        not unique_without_role
        or "comparison" in textual_keys
        or exact_name_check.get("match_all") is True
        or strong_unique_match
    )
    allow_contradicted_recovery = (
        answer_status != "contradicted"
        or contradicted_answer_can_be_recovered(fact_text, evidence_span, constraint_eval)
    )

    if (
        label == "contradict"
        and answer_status == "contradicted"
        and mode == "literal_fact"
        and grounded
        and has_textual_targets
        and all_textual_satisfied
        and allow_textual_upgrade
        and allow_contradicted_recovery
        and exact_names_recoverable
        and not explicit_conflict
    ):
        return (
            "support",
            "Although the answer denied the relation, the grounded evidence span matches the fact's key textual modifiers.",
            {
                "yesno": metadata.get("yesno"),
                "grounded": grounded,
                "explicit_conflict": False,
            },
        )

    if (
        label == "contradict"
        and answer_status == "contradicted"
        and mode == "literal_fact"
        and grounded
        and not has_textual_targets
        and near_exact_name_match
        and allow_contradicted_recovery
        and all_required_satisfied
        and not explicit_conflict
    ):
        return (
            "support",
            "Although the answer denied the relation, the grounded evidence span matches the fact's exact-name target closely enough to recover the relation.",
            {
                "yesno": metadata.get("yesno"),
                "grounded": grounded,
                "explicit_conflict": False,
            },
        )

    return label, None, metadata


def adjust_entity_by_textual_constraints(label, answer, answer_status, grounded, constraint_eval):
    has_textual_targets = bool(constraint_eval.get("textual_required_keys"))
    all_textual_satisfied = constraint_eval.get("all_textual_satisfied", True)
    exact_name_match = constraint_eval.get("exact_name_match") is True
    all_required_satisfied = constraint_eval.get("all_required_satisfied", True)

    if (
        label in {"contradict", "insufficient"}
        and answer_status not in {"insufficient", "api_error"}
        and _clean_text(answer).lower() != "insufficient"
        and grounded
        and exact_name_match
        and (not has_textual_targets or all_textual_satisfied)
        and not constraint_eval.get("explicit_conflict", False)
    ):
        return (
            "support",
            "The entity answer is grounded and matches the fact's exact-name target despite the initial verifier mismatch.",
            {"explicit_conflict": False},
        )

    if (
        label in {"contradict", "insufficient"}
        and answer_status == "supported"
        and _clean_text(answer).lower() != "insufficient"
        and grounded
        and all_required_satisfied
        and (not has_textual_targets or all_textual_satisfied)
        and not constraint_eval.get("explicit_conflict", False)
    ):
        return (
            "support",
            "The entity answer is grounded and satisfies the fact's structured constraints despite the initial target mismatch.",
            {"explicit_conflict": False},
        )

    return label, None, {"explicit_conflict": label == "contradict"}


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
    label, reason, metadata = resolve_relation_yesno_label(mode, yesno, grounded, constraint_eval)

    if (
        mode == "positive_probe_for_negation"
        and yesno == "no"
        and label == "support"
        and not span_has_effective_negation(evidence_span, fact.get("text", ""))
    ):
        if answer_status == "contradicted":
            return (
                "contradict",
                "The positive probe was denied, but the grounded evidence span does not express the negation and instead conflicts with the negated fact.",
                {"yesno": yesno, "grounded": grounded, "explicit_conflict": True},
            )
        return (
            "insufficient",
            "The positive probe was denied, but the evidence span does not explicitly ground the negated fact.",
            {"yesno": yesno, "grounded": grounded, "explicit_conflict": False},
        )

    adjusted_label, adjusted_reason, adjusted_metadata = adjust_relation_yesno_by_textual_constraints(
        label=label,
        answer_status=answer_status,
        mode=mode,
        grounded=grounded,
        constraint_eval=constraint_eval,
        metadata=metadata,
        fact_text=fact.get("text", ""),
        evidence_span=evidence_span,
    )

    if (
        adjusted_label == "contradict"
        and label == "contradict"
        and answer_status == "contradicted"
        and mode == "literal_fact"
        and grounded
        and not constraint_eval.get("textual_required_keys")
        and constraint_eval.get("all_required_satisfied", False)
        and not constraint_eval.get("explicit_conflict", False)
        and exact_names_recoverable_from_eval(constraint_eval)
        and contradicted_answer_can_be_recovered(fact.get("text", ""), evidence_span, constraint_eval)
        and fact_span_coverage_match(fact.get("text", ""), evidence_span)
    ):
        return (
            "support",
            "Although the answer denied the relation, the evidence span covers the fact's core content with no explicit value conflict.",
            {"yesno": yesno, "grounded": grounded, "explicit_conflict": False},
        )

    return adjusted_label, (adjusted_reason or reason), adjusted_metadata



def verify_time_wh(fact, answer_status, constraint_eval, evidence_span="", gold_evidence=""):
    negated = verify_helpers.contains_negation(fact)
    matched = constraint_eval["checks"]["time"].get("match")
    if matched is True:
        return ("contradict" if negated else "support"), "The extracted time value matches the time constraint.", {"explicit_conflict": False}
    if matched is False and (fact.get("constraint", {}) or {}).get("time"):
        return ("support" if negated else "contradict"), "The extracted time value conflicts with the time constraint.", {"explicit_conflict": True}
    if (
        not (fact.get("constraint", {}) or {}).get("time")
        and answer_status == "supported"
        and verify_helpers.span_grounded_in_evidence(evidence_span, gold_evidence)
        and any(verify_helpers.looks_temporal_text(text) for text in constraint_eval["checks"]["time"].get("observed", []))
    ):
        return ("contradict" if negated else "support"), "The answer provides a grounded time value for a fact with no explicit target time constraint.", {"explicit_conflict": False}
    if answer_status in {"insufficient", "api_error"}:
        return "insufficient", "The answer does not provide enough temporal information.", {"explicit_conflict": False}
    return "insufficient", "The answer does not determine the time constraint.", {"explicit_conflict": False}



def verify_quantity_wh(fact, answer_status, constraint_eval, evidence_span="", gold_evidence=""):
    negated = verify_helpers.contains_negation(fact)
    matched = constraint_eval["checks"]["quantity"].get("match")
    if matched is True:
        return ("contradict" if negated else "support"), "The extracted quantity value matches the quantity constraint.", {"explicit_conflict": False}
    if matched is False:
        return ("support" if negated else "contradict"), "The extracted quantity value conflicts with the quantity constraint.", {"explicit_conflict": True}
    if (
        not (fact.get("constraint", {}) or {}).get("quantity")
        and answer_status == "supported"
        and verify_helpers.span_grounded_in_evidence(evidence_span, gold_evidence)
        and any(verify_helpers.extract_numeric_values(text) for text in constraint_eval["checks"]["quantity"].get("observed", []))
    ):
        return ("contradict" if negated else "support"), "The answer provides a grounded quantity value for a fact with no explicit target quantity constraint.", {"explicit_conflict": False}
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
    bound_fact = dict(fact)
    bound_fact["text"] = bound_fact_text

    if question_type == "relation_yesno":
        verification_label, reason, verifier_metadata = verify_relation_yesno(bound_fact, qitem, answer, answer_status, evidence_span, gold_evidence, extracted_values, constraint_eval)
    elif question_type == "time_wh":
        verification_label, reason, verifier_metadata = verify_time_wh(fact, answer_status, constraint_eval, evidence_span, gold_evidence)
    elif question_type == "quantity_wh":
        verification_label, reason, verifier_metadata = verify_quantity_wh(fact, answer_status, constraint_eval, evidence_span, gold_evidence)
    elif question_type == "entity_wh":
        verification_label, reason = verify_helpers.verify_entity_wh(
            fact={"text": bound_fact_text},
            answer=answer,
            answer_status=answer_status,
            evidence_span=evidence_span,
            gold_evidence=gold_evidence,
        )
        verifier_metadata = {"explicit_conflict": verification_label == "contradict"}
        adjusted_label, adjusted_reason, adjusted_metadata = adjust_entity_by_textual_constraints(
            label=verification_label,
            answer=answer,
            answer_status=answer_status,
            grounded=verify_helpers.span_grounded_in_evidence(evidence_span, gold_evidence),
            constraint_eval=constraint_eval,
        )
        verification_label = adjusted_label
        if adjusted_reason:
            reason = adjusted_reason
        verifier_metadata = adjusted_metadata
        recovered = recover_entity_wh_from_constraints(answer_status, evidence_span, gold_evidence, constraint_eval)
        if verification_label == "insufficient" and recovered is not None:
            verification_label, reason, verifier_metadata = recovered
    else:
        verification_label, reason, verifier_metadata = verify_fallback(fact, answer, answer_status, evidence_span, gold_evidence)

    adjudication = run_second_pass_adjudication(
        fact=bound_fact,
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
