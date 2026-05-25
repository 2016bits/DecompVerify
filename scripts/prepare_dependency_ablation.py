import argparse
import copy
import json
import os
import re
from datetime import datetime


VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")


def _clean_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


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


def replace_placeholders(text, bindings):
    text = _clean_text(text)
    if not text or not bindings:
        return text
    for slot in sorted(bindings.keys(), key=len, reverse=True):
        value = _clean_text(bindings.get(slot, ""))
        if slot.startswith("?") and value:
            text = text.replace(slot, value)
    return text


def replace_placeholders_in_obj(obj, bindings):
    if isinstance(obj, str):
        return replace_placeholders(obj, bindings)
    if isinstance(obj, list):
        return [replace_placeholders_in_obj(item, bindings) for item in obj]
    if isinstance(obj, dict):
        return {key: replace_placeholders_in_obj(value, bindings) for key, value in obj.items()}
    return obj


def unresolved_vars_in_obj(obj):
    found = set()
    if isinstance(obj, str):
        found.update(VAR_PATTERN.findall(obj))
    elif isinstance(obj, list):
        for item in obj:
            found.update(unresolved_vars_in_obj(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            found.update(unresolved_vars_in_obj(value))
    return found


def strip_dependency_from_decomposition(decomposition, materialize_static_slots=True):
    decomposition = copy.deepcopy(decomposition or {})
    original_slots = extract_initial_bindings(decomposition)

    if materialize_static_slots:
        decomposition = replace_placeholders_in_obj(decomposition, original_slots)

    facts = decomposition.get("atomic_facts", [])
    if not isinstance(facts, list):
        facts = []

    removed_dependency_edges = 0
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        removed_dependency_edges += len(fact.get("rely_on", []) or [])
        fact["rely_on"] = []

    decomposition["atomic_facts"] = facts
    decomposition["entity_slots"] = {}
    decomposition["decomposition_type"] = "simple"

    coverage_check = decomposition.get("coverage_check", {})
    if isinstance(coverage_check, dict):
        coverage_check["unresolved_coref_mentions"] = []
        decomposition["coverage_check"] = coverage_check

    return decomposition, {
        "removed_entity_slots": len(original_slots),
        "removed_dependency_edges": removed_dependency_edges,
        "unresolved_variables_after_strip": sorted(unresolved_vars_in_obj(decomposition)),
    }


def strip_dependency_from_question_plan(question_plan, materialize_static_slots=True, bindings=None):
    question_plan = copy.deepcopy(question_plan or {})
    bindings = bindings or {}
    if materialize_static_slots:
        question_plan = replace_placeholders_in_obj(question_plan, bindings)

    qitems = question_plan.get("question_items", [])
    if not isinstance(qitems, list):
        qitems = []

    for item in qitems:
        if not isinstance(item, dict):
            continue
        item["rely_on"] = []
        item["answer_slot"] = None

    question_plan["question_items"] = qitems
    question_plan["decomposition_type"] = "simple"
    return question_plan


def strip_dependency_from_answer_result(answer_result):
    answer_result = copy.deepcopy(answer_result or {})
    answers = answer_result.get("answers", [])
    if not isinstance(answers, list):
        answers = []
    for answer in answers:
        if isinstance(answer, dict):
            answer["bindings_update"] = {}
    answer_result["answers"] = answers
    answer_result["initial_bindings"] = {}
    answer_result["final_bindings"] = {}
    return answer_result


def process_item(item, materialize_static_slots=True, keep_question_plan=False):
    out = copy.deepcopy(item)
    decomposition = item.get("decomposition", {}) or {}
    static_bindings = extract_initial_bindings(decomposition)

    stripped_decomposition, stats = strip_dependency_from_decomposition(
        decomposition,
        materialize_static_slots=materialize_static_slots,
    )
    out["decomposition"] = stripped_decomposition

    if keep_question_plan and "question_plan" in out:
        out["question_plan"] = strip_dependency_from_question_plan(
            out.get("question_plan", {}),
            materialize_static_slots=materialize_static_slots,
            bindings=static_bindings,
        )
    else:
        out.pop("question_plan", None)

    if "answer_result" in out:
        out["answer_result"] = strip_dependency_from_answer_result(out.get("answer_result", {}))

    out["dependency_ablation"] = {
        "mode": "wo_dependency",
        "materialized_static_entity_slots": materialize_static_slots,
        "removed_runtime_bindings": True,
        "removed_rely_on": True,
        "removed_nested_fact_chain_hint": True,
        **stats,
    }
    return out, stats


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
    out_path = resolve_path(args.out_path, args)

    with open(in_path, "r", encoding="utf-8") as file_obj:
        raws = json.load(file_obj)

    if args.slice_input:
        raws = raws[args.start:args.end]

    results = []
    total_removed_slots = 0
    total_removed_edges = 0
    unresolved_items = 0

    for item in raws:
        out, stats = process_item(
            item,
            materialize_static_slots=args.materialize_static_slots,
            keep_question_plan=args.keep_question_plan,
        )
        total_removed_slots += stats["removed_entity_slots"]
        total_removed_edges += stats["removed_dependency_edges"]
        if stats["unresolved_variables_after_strip"]:
            unresolved_items += 1
        results.append(out)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=4, ensure_ascii=False)

    print(f"Saved to {out_path}")
    print(f"Items: {len(results)}")
    print(f"Removed entity slots: {total_removed_slots}")
    print(f"Removed dependency edges: {total_removed_edges}")
    print(f"Items with unresolved variables after strip: {unresolved_items}")
    print("Program finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HOVER")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--class_num", type=str, default="2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=4000)
    parser.add_argument("--plan", type=str, default="qc_plan6.4")
    parser.add_argument("--t", type=str, default="")
    parser.add_argument("--in_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_answers_[T][S]_[E].json")
    parser.add_argument("--out_path", type=str, default="./data/[DATA]/[PLAN]/ablations/[TYPE]_[CLASS]_wo_dependency_decomposed_[T][S]_[E].json")
    parser.add_argument("--slice_input", action="store_true", help="Slice the loaded input with --start/--end before writing.")
    parser.add_argument("--keep_question_plan", action="store_true", help="Keep and strip an existing question_plan instead of forcing regeneration.")
    parser.add_argument(
        "--no_materialize_static_slots",
        dest="materialize_static_slots",
        action="store_false",
        help="Leave ?slot placeholders untouched when removing dependency metadata.",
    )
    parser.set_defaults(materialize_static_slots=True)
    main(parser.parse_args())
