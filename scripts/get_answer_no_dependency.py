import argparse
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial

from tqdm import tqdm

import get_answer


def sanitize_atomic_fact(fact, fallback_fact_id=""):
    fact = copy.deepcopy(fact or {})
    fact.setdefault("id", fallback_fact_id)
    fact["rely_on"] = []
    return fact


def sanitize_question_item(question_item):
    question_item = copy.deepcopy(question_item or {})
    question_item["rely_on"] = []
    question_item["answer_slot"] = None
    question_item["constraint_questions"] = []
    return question_item


def generate_answers_for_item_no_dependency(data, plan, port):
    claim = data["claim"]
    evidence = data["gold_evidence"]
    decomposition = data.get("decomposition", {}) or {}
    question_plan = data.get("question_plan", {}) or {}

    atomic_facts = decomposition.get("atomic_facts", []) or []
    fact_map = {fact.get("id"): fact for fact in atomic_facts}
    question_items = question_plan.get("question_items", []) or []

    answers = []
    answer_issues = []

    for qitem in question_items:
        qitem = sanitize_question_item(qitem)
        fact_id = qitem.get("fact_id", "")
        atomic_fact = sanitize_atomic_fact(
            fact_map.get(fact_id, {
                "id": fact_id,
                "text": qitem.get("fact_text", ""),
                "constraint": qitem.get("constraint", {}),
                "coverage": qitem.get("coverage", []),
                "critical": qitem.get("critical", False),
                "critical_reasons": qitem.get("critical_reasons", []),
            }),
            fallback_fact_id=fact_id,
        )

        result = get_answer.answer_one_question(
            claim=claim,
            evidence=evidence,
            atomic_fact=atomic_fact,
            question_item=qitem,
            current_bindings={},
            plan=plan,
            port=port,
        )
        result["bindings_update"] = {}
        answers.append(result)

        unresolved = sorted(set(
            get_answer.unresolved_vars(result.get("question", ""))
            + get_answer.unresolved_vars(atomic_fact.get("text", ""))
        ))
        if unresolved:
            answer_issues.append(
                f"No-dependency run left unresolved variables for {fact_id}: {unresolved}"
            )

    return {
        "claim": claim,
        "answers": answers,
        "initial_bindings": {},
        "final_bindings": {},
    }, answer_issues


def process_data_item(data, plan, port):
    decomposition = copy.deepcopy(data.get("decomposition", {"claim": data["claim"], "atomic_facts": []}))
    question_plan = copy.deepcopy(data.get("question_plan", {"question_items": []}))

    for fact in decomposition.get("atomic_facts", []) or []:
        if isinstance(fact, dict):
            fact["rely_on"] = []
    for qitem in question_plan.get("question_items", []) or []:
        if isinstance(qitem, dict):
            qitem["rely_on"] = []
            qitem["answer_slot"] = None

    answer_result, answer_issues = generate_answers_for_item_no_dependency(
        {
            "claim": data["claim"],
            "gold_evidence": data.get("gold_evidence", data.get("evidence", "")),
            "decomposition": decomposition,
            "question_plan": question_plan,
        },
        plan=plan,
        port=port,
    )

    return {
        "id": data["id"],
        "claim": data["claim"],
        "gold_evidence": data.get("gold_evidence", data.get("evidence", "")),
        "num_hops": data.get("num_hops", None),
        "label": data.get("label", None),
        "decomposition": decomposition,
        "question_plan": question_plan,
        "answer_result": answer_result,
        "answer_issues": answer_issues,
        "answer_used_fallback": any(item["status"] == "api_error" for item in answer_result["answers"]),
        "dependency_ablation": {
            "mode": "wo_dependency",
            "runtime_bindings_disabled": True,
            "bindings_update_ignored": True,
            "question_order_uses_rely_on": False,
        },
    }


def resolve_path(path, args):
    return (
        path
        .replace("[DATA]", args.dataset)
        .replace("[TYPE]", args.data_type)
        .replace("[CLASS]", args.class_num)
        .replace("[T]", args.t)
        .replace("[S]", str(args.start))
        .replace("[E]", str(args.end))
        .replace("[PLAN]", args.plan)
    )


def main(args):
    in_path = resolve_path(args.in_path, args)

    with open(in_path, "r", encoding="utf-8") as file_obj:
        raws = json.load(file_obj)
    raws = raws[args.start:args.end]

    dataset = []
    for data in raws:
        dataset.append({
            "id": data["id"],
            "claim": data["claim"],
            "gold_evidence": data.get("gold_evidence", data.get("evidence", "")),
            "num_hops": data.get("num_hops", None),
            "label": data.get("label", None),
            "decomposition": data.get("decomposition", {"claim": data["claim"], "atomic_facts": []}),
            "question_plan": data.get("question_plan", {"question_items": []}),
        })

    partial_func = partial(process_data_item, plan=args.plan, port=args.port)
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(partial_func, item) for item in dataset]
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
    parser.add_argument("--dataset", type=str, default="HOVER")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--class_num", type=str, default="2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=4000)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--plan", type=str, default="qc_plan6.4")
    parser.add_argument("--port", type=str, default="8370")
    parser.add_argument("--t", type=str, default="")
    parser.add_argument("--in_path", type=str, default="./data/[DATA]/[PLAN]/ablations/[TYPE]_[CLASS]_wo_dependency_questions_[T][S]_[E].json")
    parser.add_argument("--out_path", type=str, default="./data/[DATA]/[PLAN]/ablations/[TYPE]_[CLASS]_wo_dependency_answers_[T][S]_[E].json")
    main(parser.parse_args())
