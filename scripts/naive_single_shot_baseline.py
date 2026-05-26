"""Naive single-shot baseline for the ablation study.

Hands the LLM the full claim plus the gold evidence and asks for a binary
supports/refutes decision in one call -- no decomposition, no question plan,
no per-fact verification, no aggregation. The output schema matches
aggregate_labels.py so evaluate.py can score it without changes.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial

from tqdm import tqdm

import get_answer


PROMPT_TEMPLATE = """You are a fact-checking assistant.

Given a CLAIM and an EVIDENCE passage, decide whether the evidence SUPPORTS the claim or REFUTES it. Use only the evidence provided. If the evidence does not fully support every part of the claim, answer "refutes".

CLAIM:
{claim}

EVIDENCE:
{evidence}

Respond with a single JSON object on one line:
{{"label": "supports" | "refutes", "reason": "<one sentence>"}}
"""


def _normalize_label(text):
    text = (text or "").strip().lower()
    if text in {"supports", "support", "supported", "true", "yes"}:
        return "supports"
    if text in {"refutes", "refute", "refuted", "contradicts", "contradicted",
                "not enough information", "nei", "insufficient", "false", "no"}:
        return "refutes"
    return None


def _parse_response(content):
    try:
        block = get_answer.extract_json_block(content)
        obj = json.loads(block)
        label = _normalize_label(obj.get("label"))
        reason = str(obj.get("reason", "")).strip()
    except Exception:
        # Fallback: scan the raw text for the first label-like token.
        match = re.search(r"\b(supports?|refutes?|contradicts?|insufficient)\b", content or "", flags=re.IGNORECASE)
        label = _normalize_label(match.group(1)) if match else None
        reason = (content or "").strip()[:300]
    if label is None:
        label = "refutes"
        reason = (reason + " [fallback: unparseable label, defaulted to refutes]").strip()
    return label, reason


def process_item(item, plan, port, max_retries=2):
    claim = (item.get("claim") or "").strip()
    evidence = (item.get("gold_evidence") or item.get("evidence") or "").strip()
    prompt = PROMPT_TEMPLATE.format(claim=claim, evidence=evidence)

    label = None
    reason = ""
    last_error = None
    raw = ""
    for attempt in range(max_retries + 1):
        try:
            raw = get_answer.llm(prompt, plan=plan, port=port)
            label, reason = _parse_response(raw)
            break
        except Exception as exc:
            last_error = exc

    if label is None:
        label = "refutes"
        reason = f"API failure after {max_retries + 1} attempts: {last_error}"

    out = dict(item)
    out["aggregation_result"] = {
        "final_label": label,
        "final_label_binary": label,
        "decision_reason": reason,
        "aggregation_mode": "naive_single_shot",
        "raw_response": raw,
    }
    return out


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

    with open(in_path, "r", encoding="utf-8") as file_obj:
        raws = json.load(file_obj)
    raws = raws[args.start:args.end]

    partial_func = partial(process_item, plan=args.plan, port=args.port, max_retries=args.max_retries)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(partial_func, data) for data in raws]
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
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--plan", type=str, default="qc_plan")
    parser.add_argument("--port", type=str, default="8370")
    parser.add_argument("--t", type=str, default="")
    parser.add_argument("--in_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_decomposed_[T][S]_[E].json")
    parser.add_argument("--out_path", type=str, default="./data/[DATA]/[PLAN]/ablations_[S]_[E]/naive_single_shot_aggregate.json")
    main(parser.parse_args())
