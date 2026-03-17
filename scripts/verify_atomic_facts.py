import json
import argparse
import re
import os

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from prompts.verify_atomic_facts_prompts import get_verify_prompt


def llm(input_text, port='8370'):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:PORT/v1".replace('PORT', port)
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base
    )

    predictions = client.chat.completions.create(
        model="Meta-Llama-3-70B-Instruct-AutoAWQ-4bit",
        messages=[
            {"role": "user", "content": input_text},
        ],
        temperature=0.0
    )
    return predictions.choices[0].message.content


def extract_json_block(text):
    text = text.strip()

    code_block_pattern = r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```"
    match = re.search(code_block_pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    first_brace = text.find("{")
    first_bracket = text.find("[")

    candidates = []
    if first_brace != -1:
        candidates.append((first_brace, "{", "}"))
    if first_bracket != -1:
        candidates.append((first_bracket, "[", "]"))

    if not candidates:
        raise ValueError("No JSON object or array found in model output.")

    candidates.sort(key=lambda x: x[0])
    start_idx, open_ch, close_ch = candidates[0]

    stack = 0
    in_string = False
    escape = False
    for i in range(start_idx, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == open_ch:
                stack += 1
            elif ch == close_ch:
                stack -= 1
                if stack == 0:
                    return text[start_idx:i + 1].strip()

    raise ValueError("Incomplete JSON block in model output.")


def parse_verify_output(output_text):
    json_text = extract_json_block(output_text)
    return json.loads(json_text)


def normalize_verification_result(parsed, answer_item):
    fact_id = parsed.get("fact_id", answer_item["fact_id"])
    verification_label = str(parsed.get("verification_label", "insufficient")).strip().lower()
    reason = str(parsed.get("reason", "")).strip()
    evidence_span = str(parsed.get("evidence_span", "")).strip()

    if verification_label not in {"support", "contradict", "insufficient"}:
        verification_label = "insufficient"

    return {
        "fact_id": fact_id,
        "filled_fact": answer_item["filled_fact"],
        "verification_label": verification_label,
        "reason": reason,
        "evidence_span": evidence_span
    }


def validate_verification_result(result, answer_item):
    if result["fact_id"] != answer_item["fact_id"]:
        raise ValueError(f"fact_id mismatch: {result['fact_id']} vs {answer_item['fact_id']}")

    if result["verification_label"] not in {"support", "contradict", "insufficient"}:
        raise ValueError(f"invalid verification_label: {result['verification_label']}")

    return True


def fallback_verification_result(answer_item):
    """
    如果第四步验证失败，则用第三步 answering 的 status 做保守映射
    """
    prev_status = str(answer_item.get("status", "insufficient")).strip().lower()

    if prev_status == "supported":
        label = "support"
    elif prev_status == "contradicted":
        label = "contradict"
    else:
        label = "insufficient"

    return {
        "fact_id": answer_item["fact_id"],
        "filled_fact": answer_item["filled_fact"],
        "verification_label": label,
        "reason": "Fallback from previous answering status.",
        "evidence_span": answer_item.get("evidence_span", "")
    }


def verify_one_fact(claim, evidence, answer_item, port):
    infer_count = 0
    last_error = None

    while infer_count < 3:
        try:
            prompt = get_verify_prompt(claim, evidence, answer_item)
            output = llm(prompt, port)
            parsed = parse_verify_output(output)
            result = normalize_verification_result(parsed, answer_item)
            validate_verification_result(result, answer_item)
            return result
        except Exception as e:
            last_error = e
            print(f"[Retry {infer_count + 1}/3] Error during fact verification for {answer_item['fact_id']}: {e}")
            infer_count += 1

    print(f"[Fallback] fact: {answer_item['fact_id']}, last_error: {last_error}")
    return fallback_verification_result(answer_item)


def verify_all_facts(claim, evidence, answer_plan, port):
    fact_verification = []

    for answer_item in answer_plan:
        result = verify_one_fact(
            claim=claim,
            evidence=evidence,
            answer_item=answer_item,
            port=port
        )
        fact_verification.append(result)

    return fact_verification


def process_data_item(data, port):
    claim = data["claim"]
    evidence = data["gold_evidence"]

    decomposition = data.get("decomposition", None)
    if decomposition is None and "atomic_facts" in data and isinstance(data["atomic_facts"], dict):
        decomposition = data["atomic_facts"]
    if decomposition is None:
        decomposition = {
            "claim": claim,
            "atomic_facts": [],
            "constraints": []
        }

    question_plan = data.get("question_plan", [])
    answer_plan = data.get("answer_plan", [])
    final_bindings = data.get("final_bindings", {})

    fact_verification = verify_all_facts(
        claim=claim,
        evidence=evidence,
        answer_plan=answer_plan,
        port=port
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
        "fact_verification": fact_verification
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

    print(f"Loading data from {in_path}...")
    with open(in_path, "r", encoding="utf-8") as f:
        raws = json.load(f)

    # 只处理指定范围
    # raws = raws[args.start: args.end]

    dataset = []
    for data in raws:
        dataset.append({
            "id": data["id"],
            "claim": data["claim"],
            "gold_evidence": data.get("gold_evidence", data.get("evidence", "")),
            "num_hops": data.get("num_hops", None),
            "label": data.get("label", None),
            "decomposition": data.get("decomposition", data.get("atomic_facts", None)),
            "question_plan": data.get("question_plan", []),
            "answer_plan": data.get("answer_plan", []),
            "final_bindings": data.get("final_bindings", {})
        })

    partial_func = partial(process_data_item, port=args.port)
    results = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(partial_func, data) for data in dataset]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Error in future: {e}")
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
    parser.add_argument('--end', type=int, default=2000, help='End index')
    parser.add_argument('--port', type=str, default='8370', help='Port for LLM API')
    parser.add_argument('--max_workers', type=int, default=16, help='Number of threads')

    parser.add_argument(
        '--in_path',
        type=str,
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_answer_[T][S]_[E].json',
        help='Input path template'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_fact_verify_[T][S]_[E].json',
        help='Output path template'
    )
    parser.add_argument('--plan', type=str, default='plan3', help='Plan name to distinguish different question generation strategies')
    parser.add_argument('--t', type=str, default='')

    args = parser.parse_args()
    main(args)
    