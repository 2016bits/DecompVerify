import json
import argparse
import re
import os

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from prompts.answer_prompts import get_answer_prompt

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


def parse_answer_output(output_text):
    json_text = extract_json_block(output_text)
    return json.loads(json_text)


def extract_variables(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", text))


def fact_variables(fact):
    return extract_variables(fact.get("subject", "")) | extract_variables(fact.get("object", ""))


def fill_text(text, bindings):
    if not isinstance(text, str):
        return text
    for var, value in bindings.items():
        text = text.replace(var, value)
    return text


def fill_fact(fact, bindings):
    return {
        "id": fact["id"],
        "subject": fill_text(fact["subject"], bindings),
        "predicate": fact["predicate"],
        "object": fill_text(fact["object"], bindings),
        "role": fact["role"],
        "depends_on": fact.get("depends_on", [])
    }


def normalize_answer_result(parsed, atomic_fact, selected_question):
    fact_id = parsed.get("fact_id", atomic_fact["id"])
    question = str(parsed.get("question", selected_question)).strip()
    answer = str(parsed.get("answer", "")).strip()
    status = str(parsed.get("status", "insufficient")).strip().lower()
    evidence_span = str(parsed.get("evidence_span", "")).strip()
    bindings_update = parsed.get("bindings_update", {})

    if not isinstance(bindings_update, dict):
        bindings_update = {}

    if status not in {"supported", "contradicted", "insufficient"}:
        status = "insufficient"

    normalized_bindings = {}
    for k, v in bindings_update.items():
        if isinstance(k, str) and k.startswith("?"):
            normalized_bindings[k] = str(v).strip()

    return {
        "fact_id": fact_id,
        "question": question,
        "answer": answer,
        "status": status,
        "evidence_span": evidence_span,
        "bindings_update": normalized_bindings
    }


def validate_answer_result(result, atomic_fact):
    if result["fact_id"] != atomic_fact["id"]:
        raise ValueError(f"fact_id mismatch: {result['fact_id']} vs {atomic_fact['id']}")

    if result["status"] not in {"supported", "contradicted", "insufficient"}:
        raise ValueError(f"invalid status: {result['status']}")

    return True


def topo_sort_atomic_facts(atomic_facts):
    # Kahn's algorithm for topological sorting
    fact_map = {fact["id"]: fact for fact in atomic_facts}
    indegree = {fact["id"]: 0 for fact in atomic_facts}
    graph = {fact["id"]: [] for fact in atomic_facts}

    for fact in atomic_facts:
        for dep in fact.get("depends_on", []):
            if dep in fact_map:
                graph[dep].append(fact["id"])
                indegree[fact["id"]] += 1

    queue = [fid for fid, deg in indegree.items() if deg == 0]
    queue.sort()

    order = []
    while queue:
        cur = queue.pop(0)
        order.append(fact_map[cur])
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
                queue.sort()

    if len(order) != len(atomic_facts):
        raise ValueError("Cycle detected or invalid dependency graph.")

    return order


def choose_question(question_plan_item, filled_fact):
    q = question_plan_item["question_templates"]
    role = question_plan_item["role"]

    subject = filled_fact["subject"]
    obj = filled_fact["object"]

    subject_is_var = isinstance(subject, str) and subject.startswith("?")
    object_is_var = isinstance(obj, str) and obj.startswith("?")
    unresolved = subject_is_var or object_is_var

    if role == "bridge":
        if unresolved:
            return q.get("primary", "")
        if q.get("boolean"):
            return q["boolean"]
        if q.get("boolean_template"):
            # 尝试把当前已知实体填入 {value}
            known_value = subject if not subject_is_var else obj
            return q["boolean_template"].replace("{value}", known_value)
        return q.get("alternate") or q.get("primary", "")

    if role == "verify":
        if q.get("boolean"):
            return q["boolean"]
        if q.get("boolean_template"):
            known_value = subject if not subject_is_var else obj
            return q["boolean_template"].replace("{value}", known_value)
        return q.get("alternate") or q.get("primary", "")

    if role == "anchor":
        return q.get("primary", "")

    return q.get("primary", "")


def infer_bindings_update(filled_fact, answer):
    """
    当模型未显式返回 bindings_update 时，尝试根据 filled_fact 自动推断。
    """
    update = {}
    subject = filled_fact["subject"]
    obj = filled_fact["object"]

    if not isinstance(answer, str):
        return update

    ans = answer.strip()
    if not ans:
        return update

    if isinstance(subject, str) and subject.startswith("?") and not (isinstance(obj, str) and obj.startswith("?")):
        update[subject] = ans
    elif isinstance(obj, str) and obj.startswith("?") and not (isinstance(subject, str) and subject.startswith("?")):
        update[obj] = ans

    return update


def fallback_answer_result(atomic_fact, filled_fact, selected_question):
    return {
        "fact_id": atomic_fact["id"],
        "question": selected_question,
        "answer": "",
        "status": "insufficient",
        "evidence_span": "",
        "bindings_update": {}
    }


def answer_one_fact(claim, evidence, atomic_fact, question_plan_item, bindings, port):
    filled_fact = fill_fact(atomic_fact, bindings)
    selected_question = choose_question(question_plan_item, filled_fact)

    infer_count = 0
    last_error = None

    while infer_count < 3:
        try:
            prompt = get_answer_prompt(
                claim=claim,
                evidence=evidence,
                atomic_fact=atomic_fact,
                filled_fact=filled_fact,
                question_plan=question_plan_item,
                bindings=bindings,
                question=selected_question
            )
            output = llm(prompt, port)
            parsed = parse_answer_output(output)
            result = normalize_answer_result(parsed, atomic_fact, selected_question)
            validate_answer_result(result, atomic_fact)

            if not result["bindings_update"] and result["answer"]:
                inferred = infer_bindings_update(filled_fact, result["answer"])
                result["bindings_update"] = inferred

            return result, filled_fact
        except Exception as e:
            last_error = e
            print(f"[Retry {infer_count + 1}/3] Error during answering for {atomic_fact['id']}: {e}")
            infer_count += 1

    print(f"[Fallback] fact: {atomic_fact['id']}, last_error: {last_error}")
    return fallback_answer_result(atomic_fact, filled_fact, selected_question), filled_fact


def execute_answering(claim, evidence, decomposition, question_plan, port):
    atomic_facts = decomposition.get("atomic_facts", [])
    ordered_atomic_facts = topo_sort_atomic_facts(atomic_facts)

    question_plan_map = {item["fact_id"]: item for item in question_plan}

    bindings = {}
    answer_plan = []

    for atomic_fact in ordered_atomic_facts:
        if atomic_fact["id"] not in question_plan_map:
            raise ValueError(f"Missing question_plan for fact {atomic_fact['id']}")

        question_item = question_plan_map[atomic_fact["id"]]
        result, filled_fact = answer_one_fact(
            claim=claim,
            evidence=evidence,
            atomic_fact=atomic_fact,
            question_plan_item=question_item,
            bindings=bindings,
            port=port
        )

        bindings.update(result.get("bindings_update", {}))

        answer_plan.append({
            "fact_id": atomic_fact["id"],
            "filled_fact": filled_fact,
            "question_used": result["question"],
            "answer": result["answer"],
            "status": result["status"],
            "evidence_span": result["evidence_span"],
            "bindings_update": result["bindings_update"]
        })

    return answer_plan, bindings


def process_data_item(data, port):
    claim = data["claim"]
    evidence = data["gold_evidence"]
    decomposition = data["decomposition"]
    question_plan = data["question_plan"]

    answer_plan, final_bindings = execute_answering(
        claim=claim,
        evidence=evidence,
        decomposition=decomposition,
        question_plan=question_plan,
        port=port
    )

    return {
        "id": data["id"],
        "claim": claim,
        "gold_evidence": evidence,
        "num_hops": data["num_hops"],
        "label": data["label"],
        "decomposition": decomposition,
        "question_plan": question_plan,
        "answer_plan": answer_plan,
        "final_bindings": final_bindings
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

    dataset = []
    for data in raws:
        decomposition = data.get("decomposition", None)
        question_plan = data.get("question_plan", None)

        # 兼容旧格式：decomposition 可能还放在 atomic_facts 字段里
        if decomposition is None and "atomic_facts" in data and isinstance(data["atomic_facts"], dict):
            decomposition = data["atomic_facts"]

        if decomposition is None:
            decomposition = {
                "claim": data["claim"],
                "atomic_facts": [],
                "constraints": []
            }

        if question_plan is None:
            question_plan = []

        dataset.append({
            "id": data["id"],
            "claim": data["claim"],
            "gold_evidence": data.get("gold_evidence", data.get("evidence", "")),
            "num_hops": data.get("num_hops", None),
            "label": data.get("label", None),
            "decomposition": decomposition,
            "question_plan": question_plan
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
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_question_[T][S]_[E].json',
        help='Input path template'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_answer_[T][S]_[E].json',
        help='Output path template'
    )
    parser.add_argument('--plan', type=str, default='plan3', help='Plan name to distinguish different question generation strategies')
    parser.add_argument('--t', type=str, default='')

    args = parser.parse_args()
    main(args)
