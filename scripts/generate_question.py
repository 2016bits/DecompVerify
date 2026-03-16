import json
import argparse
import re
import os

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from prompts.question_prompts import get_question_prompt

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


def parse_question_output(output_text):
    json_text = extract_json_block(output_text)
    return json.loads(json_text)


def extract_variables(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", text))


def fact_variables(fact):
    return extract_variables(fact.get("subject", "")) | extract_variables(fact.get("object", ""))


def normalize_question_templates(q):
    if not isinstance(q, dict):
        q = {}

    primary = str(q.get("primary", "")).strip()
    alternate = str(q.get("alternate", "")).strip()
    boolean = str(q.get("boolean", "")).strip()
    boolean_template = str(q.get("boolean_template", "")).strip()

    return {
        "primary": primary,
        "alternate": alternate,
        "boolean": boolean,
        "boolean_template": boolean_template
    }


def normalize_question_plan(parsed, atomic_fact):
    fact_id = parsed.get("fact_id", atomic_fact["id"])
    role = parsed.get("role", atomic_fact["role"])
    question_templates = normalize_question_templates(parsed.get("question_templates", {}))

    if role not in {"bridge", "verify", "anchor"}:
        role = atomic_fact["role"]

    return {
        "fact_id": fact_id,
        "role": role,
        "question_templates": question_templates
    }


def validate_question_plan(plan, atomic_fact):
    if plan["fact_id"] != atomic_fact["id"]:
        raise ValueError(f"fact_id mismatch: {plan['fact_id']} vs {atomic_fact['id']}")

    if plan["role"] != atomic_fact["role"]:
        raise ValueError(f"role mismatch: {plan['role']} vs {atomic_fact['role']}")

    q = plan["question_templates"]

    if not q["primary"]:
        raise ValueError(f"primary question is empty for fact {atomic_fact['id']}")

    # bridge 通常应包含变量定位能力
    if atomic_fact["role"] == "bridge":
        if not q["boolean_template"] and not q["alternate"]:
            raise ValueError(f"bridge fact {atomic_fact['id']} lacks boolean_template/alternate question")

    # verify 通常应有可判断问题
    if atomic_fact["role"] == "verify":
        if not q["boolean"] and not q["boolean_template"]:
            # 允许有些 verify 主问题是 span 型，但这里给一点约束
            pass

    return True


def fallback_question_plan(claim, atomic_fact):
    subject = atomic_fact["subject"]
    predicate = atomic_fact["predicate"]
    obj = atomic_fact["object"]
    role = atomic_fact["role"]
    vars_in_fact = fact_variables(atomic_fact)

    primary = f"What is the relation in this fact: {subject} {predicate} {obj}?"
    alternate = ""
    boolean = ""
    boolean_template = ""

    if role == "bridge":
        if obj.startswith("?"):
            primary = f"Who or what did {subject} {predicate}?"
            alternate = f"What is the missing value in: {subject} {predicate} {obj}?"
            boolean_template = f"Did {subject} {predicate} {{value}}?"
        elif subject.startswith("?"):
            primary = f"Who or what {predicate} {obj}?"
            alternate = f"What is the missing subject in: {subject} {predicate} {obj}?"
            boolean_template = f"Did {{value}} {predicate} {obj}?"
    elif role == "verify":
        primary = f"What evidence shows that {subject} {predicate} {obj}?"
        boolean = f"Is it true that {subject} {predicate} {obj}?"
        if vars_in_fact:
            boolean_template = f"Is it true that {subject} {predicate} {obj}?"
    elif role == "anchor":
        primary = f"What is the missing value in: {subject} {predicate} {obj}?"
        alternate = f"Find the value for: {subject} {predicate} {obj}."

    return {
        "fact_id": atomic_fact["id"],
        "role": role,
        "question_templates": {
            "primary": primary,
            "alternate": alternate,
            "boolean": boolean,
            "boolean_template": boolean_template
        }
    }


def generate_questions_for_fact(claim, atomic_fact, port):
    prompt = get_question_prompt(claim, atomic_fact)

    infer_count = 0
    last_error = None

    while infer_count < 3:
        try:
            output = llm(prompt, port)
            parsed = parse_question_output(output)
            plan = normalize_question_plan(parsed, atomic_fact)
            validate_question_plan(plan, atomic_fact)
            return plan
        except Exception as e:
            last_error = e
            print(f"[Retry {infer_count + 1}/3] Error during question generation for {atomic_fact['id']}: {e}")
            infer_count += 1

    print(f"[Fallback] fact: {atomic_fact['id']}, last_error: {last_error}")
    return fallback_question_plan(claim, atomic_fact)


def topo_sort_atomic_facts(atomic_facts):
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


def generate_question_plan(claim, decomposition, port):
    atomic_facts = decomposition.get("atomic_facts", [])
    ordered_atomic_facts = topo_sort_atomic_facts(atomic_facts)

    question_plan = []
    for atomic_fact in ordered_atomic_facts:
        plan = generate_questions_for_fact(claim, atomic_fact, port)
        question_plan.append(plan)

    return question_plan


def process_data_item(data, port):
    claim = data["claim"]
    decomposition = data["decomposition"]
    question_plan = generate_question_plan(claim, decomposition, port)

    return {
        "id": data["id"],
        "claim": claim,
        "gold_evidence": data["gold_evidence"],
        "num_hops": data["num_hops"],
        "label": data["label"],
        "decomposition": decomposition,
        "question_plan": question_plan
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
    )

    with open(in_path, "r", encoding="utf-8") as f:
        raws = json.load(f)

    dataset = []
    for data in raws:
        decomposition = data.get("decomposition", None)

        if decomposition is None and "atomic_facts" in data and isinstance(data["atomic_facts"], dict):
            decomposition = data["atomic_facts"]

        if decomposition is None:
            decomposition = {
                "claim": data["claim"],
                "atomic_facts": [],
                "constraints": []
            }

        dataset.append({
            "id": data["id"],
            "claim": data["claim"],
            "gold_evidence": data.get("gold_evidence", data.get("evidence", "")),
            "num_hops": data.get("num_hops", None),
            "label": data.get("label", None),
            "decomposition": decomposition
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
        default='./data/[DATA]/plan2/[TYPE]_[CLASS]_decomposed_[S]_[E].json',
        help='Input path template'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='./data/[DATA]/plan2/[TYPE]_[CLASS]_question_[T][S]_[E].json',
        help='Output path template'
    )
    parser.add_argument('--t', type=str, default='')

    args = parser.parse_args()
    main(args)