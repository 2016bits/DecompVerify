import json
import argparse
import re
import os

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from prompts.question_prompts import get_question_prompt


def build_client_and_model(plan, port='8370'):
    """
    根据 plan 自动构建 client 和 model
    支持:
      - local
      - scnet
      - iflow
      - scnet_qwen235b
      - iflow_qwen32b
      - azure
      - azure_gpt4o
    """
    plan = (plan or "").lower()

    # 1) 本地 OpenAI-compatible 服务
    if plan in {"local", "localhost", "vllm"} or plan.startswith("plan"):
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1"
        )
        model = "Meta-Llama-3-70B-Instruct-AutoAWQ-4bit"
        system_prompt = None
        extra_body = None
        max_tokens = None
        temperature = 0.0
        return client, model, system_prompt, extra_body, max_tokens, temperature

    # 2) SCNet
    elif plan in {"scnet", "scnet_qwen235b"}:
        api_key = os.getenv("SCNET_API_KEY")
        if not api_key:
            raise ValueError("SCNET_API_KEY is not set in environment variables.")

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.scnet.cn/api/llm/v1"
        )
        model = "Qwen3-235B-A22B"
        system_prompt = "You are a helpful assistant"
        extra_body = None
        max_tokens = 512
        temperature = 0.0
        return client, model, system_prompt, extra_body, max_tokens, temperature

    # 3) iFlow
    elif plan in {"iflow", "iflow_qwen32b"}:
        api_key = os.getenv("IFLOW_API_KEY")
        if not api_key:
            raise ValueError("IFLOW_API_KEY is not set in environment variables.")

        client = OpenAI(
            base_url="https://apis.iflow.cn/v1",
            api_key=api_key,
        )
        model = "qwen3-32b"
        system_prompt = None
        extra_body = {}
        max_tokens = 512
        temperature = 0.0
        return client, model, system_prompt, extra_body, max_tokens, temperature
    
    # azure openai gpt-4o
    elif plan in {"azure", "azure_gpt4o"}:
        api_key = os.getenv("AZURE_API_KEY")
        if not api_key:
            raise ValueError("AZURE_API_KEY is not set in environment variables.")
        azure_endpoint = "https://open1027.openai.azure.com/openai/v1/"
        model = "gpt-4o"
        
        client = OpenAI(
            api_key=api_key,
            base_url=azure_endpoint
        )
        system_prompt = None
        extra_body = {}
        max_tokens = 512
        temperature = 0.7
        return client, model, system_prompt, extra_body, max_tokens, temperature
    
    elif plan.startswith("qwen_plan"):
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1"
        )
        model = "gpt-4o"
        system_prompt = "You are a helpful assistant."
        extra_body = None
        max_tokens = None
        temperature = 0.7
        return client, model, system_prompt, extra_body, max_tokens, temperature

    else:
        raise ValueError(
            f"Unsupported plan: {plan}. "
            f"Supported plans are: local, scnet, iflow, scnet_qwen235b, iflow_qwen32b, azure, azure_gpt4o"
        )


def llm(input_text, plan='local', port='8370'):
    client, model, system_prompt, extra_body, max_tokens, temperature = build_client_and_model(plan, port)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": input_text})

    kwargs = {
        "model": model,
        "messages": messages,
    }

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
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
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
    if text is None:
        raise ValueError("Model output is None.")
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        raise ValueError("Model output is empty.")

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
        raise ValueError(f"No JSON object or array found in model output:\n{text[:500]}")

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

    raise ValueError(f"Incomplete JSON block in model output:\n{text[:500]}")


def parse_question_output(output_text):
    json_text = extract_json_block(output_text)
    data = json.loads(json_text)
    if not isinstance(data, dict):
        raise ValueError(f"Parsed JSON is not a dict: {type(data)}")
    return data


def extract_variables(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", text))


def fact_variables(fact):
    return extract_variables(fact.get("subject", "")) | extract_variables(fact.get("object", ""))


def normalize_question_templates(q):
    if q is None:
        q = {}
    if not isinstance(q, dict):
        q = {}

    primary = "" if q.get("primary") is None else str(q.get("primary")).strip()
    alternate = "" if q.get("alternate") is None else str(q.get("alternate")).strip()
    boolean = "" if q.get("boolean") is None else str(q.get("boolean")).strip()
    boolean_template = "" if q.get("boolean_template") is None else str(q.get("boolean_template")).strip()

    return {
        "primary": primary,
        "alternate": alternate,
        "boolean": boolean,
        "boolean_template": boolean_template
    }


def normalize_question_plan(parsed, atomic_fact):
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Question output is not a dict: {parsed}")

    fact_id = parsed.get("fact_id", atomic_fact["id"])
    role = parsed.get("role", atomic_fact["role"])
    question_templates = normalize_question_templates(parsed.get("question_templates", {}))

    if fact_id is None:
        fact_id = atomic_fact["id"]
    if role is None:
        role = atomic_fact["role"]

    fact_id = str(fact_id).strip()
    role = str(role).strip().lower()

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


def generate_questions_for_fact(claim, atomic_fact, plan, port):
    prompt = get_question_prompt(claim, atomic_fact)

    infer_count = 0
    last_error = None

    while infer_count < 3:
        output = None
        try:
            output = llm(prompt, plan=plan, port=port)
            parsed = parse_question_output(output)
            plan_result = normalize_question_plan(parsed, atomic_fact)
            validate_question_plan(plan_result, atomic_fact)
            return plan_result
        except Exception as e:
            last_error = e
            preview = output[:500] if isinstance(output, str) else repr(output)
            print(f"[Retry {infer_count + 1}/3] Error during question generation for {atomic_fact['id']}: {e}")
            print(f"[Output Preview] {preview}")
            infer_count += 1

    print(f"[Fallback] fact: {atomic_fact['id']}, last_error: {last_error}")
    return fallback_question_plan(claim, atomic_fact)


def topo_sort_atomic_facts(atomic_facts):
    if atomic_facts is None:
        atomic_facts = []
    if not isinstance(atomic_facts, list):
        raise ValueError("atomic_facts must be a list.")

    clean_atomic_facts = []
    for idx, fact in enumerate(atomic_facts):
        if fact is None:
            print(f"[Warn] Skip None atomic fact at idx={idx}")
            continue
        if not isinstance(fact, dict):
            print(f"[Warn] Skip non-dict atomic fact at idx={idx}: {fact}")
            continue
        if "id" not in fact:
            print(f"[Warn] Skip atomic fact without id at idx={idx}: {fact}")
            continue
        if "depends_on" not in fact or fact["depends_on"] is None:
            fact["depends_on"] = []
        clean_atomic_facts.append(fact)

    fact_map = {fact["id"]: fact for fact in clean_atomic_facts}
    indegree = {fact["id"]: 0 for fact in clean_atomic_facts}
    graph = {fact["id"]: [] for fact in clean_atomic_facts}

    for fact in clean_atomic_facts:
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

    if len(order) != len(clean_atomic_facts):
        raise ValueError("Cycle detected or invalid dependency graph.")

    return order


def generate_question_plan(claim, decomposition, plan, port):
    if decomposition is None:
        decomposition = {
            "claim": claim,
            "atomic_facts": [],
            "constraints": []
        }

    atomic_facts = decomposition.get("atomic_facts", [])
    ordered_atomic_facts = topo_sort_atomic_facts(atomic_facts)

    question_plan = []
    for atomic_fact in ordered_atomic_facts:
        plan_result = generate_questions_for_fact(claim, atomic_fact, plan, port)
        question_plan.append(plan_result)

    return question_plan


def process_data_item(data, plan, port):
    claim = data["claim"]
    decomposition = data["decomposition"]
    question_plan = generate_question_plan(claim, decomposition, plan, port)

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
        .replace("[PLAN]", args.plan)
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

    partial_func = partial(process_data_item, plan=args.plan, port=args.port)
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
    parser.add_argument('--end', type=int, default=200, help='End index')
    parser.add_argument('--port', type=str, default='8370', help='Port for local LLM API')
    parser.add_argument('--max_workers', type=int, default=8, help='Number of threads')
    parser.add_argument('--plan', type=str, default='plan3.3', help='LLM plan: local/scnet/iflow')

    parser.add_argument(
        '--in_path',
        type=str,
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_decomposed_[T][S]_[E].json',
        help='Input path template'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_question_[T][S]_[E].json',
        help='Output path template'
    )
    parser.add_argument('--t', type=str, default='')

    args = parser.parse_args()
    main(args)