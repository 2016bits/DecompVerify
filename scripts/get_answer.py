import json
import argparse
import re
import os

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from prompts.answer_prompts import get_answer_prompt


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
    if plan in {"local", "localhost", "vllm"} or "plan" in plan:
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

    # 4) Azure OpenAI GPT-4o
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


def parse_answer_output(output_text):
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


def fill_text(text, bindings):
    if not isinstance(text, str):
        return text
    for var, value in bindings.items():
        text = text.replace(var, value)
    return text


def fill_fact(fact, bindings):
    if fact is None:
        fact = {}
    return {
        "id": fact.get("id", ""),
        "subject": fill_text(fact.get("subject", ""), bindings),
        "predicate": fact.get("predicate", ""),
        "object": fill_text(fact.get("object", ""), bindings),
        "role": fact.get("role", "verify"),
        "depends_on": fact.get("depends_on", []) if isinstance(fact.get("depends_on", []), list) else []
    }


def normalize_answer_result(parsed, atomic_fact, selected_question):
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Answer output is not a dict: {parsed}")

    fact_id = parsed.get("fact_id", atomic_fact["id"])
    if fact_id is None:
        fact_id = atomic_fact["id"]
    fact_id = str(fact_id).strip()

    question = parsed.get("question", selected_question)
    if question is None:
        question = selected_question
    question = str(question).strip()

    answer = parsed.get("answer", "")
    if answer is None:
        answer = ""
    answer = str(answer).strip()

    status = parsed.get("status", "insufficient")
    if status is None:
        status = "insufficient"
    status = str(status).strip().lower()

    evidence_span = parsed.get("evidence_span", "")
    if evidence_span is None:
        evidence_span = ""
    evidence_span = str(evidence_span).strip()

    bindings_update = parsed.get("bindings_update", {})
    if not isinstance(bindings_update, dict):
        bindings_update = {}

    if status not in {"supported", "contradicted", "insufficient"}:
        status = "insufficient"

    normalized_bindings = {}
    for k, v in bindings_update.items():
        if isinstance(k, str) and k.startswith("?") and v is not None:
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


def choose_question(question_plan_item, filled_fact):
    q = question_plan_item.get("question_templates", {})
    role = question_plan_item.get("role", "verify")

    subject = filled_fact.get("subject", "")
    obj = filled_fact.get("object", "")

    subject_is_var = isinstance(subject, str) and subject.startswith("?")
    object_is_var = isinstance(obj, str) and obj.startswith("?")
    unresolved = subject_is_var or object_is_var

    if role == "bridge":
        if unresolved:
            return q.get("primary", "")
        if q.get("boolean"):
            return q["boolean"]
        if q.get("boolean_template"):
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
    subject = filled_fact.get("subject", "")
    obj = filled_fact.get("object", "")

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
        "fact_id": atomic_fact.get("id", ""),
        "question": selected_question,
        "answer": "",
        "status": "insufficient",
        "evidence_span": "",
        "bindings_update": {}
    }


def answer_one_fact(claim, evidence, atomic_fact, question_plan_item, bindings, plan, port):
    filled_fact = fill_fact(atomic_fact, bindings)
    selected_question = choose_question(question_plan_item, filled_fact)

    infer_count = 0
    last_error = None

    while infer_count < 3:
        output = None
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
            output = llm(prompt, plan=plan, port=port)
            parsed = parse_answer_output(output)
            result = normalize_answer_result(parsed, atomic_fact, selected_question)
            validate_answer_result(result, atomic_fact)

            if not result["bindings_update"] and result["answer"]:
                inferred = infer_bindings_update(filled_fact, result["answer"])
                result["bindings_update"] = inferred

            return result, filled_fact
        except Exception as e:
            last_error = e
            preview = output[:500] if isinstance(output, str) else repr(output)
            print(f"[Retry {infer_count + 1}/3] Error during answering for {atomic_fact.get('id', '')}: {e}")
            print(f"[Output Preview] {preview}")
            infer_count += 1

    print(f"[Fallback] fact: {atomic_fact.get('id', '')}, last_error: {last_error}")
    return fallback_answer_result(atomic_fact, filled_fact, selected_question), filled_fact


def execute_answering(claim, evidence, decomposition, question_plan, plan, port):
    if decomposition is None:
        decomposition = {
            "claim": claim,
            "atomic_facts": [],
            "constraints": []
        }
    if question_plan is None:
        question_plan = []

    atomic_facts = decomposition.get("atomic_facts", [])
    ordered_atomic_facts = topo_sort_atomic_facts(atomic_facts)

    question_plan_map = {}
    for item in question_plan:
        if isinstance(item, dict) and "fact_id" in item:
            question_plan_map[item["fact_id"]] = item

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
            plan=plan,
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


def process_data_item(data, plan, port):
    claim = data["claim"]
    evidence = data["gold_evidence"]
    decomposition = data["decomposition"]
    question_plan = data["question_plan"]

    answer_plan, final_bindings = execute_answering(
        claim=claim,
        evidence=evidence,
        decomposition=decomposition,
        question_plan=question_plan,
        plan=plan,
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
    parser.add_argument('--plan', type=str, default='plan3.3', help='LLM plan: local/scnet/iflow/azure')

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
    parser.add_argument('--t', type=str, default='')

    args = parser.parse_args()
    main(args)