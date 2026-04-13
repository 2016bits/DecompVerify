import json
import argparse
import re
import os
import time
import random
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from datetime import datetime

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
    
    elif plan.startswith("qwen_plan"):
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1"
        )
        model = "qwen2.5-72b-awq"
        system_prompt = "You are a helpful assistant."
        extra_body = None
        max_tokens = None
        temperature = 0.7
        return client, model, system_prompt, extra_body, max_tokens, temperature
    
    elif plan.startswith("qc_plan"):
        api_key = os.getenv("AIPING_API_KEY")
        if not api_key:
            raise ValueError("AIPING_API_KEY is not set in environment variables.")
        azure_endpoint = "https://www.aiping.cn/api/v1"
        model = "DeepSeek-V3.2"
        
        client = OpenAI(
            api_key=api_key,
            base_url=azure_endpoint
        )
        system_prompt = None
        extra_body = {}
        max_tokens = 2048
        temperature = 0.7
        return client, model, system_prompt, extra_body, max_tokens, temperature

    else:
        raise ValueError(
            f"Unsupported plan: {plan}. "
            f"Supported plans are: local, azure, azure_gpt4o, qc_plan, qwen_plan"
        )


def llm(input_text, plan='local', port='8370'):
    client, model, system_prompt, extra_body, max_tokens, temperature = build_client_and_model(plan, port)

    # simple throttle to reduce 429 rate-limit errors
    # time.sleep(1.5)

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
    if plan.startswith('qc_plan') or plan.startswith('azure'):
        time.sleep(1.2)

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


def _clean_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_json_block(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

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


def replace_placeholders(text, bindings):
    text = _clean_text(text)
    if not text:
        return text
    for var in sorted(bindings.keys(), key=len, reverse=True):
        value = _clean_text(bindings[var])
        if var.startswith("?") and value:
            text = text.replace(var, value)
    return text


def replace_placeholders_in_obj(obj, bindings):
    if isinstance(obj, str):
        return replace_placeholders(obj, bindings)
    if isinstance(obj, list):
        return [replace_placeholders_in_obj(x, bindings) for x in obj]
    if isinstance(obj, dict):
        return {k: replace_placeholders_in_obj(v, bindings) for k, v in obj.items()}
    return obj


def unresolved_vars(text):
    return sorted(set(re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", _clean_text(text))))


def merge_bindings(current_bindings, new_bindings):
    merged = dict(current_bindings)
    for k, v in (new_bindings or {}).items():
        k = _clean_text(k)
        v = _clean_text(v)
        if k.startswith("?") and v:
            merged[k] = v
    return merged


def infer_answer_slot_from_question(question_item):
    slot = question_item.get("answer_slot", None)
    if slot:
        return _clean_text(slot)

    q = _clean_text(question_item.get("main_question", "")).lower()
    mapping = [
        (r"what team", "?team"),
        (r"who is an ambassador", "?ambassador"),
        (r"who is the ambassador", "?ambassador"),
        (r"what is the home ground", "?home_ground"),
        (r"which film", "?film"),
        (r"what film", "?film"),
        (r"what animal", "?animal"),
        (r"where was .* born", "?village"),
        (r"in which village .* born", "?village"),
    ]
    for pat, slot in mapping:
        if re.search(pat, q):
            return slot
    return None


def normalize_yesno_by_question_type(question_type, answer, evidence_span):
    answer = _clean_text(answer)
    lower = answer.lower()

    if question_type == "relation_yesno":
        if lower in {"yes", "no"}:
            return answer
        if lower.startswith("yes"):
            return "Yes"
        if lower.startswith("no"):
            return "No"
        if answer and evidence_span and lower != "insufficient":
            return "Yes"

    return answer


def normalize_bindings_update(parsed, question_item, question_type, answer):
    bindings_update = parsed.get("bindings_update", {})
    if not isinstance(bindings_update, dict):
        bindings_update = {}

    answer_slot = infer_answer_slot_from_question(question_item)
    if question_type == "entity_wh" and answer_slot and _clean_text(answer).lower() != "insufficient":
        bindings_update.setdefault(answer_slot, _clean_text(answer))

        if answer_slot == "?home_ground":
            bindings_update.setdefault("?stadium", _clean_text(answer))
        if answer_slot == "?stadium":
            bindings_update.setdefault("?home_ground", _clean_text(answer))

    q_l = _clean_text(question_item.get("main_question", "")).lower()
    if "ambassador" in q_l and _clean_text(answer).lower() != "insufficient":
        bindings_update.setdefault("?ambassador", _clean_text(answer))
    if "what team" in q_l and _clean_text(answer).lower() != "insufficient":
        bindings_update.setdefault("?team", _clean_text(answer))

    return bindings_update


def normalize_answer_result(parsed, fact_id, used_question, question_item):
    answer = _clean_text(parsed.get("answer", "insufficient"))
    question_type = _clean_text(question_item.get("question_type", ""))
    answer = normalize_yesno_by_question_type(question_type, answer, parsed.get("evidence_span", ""))

    status_raw = _clean_text(parsed.get("status", "insufficient")).lower()
    status_map = {
        "support": "supported",
        "supported": "supported",
        "contradict": "contradicted",
        "contradicted": "contradicted",
        "insufficient": "insufficient",
        "nei": "insufficient",
        "api_error": "api_error",
    }
    status = status_map.get(status_raw, "insufficient")
    evidence_span = _clean_text(parsed.get("evidence_span", ""))
    bindings_update = normalize_bindings_update(parsed, question_item, question_type, answer)

    return {
        "fact_id": fact_id,
        "question": used_question,
        "answer": answer if answer else "insufficient",
        "status": status,
        "evidence_span": evidence_span,
        "bindings_update": bindings_update,
    }


def validate_answer_result(result, fact_id):
    if result["fact_id"] != fact_id:
        raise ValueError(f"fact_id mismatch: {result['fact_id']} vs {fact_id}")
    if result["status"] not in {"supported", "contradicted", "insufficient", "api_error"}:
        raise ValueError(f"invalid status: {result['status']}")
    if not isinstance(result["bindings_update"], dict):
        raise ValueError("bindings_update must be a dict")
    return True


def is_rate_limit_error(e):
    msg = str(e)
    return "429" in msg or "频率超限" in msg or "rate limit" in msg.lower()


def fallback_answer_result(fact_id, used_question, error_type="api_error"):
    return {
        "fact_id": fact_id,
        "question": used_question,
        "answer": error_type,
        "status": error_type,
        "evidence_span": "",
        "bindings_update": {},
    }


def answer_one_question(claim, evidence, atomic_fact, question_item, current_bindings, plan, port):
    filled_fact = replace_placeholders_in_obj(atomic_fact, current_bindings)
    filled_question_item = replace_placeholders_in_obj(question_item, current_bindings)
    used_question = _clean_text(filled_question_item.get("main_question", ""))
    fact_id = atomic_fact.get("id", "")

    infer_count = 0
    last_error = None
    while infer_count < 5:
        try:
            prompt = get_answer_prompt(
                claim=claim,
                evidence=evidence,
                atomic_fact=filled_fact,
                question_item=filled_question_item,
                bindings=current_bindings,
                question=used_question,
            )
            output = llm(prompt, plan=plan, port=port)
            parsed = parse_answer_output(output)
            result = normalize_answer_result(parsed, fact_id, used_question, filled_question_item)
            validate_answer_result(result, fact_id)
            return result
        except Exception as e:
            last_error = e
            print(f"[Retry {infer_count + 1}/5] Error during answer generation for {fact_id}: {e}")

            if is_rate_limit_error(e):
                wait_time = (2 ** infer_count) + random.uniform(0.5, 1.5)
            else:
                wait_time = 1.0 + random.uniform(0.2, 0.8)

            print(f"Sleeping {wait_time:.2f}s before retry...")
            time.sleep(wait_time)
            infer_count += 1

    print(f"[Fallback] fact: {fact_id}, last_error: {last_error}")
    return fallback_answer_result(fact_id, used_question, error_type="api_error")


def order_question_items(question_items):
    indexed = list(enumerate(question_items))
    indexed.sort(key=lambda x: (len(x[1].get("rely_on", []) or []), x[0]))
    return [item for _, item in indexed]


def generate_answers_for_item(data, plan, port):
    claim = data["claim"]
    evidence = data["gold_evidence"]
    decomposition = data.get("decomposition", {}) or {}
    question_plan = data.get("question_plan", {}) or {}

    atomic_facts = decomposition.get("atomic_facts", []) or []
    fact_map = {f.get("id"): f for f in atomic_facts}

    ordered_question_items = order_question_items(question_plan.get("question_items", []) or [])
    final_bindings = {}
    answers = []
    answer_issues = []

    for qitem in ordered_question_items:
        fact_id = qitem.get("fact_id", "")
        atomic_fact = fact_map.get(fact_id, {
            "id": fact_id,
            "text": qitem.get("fact_text", ""),
            "rely_on": qitem.get("rely_on", []),
            "constraint": qitem.get("constraint", {}),
            "critical": qitem.get("critical", False),
        })

        result = answer_one_question(
            claim=claim,
            evidence=evidence,
            atomic_fact=atomic_fact,
            question_item=qitem,
            current_bindings=final_bindings,
            plan=plan,
            port=port,
        )

        answers.append(result)
        final_bindings = merge_bindings(final_bindings, result.get("bindings_update", {}))

        still_unresolved = unresolved_vars(result.get("question", ""))
        if still_unresolved and result.get("status") in {"insufficient", "api_error"}:
            answer_issues.append(
                f"Question for {fact_id} still has unresolved variables after answering: {still_unresolved}"
            )

    return {
        "claim": claim,
        "answers": answers,
        "final_bindings": final_bindings,
    }, answer_issues


def process_data_item(data, plan, port):
    decomposition = data.get("decomposition", {"claim": data["claim"], "atomic_facts": []})
    question_plan = data.get("question_plan", {"question_items": []})

    answer_result, answer_issues = generate_answers_for_item(
        {
            "claim": data["claim"],
            "gold_evidence": data["gold_evidence"],
            "decomposition": decomposition,
            "question_plan": question_plan,
        },
        plan=plan,
        port=port,
    )

    return {
        "id": data["id"],
        "claim": data["claim"],
        "gold_evidence": data["gold_evidence"],
        "num_hops": data.get("num_hops", None),
        "label": data.get("label", None),
        "decomposition": decomposition,
        "question_plan": question_plan,
        "answer_result": answer_result,
        "answer_issues": answer_issues,
        "answer_used_fallback": any(x["status"] == "api_error" for x in answer_result["answers"]),
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
            "question_plan": data.get("question_plan", {"question_items": []})
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
    print("程序结束时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HOVER')
    parser.add_argument('--data_type', type=str, default='dev')
    parser.add_argument('--class_num', type=str, default='2')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=200)
    parser.add_argument('--port', type=str, default='8370')
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--plan', type=str, default='local')
    parser.add_argument('--in_path', type=str, default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_question_[T][S]_[E].json')
    parser.add_argument('--out_path', type=str, default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_answer_[T][S]_[E].json')
    parser.add_argument('--t', type=str, default='')
    args = parser.parse_args()
    main(args)