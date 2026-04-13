import os
import re
import json
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI
from datetime import datetime

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

    # 4) Azure OpenAI GPT-4o
    elif plan.startswith("azure"):
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set in environment variables.")

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
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    first_brace = text.find("{")
    if first_brace == -1:
        raise ValueError("No JSON found.")

    stack = 0
    in_string = False
    escape = False
    for i in range(first_brace, len(text)):
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

        if ch == "{":
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0:
                return text[first_brace:i + 1]
    raise ValueError("Incomplete JSON block.")


def parse_question_output(output_text):
    json_text = extract_json_block(output_text)
    return json.loads(json_text)


def _clean_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")


def infer_answer_slot(question_type, fact_text, main_question):
    fact_text_l = _clean_text(fact_text).lower()
    q_l = _clean_text(main_question).lower()

    explicit_vars = [x for x in VAR_PATTERN.findall(fact_text) if x.startswith("?")]
    if len(explicit_vars) == 1 and question_type == "entity_wh":
        return explicit_vars[0]

    patterns = [
        (r"what team", "?team"),
        (r"who is an ambassador", "?ambassador"),
        (r"who is the ambassador", "?ambassador"),
        (r"what is the home ground", "?home_ground"),
        (r"which film", "?film"),
        (r"what film", "?film"),
        (r"what animal", "?animal"),
        (r"who wrote", "?author"),
        (r"who plays", "?actor"),
        (r"who is the american actress", "?actress"),
        (r"where was .* born", "?village"),
        (r"in which village .* born", "?village"),
    ]
    for pat, slot in patterns:
        if re.search(pat, q_l):
            return slot

    if question_type == "entity_wh":
        if "team" in fact_text_l:
            return "?team"
        if "home ground" in fact_text_l:
            return "?home_ground"
        if "ambassador" in fact_text_l:
            return "?ambassador"
        if "film" in fact_text_l:
            return "?film"
        if "animal" in fact_text_l:
            return "?animal"
        if "actress" in fact_text_l:
            return "?actress"
        if "author" in fact_text_l:
            return "?author"
        if "village" in fact_text_l and "born" in fact_text_l:
            return "?village"

    return None


def normalize_question_item(item, atomic_fact):
    if not isinstance(item, dict):
        item = {}

    qtype = _clean_text(item.get("question_type", "relation_yesno"))
    if qtype not in {"relation_yesno", "entity_wh", "time_wh", "quantity_wh"}:
        qtype = "relation_yesno"

    main_question = _clean_text(item.get("main_question", ""))
    if not main_question:
        main_question = _clean_text(item.get("question", ""))

    search_hints = item.get("search_hints", [])
    if not isinstance(search_hints, list):
        search_hints = [search_hints]
    search_hints = [_clean_text(x) for x in search_hints if _clean_text(x)]

    constraint_questions = item.get("constraint_questions", [])
    if not isinstance(constraint_questions, list):
        constraint_questions = []

    answer_slot = _clean_text(item.get("answer_slot", ""))
    if not answer_slot:
        answer_slot = infer_answer_slot(
            qtype,
            atomic_fact.get("text", ""),
            main_question
        )

    return {
        "fact_id": atomic_fact.get("id", ""),
        "fact_text": atomic_fact.get("text", ""),
        "rely_on": atomic_fact.get("rely_on", []),
        "constraint": atomic_fact.get("constraint", {}),
        "critical": atomic_fact.get("critical", False),
        "main_question": main_question,
        "question_type": qtype,
        "answer_slot": answer_slot,
        "constraint_questions": constraint_questions,
        "search_hints": search_hints,
    }


def normalize_question_result(parsed, decomposition):
    qitems = parsed.get("question_items", [])
    if not isinstance(qitems, list):
        qitems = []

    fact_map = {f["id"]: f for f in decomposition.get("atomic_facts", [])}
    normalized = []
    used = set()

    for item in qitems:
        fid = _clean_text(item.get("fact_id", ""))
        if fid in fact_map:
            normalized.append(normalize_question_item(item, fact_map[fid]))
            used.add(fid)

    for fact in decomposition.get("atomic_facts", []):
        if fact["id"] in used:
            continue

        fact_text = _clean_text(fact.get("text", ""))
        constraint = fact.get("constraint", {}) or {}

        if constraint.get("time"):
            qtype = "time_wh"
            mq = f"When did {fact_text}?"
        elif constraint.get("quantity"):
            qtype = "quantity_wh"
            mq = f"What is the quantity for: {fact_text}?"
        elif re.search(r"\b(is|are|was|were|has|have|did|does|do)\b", fact_text.lower()):
            qtype = "relation_yesno"
            mq = f"Is it true that {fact_text}?"
        else:
            qtype = "entity_wh"
            mq = f"What entity satisfies: {fact_text}?"

        normalized.append({
            "fact_id": fact["id"],
            "fact_text": fact_text,
            "rely_on": fact.get("rely_on", []),
            "constraint": fact.get("constraint", {}),
            "critical": fact.get("critical", False),
            "main_question": mq,
            "question_type": qtype,
            "answer_slot": infer_answer_slot(qtype, fact_text, mq),
            "constraint_questions": [],
            "search_hints": [fact_text],
        })

    return {
        "claim": decomposition.get("claim", ""),
        "decomposition_type": decomposition.get("decomposition_type", "simple"),
        "question_items": normalized
    }


def process_data_item(data, plan, port):
    decomposition = data.get("decomposition", {})
    claim = data["claim"]

    infer_count = 0
    last_error = None
    while infer_count < 3:
        try:
            prompt = get_question_prompt(claim, decomposition)
            output = llm(prompt, plan=plan, port=port)
            parsed = parse_question_output(output)
            qplan = normalize_question_result(parsed, decomposition)

            out = dict(data)
            out["question_plan"] = qplan
            return out
        except Exception as e:
            last_error = e
            print(f"[Retry {infer_count + 1}/3] Error during question generation: {e}")
            infer_count += 1

    print(f"[Fallback] question generation failed: {last_error}")
    out = dict(data)
    out["question_plan"] = normalize_question_result({}, decomposition)
    return out


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

    partial_func = partial(process_data_item, plan=args.plan, port=args.port)
    results = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(partial_func, data) for data in raws]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error in future: {e}")

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
    parser.add_argument("--dataset", type=str, default="HOVER")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--class_num", type=str, default="2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument("--port", type=str, default="8370")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--plan", type=str, default="local")
    parser.add_argument("--t", type=str, default="")

    parser.add_argument(
        "--in_path",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_decomposed_[T][S]_[E].json"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_question_[T][S]_[E].json"
    )

    args = parser.parse_args()
    main(args)