import os
import re
import json
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI

from prompts.decompose_prompts import get_decompose_prompt


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


def parse_decompose_output(output_text):
    json_text = extract_json_block(output_text)
    return json.loads(json_text)


def _clean_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_constraint(constraint):
    if not isinstance(constraint, dict):
        constraint = {}

    neg = constraint.get("negation", None)
    if neg is not True:
        neg = None

    time_vals = constraint.get("time", [])
    if time_vals is None:
        time_vals = []
    if not isinstance(time_vals, list):
        time_vals = [time_vals]
    time_vals = [_clean_text(x) for x in time_vals if _clean_text(x)]

    quantity_vals = constraint.get("quantity", [])
    if quantity_vals is None:
        quantity_vals = []
    if not isinstance(quantity_vals, list):
        quantity_vals = [quantity_vals]
    quantity_vals = [_clean_text(x) for x in quantity_vals if _clean_text(x)]

    return {
        "negation": neg,
        "time": time_vals,
        "quantity": quantity_vals,
    }


def infer_critical(text, constraint):
    text_l = _clean_text(text).lower()
    constraint = constraint or {}

    if constraint.get("negation") is True:
        return True
    if constraint.get("time"):
        return True
    if constraint.get("quantity"):
        return True

    critical_keywords = [
        "largest", "smallest", "biggest", "highest", "lowest",
        "oldest", "youngest", "first", "last", "same", "different",
        "not the same", "equal", "unequal", "more than", "less than",
        "at least", "at most", "exactly", "only", "former", "latter",
        "born in", "shown on", "shown in", "located in", "took place in",
        "directed by", "same director", "same battle", "same person"
    ]
    if any(k in text_l for k in critical_keywords):
        return True

    # key location constraints at the tail
    if re.search(r"\b(in|on|at)\s+[A-Z][A-Za-z0-9,\- ]+$", text):
        return True

    return False


def normalize_fact(fact, idx):
    if not isinstance(fact, dict):
        fact = {}

    text = _clean_text(fact.get("text", ""))
    rely_on = fact.get("rely_on", [])
    if rely_on is None:
        rely_on = []
    if not isinstance(rely_on, list):
        rely_on = [rely_on]
    rely_on = [_clean_text(x) for x in rely_on if _clean_text(x)]

    constraint = normalize_constraint(fact.get("constraint", {}))

    critical_raw = fact.get("critical", None)
    critical = True if critical_raw is True else infer_critical(text, constraint)

    return {
        "id": _clean_text(fact.get("id", f"f{idx + 1}")) or f"f{idx + 1}",
        "text": text,
        "rely_on": rely_on,
        "constraint": constraint,
        "critical": critical,
    }


def normalize_decomposition_result(parsed, claim):
    dtype = _clean_text(parsed.get("decomposition_type", "simple")).lower()
    if dtype not in {"simple", "coordinated", "nested", "comparison"}:
        dtype = "simple"

    atomic_facts = parsed.get("atomic_facts", [])
    if not isinstance(atomic_facts, list):
        atomic_facts = []

    normalized_facts = []
    for i, fact in enumerate(atomic_facts):
        normalized_facts.append(normalize_fact(fact, i))

    return {
        "claim": claim,
        "decomposition_type": dtype,
        "atomic_facts": normalized_facts,
    }


def validate_decomposition_result(result):
    if "claim" not in result or "atomic_facts" not in result:
        raise ValueError("Invalid decomposition result.")
    for fact in result["atomic_facts"]:
        if "id" not in fact or "text" not in fact or "rely_on" not in fact or "constraint" not in fact:
            raise ValueError("Invalid atomic fact.")
        if "critical" not in fact:
            raise ValueError("Missing critical field.")
    return True


def fallback_decomposition(claim):
    return {
        "claim": claim,
        "decomposition_type": "simple",
        "atomic_facts": [
            {
                "id": "f1",
                "text": claim,
                "rely_on": [],
                "constraint": {
                    "negation": None,
                    "time": [],
                    "quantity": []
                },
                "critical": True
            }
        ]
    }


def process_data_item(data, plan, port):
    claim = data["claim"]
    infer_count = 0
    last_error = None

    while infer_count < 3:
        try:
            prompt = get_decompose_prompt(claim)
            output = llm(prompt, plan=plan, port=port)
            parsed = parse_decompose_output(output)
            result = normalize_decomposition_result(parsed, claim)
            validate_decomposition_result(result)

            out = dict(data)
            out["decomposition"] = result
            return out
        except Exception as e:
            last_error = e
            print(f"[Retry {infer_count + 1}/3] Error during decomposition: {e}")
            infer_count += 1

    print(f"[Fallback] decomposition failed: {last_error}")
    out = dict(data)
    out["decomposition"] = fallback_decomposition(claim)
    return out


def main(args):
    in_path = (
        args.in_path
        .replace("[DATA]", args.dataset)
        .replace("[TYPE]", args.data_type)
    )

    with open(in_path, "r", encoding="utf-8") as f:
        raws = json.load(f)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HOVER_subset")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--class_num", type=str, default="2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument("--port", type=str, default="8270")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--plan", type=str, default="local")
    parser.add_argument("--t", type=str, default="")

    parser.add_argument('--in_path', type=str, default='./data/[DATA]/converted_data/[TYPE].json', help='Input path template')
    parser.add_argument('--out_path', type=str, default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_decomposed_[T][S]_[E].json', help='Output path template')

    args = parser.parse_args()
    main(args)