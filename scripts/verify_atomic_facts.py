import json
import argparse
import re
import os

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from prompts.verify_atomic_facts_prompts import get_verify_prompt


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


def parse_verify_output(output_text):
    json_text = extract_json_block(output_text)
    data = json.loads(json_text)
    if not isinstance(data, dict):
        raise ValueError(f"Parsed JSON is not a dict: {type(data)}")
    return data


def normalize_verification_result(parsed, answer_item):
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Verification output is not a dict: {parsed}")

    fact_id = parsed.get("fact_id", answer_item["fact_id"])
    if fact_id is None:
        fact_id = answer_item["fact_id"]
    fact_id = str(fact_id).strip()

    verification_label = parsed.get("verification_label", "insufficient")
    if verification_label is None:
        verification_label = "insufficient"
    verification_label = str(verification_label).strip().lower()

    reason = parsed.get("reason", "")
    if reason is None:
        reason = ""
    reason = str(reason).strip()

    evidence_span = parsed.get("evidence_span", "")
    if evidence_span is None:
        evidence_span = ""
    evidence_span = str(evidence_span).strip()

    if verification_label not in {"support", "contradict", "insufficient"}:
        verification_label = "insufficient"

    return {
        "fact_id": fact_id,
        "filled_fact": answer_item.get("filled_fact", {}),
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
        "fact_id": answer_item.get("fact_id", ""),
        "filled_fact": answer_item.get("filled_fact", {}),
        "verification_label": label,
        "reason": "Fallback from previous answering status.",
        "evidence_span": answer_item.get("evidence_span", "")
    }


def verify_one_fact(claim, evidence, answer_item, plan, port):
    infer_count = 0
    last_error = None

    while infer_count < 3:
        output = None
        try:
            prompt = get_verify_prompt(claim, evidence, answer_item)
            output = llm(prompt, plan=plan, port=port)
            parsed = parse_verify_output(output)
            result = normalize_verification_result(parsed, answer_item)
            validate_verification_result(result, answer_item)
            return result
        except Exception as e:
            last_error = e
            preview = output[:500] if isinstance(output, str) else repr(output)
            print(f"[Retry {infer_count + 1}/3] Error during fact verification for {answer_item.get('fact_id', '')}: {e}")
            print(f"[Output Preview] {preview}")
            infer_count += 1

    print(f"[Fallback] fact: {answer_item.get('fact_id', '')}, last_error: {last_error}")
    return fallback_verification_result(answer_item)


def verify_all_facts(claim, evidence, answer_plan, plan, port):
    if answer_plan is None:
        answer_plan = []
    if not isinstance(answer_plan, list):
        raise ValueError("answer_plan must be a list.")

    fact_verification = []

    for answer_item in answer_plan:
        if not isinstance(answer_item, dict):
            print(f"[Warn] Skip non-dict answer item: {answer_item}")
            continue

        result = verify_one_fact(
            claim=claim,
            evidence=evidence,
            answer_item=answer_item,
            plan=plan,
            port=port
        )
        fact_verification.append(result)

    return fact_verification


def process_data_item(data, plan, port):
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
        plan=plan,
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
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_answer_[T][S]_[E].json',
        help='Input path template'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_fact_verify_[T][S]_[E].json',
        help='Output path template'
    )
    parser.add_argument('--t', type=str, default='')

    args = parser.parse_args()
    main(args)