import json
import argparse
import re
import os
import ast

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from scripts.prompts.decompose_prompts import get_decompose_prompt, get_repair_prompt

ALLOWED_DECOMP_TYPES = {
    "simple",
    "coordinated",
    "nested",
    "quantified_or_comparative",
}
VAR_PATTERN = r"\?[A-Za-z_][A-Za-z0-9_]*"


def build_client_and_model(plan, port='8370'):
    """
    根据 plan 自动构建 client 和 model
    支持:
      - local
      - scnet
      - iflow
      - scnet_qwen235b
      - iflow_qwen32b
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
        temperature = None
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
        temperature = 0.7
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
        temperature = 0.7
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
            f"Supported plans are: local, scnet, iflow, scnet_qwen235b, iflow_qwen32b, azure, azure_gpt4o"
        )



def llm(input_text, plan="scnet", port="8370"):
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
    content = response.choices[0].message.content
    return content if isinstance(content, str) else str(content)



def extract_json_block(text):
    """从模型输出中提取最可能的 JSON 块。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output.")

    # fenced code block
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    if text.startswith("{") or text.startswith("["):
        return text

    starts = [idx for idx in [text.find("{"), text.find("[")] if idx != -1]
    if not starts:
        raise ValueError("No JSON object or array found in model output.")
    start_idx = min(starts)

    opener = text[start_idx]
    closer = "}" if opener == "{" else "]"

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

        if ch == opener:
            stack += 1
        elif ch == closer:
            stack -= 1
            if stack == 0:
                return text[start_idx : i + 1].strip()

    raise ValueError("Incomplete JSON block in model output.")



def _strip_trailing_commas(s):
    return re.sub(r",(\s*[}\]])", r"\1", s)



def _quote_unquoted_keys(s):
    # { claim: "x" } -> { "claim": "x" }
    pattern = r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)' 
    return re.sub(pattern, r'\1"\2"\3', s)



def try_parse_json(text):
    raw = extract_json_block(text)

    # 1) standard JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    repaired = raw
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = repaired.replace("True", "true").replace("False", "false").replace("None", "null")
    repaired = _quote_unquoted_keys(repaired)
    repaired = _strip_trailing_commas(repaired)

    # 2) repaired JSON
    try:
        return json.loads(repaired)
    except Exception:
        pass

    # 3) python literal fallback
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass

    # 4) python literal after mild repair
    py_like = repaired.replace("true", "True").replace("false", "False").replace("null", "None")
    return ast.literal_eval(py_like)



def parse_items(output_text):
    return try_parse_json(output_text)



def _clean_text(s):
    return re.sub(r"\s+", " ", str(s or "").strip())



def extract_variables(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(VAR_PATTERN, text))



def normalize_decomposition_type(value, claim="", atomic_facts=None):
    atomic_facts = atomic_facts or []
    if value is None:
        value = ""

    v = str(value).strip().lower()
    v = v.replace("-", "_").replace(" ", "_").replace("/", "_")

    mapping = {
        "simple": "simple",
        "simple_claim": "simple",
        "single": "simple",

        "coordinated": "coordinated",
        "coordinate": "coordinated",
        "coordination": "coordinated",
        "parallel": "coordinated",

        "nested": "nested",
        "nested_claim": "nested",
        "subordinate": "nested",
        "clausal": "nested",

        "quantified": "quantified_or_comparative",
        "comparative": "quantified_or_comparative",
        "superlative": "quantified_or_comparative",
        "quantified_comparative": "quantified_or_comparative",
        "quantified_or_comparative": "quantified_or_comparative",
    }
    if v in mapping:
        return mapping[v]

    return infer_decomposition_type(claim, atomic_facts)



def normalize_constraint(constraint):
    if not isinstance(constraint, dict):
        constraint = {}

    neg = constraint.get("negation", None)
    if isinstance(neg, str):
        low = neg.strip().lower()
        if low in {"true", "yes", "not", "negated", "negative"}:
            neg = True
        else:
            neg = None
    elif neg is True:
        neg = True
    else:
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



def normalize_atomic_fact(fact, idx):
    if not isinstance(fact, dict):
        fact = {"text": str(fact)}

    fact_id = str(fact.get("id", f"f{idx}")).strip() or f"f{idx}"
    text = _clean_text(fact.get("text", ""))

    rely_on = fact.get("rely_on", [])
    if rely_on is None:
        rely_on = []
    if not isinstance(rely_on, list):
        rely_on = [rely_on]
    rely_on = [str(x).strip() for x in rely_on if str(x).strip()]

    constraint = normalize_constraint(fact.get("constraint", {}))
    return {
        "id": fact_id,
        "text": text,
        "rely_on": rely_on,
        "constraint": constraint,
    }



def infer_decomposition_type(claim, atomic_facts):
    claim_l = (claim or "").lower()
    texts = " ".join(f.get("text", "").lower() for f in atomic_facts)

    if any(" than " in f.get("text", "").lower() for f in atomic_facts):
        return "quantified_or_comparative"
    if any(f.get("constraint", {}).get("quantity") for f in atomic_facts):
        return "quantified_or_comparative"
    if any(tok in claim_l or tok in texts for tok in ["earlier than", "later than", "more than", "less than", "highest", "lowest", "first", "last"]):
        return "quantified_or_comparative"
    if any(f.get("rely_on") for f in atomic_facts):
        return "nested"
    if len(atomic_facts) > 1 and any(tok in claim_l for tok in [" and ", " or ", ",", "both "]):
        return "coordinated"
    return "simple" if len(atomic_facts) <= 1 else "coordinated"



def normalize_decomposition(parsed, claim):
    if not isinstance(parsed, dict):
        raise ValueError("Parsed output is not a JSON object.")

    atomic_facts = parsed.get("atomic_facts", [])
    if not isinstance(atomic_facts, list):
        atomic_facts = []

    normalized_atomic_facts = [normalize_atomic_fact(fact, idx + 1) for idx, fact in enumerate(atomic_facts)]

    # fallback: if model failed to output atomic_facts but gave top-level text-like content, degrade gracefully
    if not normalized_atomic_facts:
        fallback_text = _clean_text(parsed.get("text", ""))
        if fallback_text:
            normalized_atomic_facts = [{
                "id": "f1",
                "text": fallback_text,
                "rely_on": [],
                "constraint": {"negation": None, "time": [], "quantity": []},
            }]
        else:
            normalized_atomic_facts = [{
                "id": "f1",
                "text": _clean_text(claim),
                "rely_on": [],
                "constraint": {"negation": None, "time": [], "quantity": []},
            }]

    decomp_type = normalize_decomposition_type(
        parsed.get("decomposition_type", ""),
        claim=claim,
        atomic_facts=normalized_atomic_facts,
    )

    return {
        "claim": _clean_text(parsed.get("claim", claim)) or _clean_text(claim),
        "decomposition_type": decomp_type,
        "atomic_facts": normalized_atomic_facts,
    }



def detect_cycle(atomic_facts):
    fact_map = {fact["id"]: fact for fact in atomic_facts}
    visited = {}
    path = set()

    def dfs(fid):
        if fid in path:
            return True
        if visited.get(fid, False):
            return False
        visited[fid] = True
        path.add(fid)
        for dep in fact_map[fid].get("rely_on", []):
            if dep in fact_map and dfs(dep):
                return True
        path.remove(fid)
        return False

    for fid in fact_map:
        if dfs(fid):
            return True
    return False



def count_relation_like_units(text):
    text = (text or "").lower().strip()
    if not text:
        return 0
    chunks = re.split(r"\b(?:and|or|then|while|but)\b|;", text)
    chunks = [c.strip() for c in chunks if c.strip()]
    return max(1, len(chunks))



def has_real_dependency(src_fact, tgt_fact):
    return len(extract_variables(src_fact.get("text", "")) & extract_variables(tgt_fact.get("text", ""))) > 0



def validate_decomposition(result):
    atomic_facts = result.get("atomic_facts", [])
    decomp_type = result.get("decomposition_type", "")

    if not atomic_facts:
        raise ValueError("atomic_facts is empty.")
    if decomp_type not in ALLOWED_DECOMP_TYPES:
        raise ValueError(f"Invalid decomposition_type: {decomp_type}")

    fact_ids = []
    seen = set()
    for idx, fact in enumerate(atomic_facts, start=1):
        for key in ["id", "text", "rely_on", "constraint"]:
            if key not in fact:
                raise ValueError(f"Missing field '{key}' in atomic fact: {fact}")
        if not fact["id"]:
            raise ValueError(f"Empty fact id: {fact}")
        if fact["id"] in seen:
            raise ValueError(f"Duplicate fact id: {fact['id']}")
        seen.add(fact["id"])
        fact_ids.append(fact["id"])

        if not fact["text"]:
            raise ValueError(f"Empty text in atomic fact: {fact}")

        c = fact["constraint"]
        if set(c.keys()) != {"negation", "time", "quantity"}:
            raise ValueError(f"Constraint keys must be exactly negation/time/quantity: {fact}")
        if c["negation"] not in {None, True}:
            raise ValueError(f"negation must be null or true: {fact}")
        if not isinstance(c["time"], list) or not isinstance(c["quantity"], list):
            raise ValueError(f"time and quantity must be lists: {fact}")

    fact_id_set = set(fact_ids)
    for idx, fact in enumerate(atomic_facts):
        for dep in fact["rely_on"]:
            if dep not in fact_id_set:
                raise ValueError(f"Invalid rely_on reference: {dep}")
            if fact_ids.index(dep) >= idx:
                raise ValueError(f"rely_on must reference earlier facts only: {fact}")

    if detect_cycle(atomic_facts):
        raise ValueError("Cyclic dependencies detected")

    return True



def infer_claim_category_from_structure(result):
    return infer_decomposition_type(result.get("claim", ""), result.get("atomic_facts", []))



def find_suspicious_issues(decomposition):
    issues = []
    atomic_facts = decomposition.get("atomic_facts", [])
    decomp_type = decomposition.get("decomposition_type")
    fact_map = {fact["id"]: fact for fact in atomic_facts}

    if not atomic_facts:
        return ["atomic_facts is empty"]

    for fact in atomic_facts:
        # 保守一点：只有明显的并列/多关系才报问题，避免过度拒绝本地模型输出
        if count_relation_like_units(fact.get("text", "")) > 1 and not any(tok in fact.get("text", "").lower() for tok in [" than "]):
            issues.append(f"Atomic fact {fact['id']} may contain multiple relations")
        for dep_id in fact.get("rely_on", []):
            if dep_id not in fact_map:
                issues.append(f"Invalid dependency: {fact['id']} relies on missing fact {dep_id}")
                continue
            # 只对含变量的情况检查共享变量，避免普通嵌套 claim 被误报
            if extract_variables(fact_map[dep_id].get("text", "")) or extract_variables(fact.get("text", "")):
                if not has_real_dependency(fact_map[dep_id], fact):
                    issues.append(f"Suspicious dependency: {fact['id']} relies on {dep_id} but no shared variable found")

    inferred = infer_claim_category_from_structure(decomposition)
    if decomp_type != inferred:
        issues.append(f"decomposition_type '{decomp_type}' mismatches inferred structure '{inferred}'")

    return sorted(set(issues))



def repair_decomposition(claim, decomposition, issues, plan, port):
    prompt = get_repair_prompt(claim, decomposition, issues)
    output = llm(prompt, plan=plan, port=port)
    parsed = parse_items(output)
    repaired = normalize_decomposition(parsed, claim)
    validate_decomposition(repaired)
    return repaired



def decompose_text(claim, plan, port):
    prompt = get_decompose_prompt(claim)
    infer_count = 0
    last_error = None

    while infer_count < 3:
        try:
            output = llm(prompt, plan=plan, port=port)
            parsed = parse_items(output)
            result = normalize_decomposition(parsed, claim)
            validate_decomposition(result)
            issues = find_suspicious_issues(result)
            if not issues:
                return result, [], False

            repaired = result
            repair_count = 0
            while repair_count < 2:
                try:
                    repaired = repair_decomposition(claim, repaired, issues, plan, port)
                    new_issues = find_suspicious_issues(repaired)
                    if not new_issues:
                        return repaired, issues, True
                    issues = new_issues
                    repair_count += 1
                except Exception as e:
                    print(f"[Repair Retry {repair_count + 1}/2] Error during repair: {e}")
                    repair_count += 1

            return repaired, issues, True
        except Exception as e:
            last_error = e
            print(f"[Retry {infer_count + 1}/3] Error during decomposition: {e}")
            infer_count += 1

    print(f"[Failed] claim: {claim}\nLast error: {last_error}")
    return {
        "claim": claim,
        "decomposition_type": "simple",
        "atomic_facts": [
            {
                "id": "f1",
                "text": _clean_text(claim),
                "rely_on": [],
                "constraint": {"negation": None, "time": [], "quantity": []},
            }
        ],
    }, [f"decomposition_failed: {str(last_error)}"], False



def process_data_item(data, plan, port):
    claim = data["claim"]
    decomposition_result, issues, is_repaired = decompose_text(claim, plan, port)

    return {
        "id": data["id"],
        "claim": claim,
        "gold_evidence": data["gold_evidence"],
        "num_hops": data["num_hops"],
        "label": data["label"],
        "atomic_facts": decomposition_result,
        "issues": issues,
        "is_repaired": is_repaired,
    }



def main(args):
    in_path = args.in_path.replace("[DATA]", args.dataset).replace("[TYPE]", f"{args.data_type}")

    with open(in_path, "r", encoding="utf-8") as f:
        raws = json.load(f)

    dataset = []
    for data in raws[args.start : args.end]:
        gold_evidence = data["evidence"]
        dataset.append(
            {
                "id": data["id"],
                "claim": data["claim"],
                "gold_evidence": gold_evidence,
                "num_hops": data["num_hops"],
                "label": data["label"],
            }
        )

    partial_func = partial(process_data_item, plan=args.plan, port=args.port)
    results = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        features = [executor.submit(partial_func, data) for data in dataset]
        for f in tqdm(as_completed(features), total=len(features)):
            try:
                res = f.result()
                results.append(res)
            except Exception as e:
                print(e)
                continue

    out_path = (
        args.out_path.replace("[DATA]", args.dataset)
        .replace("[TYPE]", args.data_type)
        .replace("[CLASS]", args.class_num)
        .replace("[T]", args.t)
        .replace("[S]", str(args.start))
        .replace("[E]", str(args.end))
        .replace("[PLAN]", args.plan)
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HOVER_subset", help="Dataset name")
    parser.add_argument("--data_type", type=str, default="dev", help="Data type: train/dev/test")
    parser.add_argument("--class_num", type=str, default="2", help="Number of classes: 2/3")
    parser.add_argument("--start", type=int, default=0, help="Start index of data to process")
    parser.add_argument("--end", type=int, default=200, help="End index of data to process")
    parser.add_argument("--port", type=str, default="8370", help="Port for local LLM API")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of concurrent workers")

    parser.add_argument("--in_path", type=str, default="./data/[DATA]/converted_data/[TYPE].json", help="Input path template")
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_decomposed_[T][S]_[E].json",
        help="Output path template",
    )
    parser.add_argument("--t", type=str, default="")
    parser.add_argument("--plan", type=str, default="scnet", help="LLM plan: local/scnet/iflow/azure/qwen_plan*")

    args = parser.parse_args()
    main(args)
