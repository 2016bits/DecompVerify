import json
import argparse
import re
import os

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from scripts.prompts.decompose_prompts import get_decompose_prompt, get_repair_prompt


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


def llm(input_text, plan='scnet', port='8370'):
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
    return response.choices[0].message.content


def extract_json_block(text):
    """
    从模型输出中提取最可能的 JSON 块。
    支持：
    1. 纯 JSON
    2. ```json ... ``` 包裹
    3. 前后有解释文字
    """
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


def parse_items(output_text):
    """
    解析 LLM 输出，返回 dict:
    {
      "claim": "...",
      "atomic_facts": [...],
      "constraints": [...]
    }
    """
    json_text = extract_json_block(output_text)
    data = json.loads(json_text)
    return data


def normalize_atomic_fact(fact, idx):
    """
    将单个 atomic fact 规整到统一格式
    """
    fact_id = fact.get("id", f"f{idx}")
    subject = str(fact.get("subject", "")).strip()
    predicate = str(fact.get("predicate", "")).strip()
    obj = str(fact.get("object", "")).strip()
    role = str(fact.get("role", "verify")).strip().lower()
    depends_on = fact.get("depends_on", [])

    if not isinstance(depends_on, list):
        depends_on = []

    if role not in {"bridge", "verify", "anchor"}:
        role = "verify"

    return {
        "id": fact_id,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "role": role,
        "depends_on": depends_on,
    }


def normalize_constraint(constraint):
    """
    规整 constraint
    """
    ctype = str(constraint.get("type", "")).strip().lower()
    target_facts = constraint.get("target_facts", [])
    operator = str(constraint.get("operator", "")).strip()
    value = constraint.get("value", "")

    if not isinstance(target_facts, list):
        target_facts = []

    return {
        "type": ctype,
        "target_facts": target_facts,
        "operator": operator,
        "value": value,
    }


def normalize_decomposition(parsed, claim):
    """
    把模型输出规整成最简 schema
    """
    atomic_facts = parsed.get("atomic_facts", [])
    constraints = parsed.get("constraints", [])

    if not isinstance(atomic_facts, list):
        atomic_facts = []
    if not isinstance(constraints, list):
        constraints = []

    normalized_atomic_facts = [
        normalize_atomic_fact(fact, idx + 1)
        for idx, fact in enumerate(atomic_facts)
    ]
    normalized_constraints = [
        normalize_constraint(c) for c in constraints
    ]

    result = {
        "claim": parsed.get("claim", claim),
        "atomic_facts": normalized_atomic_facts,
        "constraints": normalized_constraints,
    }
    return result


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
        for dep in fact_map[fid].get("depends_on", []):
            if dep in fact_map and dfs(dep):
                return True
        path.remove(fid)
        return False

    for fid in fact_map:
        if dfs(fid):
            return True
    return False


def validate_decomposition(result):
    """
    基础校验：
    1. atomic_facts 非空
    2. 每个 fact 必须有 id/subject/predicate/object
    3. depends_on 引用必须存在
    4. constraints.target_facts 引用必须存在
    """
    atomic_facts = result.get("atomic_facts", [])
    constraints = result.get("constraints", [])

    if not atomic_facts:
        raise ValueError("atomic_facts is empty.")

    fact_ids = set()
    for fact in atomic_facts:
        for key in ["id", "subject", "predicate", "object", "role", "depends_on"]:
            if key not in fact:
                raise ValueError(f"Missing field '{key}' in atomic fact: {fact}")
        if not fact["id"]:
            raise ValueError(f"Empty fact id in atomic fact: {fact}")
        if not fact["subject"] or not fact["predicate"] or not fact["object"]:
            raise ValueError(f"Incomplete atomic fact: {fact}")
        fact_ids.add(fact["id"])

    for fact in atomic_facts:
        for dep in fact["depends_on"]:
            if dep not in fact_ids:
                raise ValueError(f"Invalid depends_on reference: {dep}")

    for c in constraints:
        for key in ["type", "target_facts", "operator", "value"]:
            if key not in c:
                raise ValueError(f"Missing field '{key}' in constraint: {c}")
        for fid in c["target_facts"]:
            if fid not in fact_ids:
                raise ValueError(f"Invalid target_facts reference: {fid}")

    return True


def extract_variables(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", text))


def fact_variables(fact):
    return extract_variables(fact.get("subject", "")) | extract_variables(fact.get("object", ""))


def has_real_dependency(src_fact, tgt_fact):
    """
    若 tgt 依赖 src，通常应该存在变量流：
    src 中引入的变量，在 tgt 中被使用。
    """
    src_vars = fact_variables(src_fact)
    tgt_vars = fact_variables(tgt_fact)
    return len(src_vars & tgt_vars) > 0


def find_suspicious_issues(decomposition):
    issues = []
    atomic_facts = decomposition.get("atomic_facts", [])
    constraints = decomposition.get("constraints", [])

    if not atomic_facts:
        issues.append("atomic_facts is empty")
        return issues

    fact_map = {fact["id"]: fact for fact in atomic_facts}

    # 1. depends_on 但没有变量流
    for fact in atomic_facts:
        for dep_id in fact.get("depends_on", []):
            if dep_id not in fact_map:
                issues.append(
                    f"Invalid dependency: {fact['id']} depends on missing fact {dep_id}"
                )
                continue
            src = fact_map[dep_id]
            if not has_real_dependency(src, fact):
                issues.append(
                    f"Suspicious dependency: {fact['id']} depends on {dep_id} but no shared variable found"
                )

    # 2. bridge fact 不含变量
    for fact in atomic_facts:
        if fact.get("role") == "bridge":
            vars_in_fact = fact_variables(fact)
            if len(vars_in_fact) == 0:
                issues.append(f"Bridge fact {fact['id']} contains no variable")

    # 3. 变量只出现一次，通常说明链没形成
    var_count = {}
    for fact in atomic_facts:
        for v in fact_variables(fact):
            var_count[v] = var_count.get(v, 0) + 1
    for v, c in var_count.items():
        if c == 1:
            issues.append(f"Variable {v} appears only once")

    # 4. anchor 不带变量通常可疑
    for fact in atomic_facts:
        if fact.get("role") == "anchor":
            vars_in_fact = fact_variables(fact)
            if len(vars_in_fact) == 0:
                issues.append(f"Anchor fact {fact['id']} contains no variable")

    # 5. 约束没锚到事实
    fact_ids = {fact["id"] for fact in atomic_facts}
    for c in constraints:
        for fid in c.get("target_facts", []):
            if fid not in fact_ids:
                issues.append(f"Constraint references missing fact {fid}")

    # 6. 循环依赖
    if detect_cycle(atomic_facts):
        issues.append("Cyclic dependencies detected")

    return issues


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
            if issues:
                repair_count = 0
                repaired = result
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

            return result, [], False

        except Exception as e:
            last_error = e
            print(f"[Retry {infer_count + 1}/3] Error during decomposition: {e}")
            infer_count += 1

    print(f"[Failed] claim: {claim}\nLast error: {last_error}")
    return {
        "claim": claim,
        "atomic_facts": [],
        "constraints": []
    }, [f"decomposition_failed: {str(last_error)}"], False


def process_data_item(data, plan, port):
    claim = data['claim']
    decomposition_result, issues, is_repaired = decompose_text(claim, plan, port)

    return {
        'id': data['id'],
        'claim': claim,
        'gold_evidence': data['gold_evidence'],
        'num_hops': data['num_hops'],
        'label': data['label'],
        'atomic_facts': decomposition_result,
        'issues': issues,
        'is_repaired': is_repaired
    }


def main(args):
    in_path = args.in_path.replace('[DATA]', args.dataset).replace('[TYPE]', f'{args.data_type}')

    with open(in_path, 'r') as f:
        raws = json.load(f)

    dataset = []
    for data in raws[args.start: args.end]:
        gold_evidence = data['evidence']
        dataset.append({
            'id': data['id'],
            'claim': data['claim'],
            'gold_evidence': gold_evidence,
            'num_hops': data['num_hops'],
            'label': data['label']
        })

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
        args.out_path
        .replace('[DATA]', args.dataset)
        .replace('[TYPE]', args.data_type)
        .replace('[CLASS]', args.class_num)
        .replace('[T]', args.t)
        .replace('[S]', str(args.start))
        .replace('[E]', str(args.end))
        .replace('[PLAN]', args.plan)
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HOVER_subset', help='Dataset name')
    parser.add_argument('--data_type', type=str, default='dev', help='Data type: train/dev/test')
    parser.add_argument('--class_num', type=str, default='2', help='Number of classes: 2/3')
    parser.add_argument('--start', type=int, default=0, help='Start index of data to process')
    parser.add_argument('--end', type=int, default=200, help='End index of data to process')
    parser.add_argument('--port', type=str, default='8370', help='Port for local LLM API')
    parser.add_argument('--max_workers', type=int, default=8, help='Number of concurrent workers')

    parser.add_argument('--in_path', type=str, default='./data/[DATA]/converted_data/[TYPE].json', help='Input path template')
    parser.add_argument('--out_path', type=str, default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_decomposed_[T][S]_[E].json', help='Output path template')
    parser.add_argument('--t', type=str, default='')
    parser.add_argument('--plan', type=str, default='scnet', help='LLM plan: local/scnet/iflow')

    args = parser.parse_args()
    main(args)