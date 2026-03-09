import json
import argparse
import re

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from scripts.prompts.decompose_prompts import get_decompose_prompt

def llm(input, port='8370'):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:PORT/v1".replace('PORT', port)
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base
    )
    
    predictions = client.chat.completions.create(
        model="Meta-Llama-3-70B-Instruct-AutoAWQ-4bit",
        messages=[
            {"role": "user", "content": input},
        ],
    )
    result = predictions.choices[0].message.content
    return result

def extract_json_block(text):
    """
    从模型输出中提取最可能的 JSON 块。
    支持：
    1. 纯 JSON
    2. ```json ... ``` 包裹
    3. 前后有解释文字
    """
    text = text.strip()

    # 优先处理 markdown code block
    code_block_pattern = r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```"
    match = re.search(code_block_pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    # 尝试直接找最外层对象/数组
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

def decompose_text(claim, port):
    prompt = get_decompose_prompt(claim)
    
    infer_count = 0
    last_error = None

    while infer_count < 3:
        try:
            output = llm(prompt, port)
            parsed = parse_items(output)
            result = normalize_decomposition(parsed, claim)
            validate_decomposition(result)
            return result
        except Exception as e:
            last_error = e
            print(f"[Retry {infer_count + 1}/3] Error during decomposition: {e}")
            infer_count += 1

    print(f"[Failed] claim: {claim}\nLast error: {last_error}")
    return {
        "claim": claim,
        "atomic_facts": [],
        "constraints": []
    }

def process_data_item(data, port):
    claim = data['claim']
    atomic_facts = decompose_text(claim, port)
    
    return {
        'id': data['id'],
        'claim': claim,
        'gold_evidence': data['gold_evidence'],
        'num_hops': data['num_hops'],
        'label': data['label'],
        'atomic_facts': atomic_facts
    }

def main(args):
    in_path = args.in_path.replace('[DATA]', args.dataset).replace('[TYPE]', f'{args.data_type}')
    with open(in_path, 'r') as f:
        dataset = json.load(f)
    
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
    
    partial_func = partial(process_data_item, port=args.port)
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        features = [executor.submit(partial_func, data) for data in dataset]
        for f in tqdm(as_completed(features), total=len(features)):
            try:
                res = f.result()
                results.append(res)
            except Exception as e:
                print(e)
                continue
    
    out_path = args.out_path.replace('[DATA]', args.dataset).replace('[TYPE]', args.data_type).replace('[CLASS]', args.class_num).replace('[T]', args.t).replace('[S]', str(args.start)).replace('[E]', str(args.end))
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HOVER', help='Dataset name')
    parser.add_argument('--data_type', type=str, default='dev', help='Data type: train/dev/test')
    parser.add_argument('--class_num', type=str, default='2', help='Number of classes: 2/3')
    parser.add_argument('--start', type=int, default=0, help='Start index of data to process')
    parser.add_argument('--end', type=int, default=2000, help='End index of data to process')
    parser.add_argument('--port', type=str, default='8370', help='Port for LLM API')
    
    parser.add_argument('--in_path', type=str, default='./data/[DATA]/converted_data/[TYPE].json', help='Input path template')
    parser.add_argument('--out_path', type=str, default='./data/[DATA]/plan2/[TYPE]_[CLASS]_decomposed_[T]_[S]_[E].json', help='Output path template')
    parser.add_argument('--t', type=str, default='')
    
    args = parser.parse_args()
    main(args)
    