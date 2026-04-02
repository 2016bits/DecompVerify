import json
import argparse
import re
import os

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Dict, List


from prompts.question_prompts import get_question_prompt, get_question_repair_prompt

ALLOWED_QUESTION_TYPES = {
    "relation_yesno",
    "entity_wh",
    "time_wh",
    "quantity_wh",
    "description_wh",
}

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


def extract_json_block(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output.")

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


def try_parse_json(text: str) -> Dict[str, Any]:
    json_text = extract_json_block(text)
    try:
        return json.loads(json_text)
    except Exception:
        repaired = json_text
        repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
        repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        repaired = re.sub(r"(?<!\\)'", '"', repaired)
        return json.loads(repaired)


def _clean_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def normalize_question_type(value: Any, main_question: str) -> str:
    v = _clean_text(value).lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "relation_yesno": "relation_yesno",
        "yesno": "relation_yesno",
        "boolean": "relation_yesno",
        "entity_wh": "entity_wh",
        "who": "entity_wh",
        "which": "entity_wh",
        "what_entity": "entity_wh",
        "time_wh": "time_wh",
        "when": "time_wh",
        "year": "time_wh",
        "quantity_wh": "quantity_wh",
        "how_many": "quantity_wh",
        "how_much": "quantity_wh",
        "number": "quantity_wh",
        "description_wh": "description_wh",
        "what": "description_wh",
    }
    if v in mapping:
        return mapping[v]

    q = (main_question or "").strip().lower()
    if q.startswith("when ") or q.startswith("in what year"):
        return "time_wh"
    if q.startswith("how many ") or q.startswith("how much ") or "what is the population" in q or "what is the capacity" in q:
        return "quantity_wh"
    if q.startswith("who ") or q.startswith("which "):
        return "entity_wh"
    if q.startswith("did ") or q.startswith("is ") or q.startswith("was ") or q.startswith("are ") or q.startswith("does "):
        return "relation_yesno"
    return "description_wh"


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        item = _clean_text(item)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def constraint_kind(constraint: Dict[str, Any]) -> str:
    if constraint.get("quantity"):
        return "quantity"
    if constraint.get("time"):
        return "time"
    if constraint.get("negation") is True:
        return "negation"
    return "none"


def normalize_question_item(item: Dict[str, Any], fact: Dict[str, Any]) -> Dict[str, Any]:
    fact_id = fact["id"]
    fact_text = fact["text"]
    rely_on = fact.get("rely_on", [])
    constraint = fact.get("constraint", {})

    main_question = _clean_text(item.get("main_question", ""))
    if not main_question.endswith("?") and main_question:
        main_question += "?"

    question_type = normalize_question_type(item.get("question_type"), main_question)

    constraint_questions = item.get("constraint_questions", [])
    if not isinstance(constraint_questions, list):
        constraint_questions = []

    normalized_cqs = []
    for cq in constraint_questions:
        if not isinstance(cq, dict):
            continue
        cq_type = _clean_text(cq.get("type", "")).lower()
        cq_question = _clean_text(cq.get("question", ""))
        if not cq_question:
            continue
        if not cq_question.endswith("?"):
            cq_question += "?"
        if cq_type not in {"time", "quantity"}:
            continue
        normalized_cqs.append({"type": cq_type, "question": cq_question})

    search_hints = item.get("search_hints", [])
    if not isinstance(search_hints, list):
        search_hints = []
    search_hints = dedup_keep_order(search_hints)

    return {
        "fact_id": fact_id,
        "fact_text": fact_text,
        "rely_on": rely_on,
        "constraint": constraint,
        "main_question": main_question,
        "question_type": question_type,
        "constraint_questions": normalized_cqs,
        "search_hints": search_hints,
    }


def fallback_main_question(fact: Dict[str, Any]) -> str:
    text = fact["text"].strip().rstrip(".")
    c = fact.get("constraint", {})
    has_time = bool(c.get("time"))
    has_quantity = bool(c.get("quantity"))
    has_neg = c.get("negation") is True
    lower = text.lower()

    if has_quantity:
        if "capacity" in lower:
            return f"What is the capacity described by this statement: {text}?"
        if "population" in lower:
            return f"What is the population described by this statement: {text}?"
        return f"How many or how much is specified in this statement: {text}?"

    if has_time:
        return f"When did this happen: {text}?"

    if has_neg:
        return f"Is the following statement true: {text}?"

    if lower.startswith("?") or " ?" in text:
        if any(k in lower for k in ["directed", "wrote", "starred", "played for", "won", "founded", "is the director of"]):
            return f"Who satisfies this statement: {text}?"
        return f"What entity satisfies this statement: {text}?"

    if lower.startswith("there is ") or lower.startswith("there exists "):
        return f"What entity satisfies this statement: {text}?"

    if any(lower.startswith(prefix) for prefix in ["is ", "was ", "are ", "were ", "did ", "does ", "do ", "has ", "have "]):
        return f"{text}?"

    return f"Is the following statement true: {text}?"


def fallback_constraint_questions(fact: Dict[str, Any]) -> List[Dict[str, str]]:
    c = fact.get("constraint", {})
    text = fact["text"].strip().rstrip(".")
    out = []
    if c.get("time"):
        out.append({"type": "time", "question": f"When did this happen: {text}?"})
    if c.get("quantity"):
        lower = text.lower()
        if "population" in lower:
            q = f"What is the population mentioned in this statement: {text}?"
        elif "capacity" in lower:
            q = f"What is the capacity mentioned in this statement: {text}?"
        else:
            q = f"How many or how much is mentioned in this statement: {text}?"
        out.append({"type": "quantity", "question": q})
    return out


def fallback_search_hints(fact: Dict[str, Any]) -> List[str]:
    text = fact["text"]
    hints = [text]
    c = fact.get("constraint", {})
    if c.get("time"):
        hints.append(f"{text} {' '.join(c['time'])}")
    if c.get("quantity"):
        hints.append(f"{text} {' '.join(c['quantity'])}")
    return dedup_keep_order(hints)


def fallback_question_plan(claim: str, decomposition: Dict[str, Any]) -> Dict[str, Any]:
    items = []
    for fact in decomposition.get("atomic_facts", []):
        main_q = fallback_main_question(fact)
        qtype = normalize_question_type(None, main_q)
        if constraint_kind(fact.get("constraint", {})) in {"time", "quantity"}:
            # For time/quantity constraints, prefer WH as main question.
            cqs = []
        else:
            cqs = []
        items.append({
            "fact_id": fact["id"],
            "fact_text": fact["text"],
            "rely_on": fact.get("rely_on", []),
            "constraint": fact.get("constraint", {}),
            "main_question": main_q,
            "question_type": qtype,
            "constraint_questions": cqs,
            "search_hints": fallback_search_hints(fact),
        })
    return {
        "claim": claim,
        "decomposition_type": decomposition.get("decomposition_type", "simple"),
        "question_items": items,
    }


def parse_question_output(output_text: str) -> Dict[str, Any]:
    return try_parse_json(output_text)


def normalize_question_plan(parsed: Dict[str, Any], claim: str, decomposition: Dict[str, Any]) -> Dict[str, Any]:
    items = parsed.get("question_items", [])
    if not isinstance(items, list):
        items = []

    fact_map = {fact["id"]: fact for fact in decomposition.get("atomic_facts", [])}
    normalized = []
    used = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        fact_id = _clean_text(item.get("fact_id", ""))
        if fact_id not in fact_map or fact_id in used:
            continue
        used.add(fact_id)
        normalized.append(normalize_question_item(item, fact_map[fact_id]))

    for fact in decomposition.get("atomic_facts", []):
        if fact["id"] in used:
            continue
        main_q = fallback_main_question(fact)
        normalized.append({
            "fact_id": fact["id"],
            "fact_text": fact["text"],
            "rely_on": fact.get("rely_on", []),
            "constraint": fact.get("constraint", {}),
            "main_question": main_q,
            "question_type": normalize_question_type(None, main_q),
            "constraint_questions": [],
            "search_hints": fallback_search_hints(fact),
        })

    return {
        "claim": claim,
        "decomposition_type": decomposition.get("decomposition_type", "simple"),
        "question_items": normalized,
    }


def validate_question_plan(plan: Dict[str, Any], decomposition: Dict[str, Any]) -> bool:
    items = plan.get("question_items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("question_items is empty")

    fact_ids = [fact["id"] for fact in decomposition.get("atomic_facts", [])]
    plan_fact_ids = []
    for item in items:
        for key in ["fact_id", "fact_text", "rely_on", "constraint", "main_question", "question_type", "constraint_questions", "search_hints"]:
            if key not in item:
                raise ValueError(f"Missing field {key} in question item: {item}")
        if item["fact_id"] not in fact_ids:
            raise ValueError(f"Unknown fact_id: {item['fact_id']}")
        if item["question_type"] not in ALLOWED_QUESTION_TYPES:
            raise ValueError(f"Invalid question_type: {item['question_type']}")
        if not item["main_question"] or not item["main_question"].endswith("?"):
            raise ValueError(f"Invalid main_question: {item}")
        if not isinstance(item["search_hints"], list):
            raise ValueError(f"search_hints must be a list: {item}")
        for cq in item["constraint_questions"]:
            if cq.get("type") not in {"time", "quantity"}:
                raise ValueError(f"constraint question type must be time or quantity: {item}")
            q = cq.get("question", "")
            if not q.endswith("?"):
                raise ValueError(f"constraint question must end with ?: {item}")
            lower_q = q.lower().strip()
            if cq.get("type") == "time" and not (lower_q.startswith("when ") or lower_q.startswith("in what year") or lower_q.startswith("during which") or lower_q.startswith("on what date")):
                raise ValueError(f"time constraint question should be WH-style: {item}")
            if cq.get("type") == "quantity" and not (lower_q.startswith("how many ") or lower_q.startswith("how much ") or lower_q.startswith("what is the population") or lower_q.startswith("what is the capacity") or lower_q.startswith("what number")):
                raise ValueError(f"quantity constraint question should be WH-style: {item}")
        plan_fact_ids.append(item["fact_id"])

    if set(plan_fact_ids) != set(fact_ids):
        raise ValueError("question plan does not cover all atomic facts")
    return True


def find_question_issues(plan: Dict[str, Any], decomposition: Dict[str, Any]) -> List[str]:
    issues = []
    fact_map = {fact["id"]: fact for fact in decomposition.get("atomic_facts", [])}
    for item in plan.get("question_items", []):
        fact = fact_map[item["fact_id"]]
        ck = constraint_kind(fact.get("constraint", {}))
        q = item["main_question"].lower()
        if ck == "time" and not (q.startswith("when ") or q.startswith("in what year") or q.startswith("during which") or q.startswith("on what date")):
            issues.append(f"Main question for {item['fact_id']} should preferably be a WH time question")
        if ck == "quantity" and not (q.startswith("how many ") or q.startswith("how much ") or q.startswith("what is the population") or q.startswith("what is the capacity") or q.startswith("what number")):
            issues.append(f"Main question for {item['fact_id']} should preferably be a WH quantity question")
        if fact.get("constraint", {}).get("negation") is True and item["constraint_questions"]:
            issues.append(f"Negation-only fact {item['fact_id']} should not need extra constraint questions")
    return sorted(set(issues))


def repair_question_plan(claim: str, decomposition: Dict[str, Any], question_plan: Dict[str, Any], issues: List[str], plan: str, port: str) -> Dict[str, Any]:
    prompt = get_question_repair_prompt(claim, decomposition, question_plan, issues)
    output = llm(prompt, plan=plan, port=port)
    parsed = parse_question_output(output)
    repaired = normalize_question_plan(parsed, claim, decomposition)
    validate_question_plan(repaired, decomposition)
    return repaired


def generate_question_plan(claim: str, decomposition: Dict[str, Any], plan: str, port: str):
    prompt = get_question_prompt(claim, decomposition)
    infer_count = 0
    last_error = None

    while infer_count < 3:
        try:
            output = llm(prompt, plan=plan, port=port)
            parsed = parse_question_output(output)
            result = normalize_question_plan(parsed, claim, decomposition)
            validate_question_plan(result, decomposition)
            issues = find_question_issues(result, decomposition)
            if not issues:
                return result, [], False

            repaired = result
            repair_count = 0
            while repair_count < 2:
                try:
                    repaired = repair_question_plan(claim, decomposition, repaired, issues, plan, port)
                    new_issues = find_question_issues(repaired, decomposition)
                    if not new_issues:
                        return repaired, issues, True
                    issues = new_issues
                    repair_count += 1
                except Exception as e:
                    print(f"[Repair Retry {repair_count + 1}/2] Error during question repair: {e}")
                    repair_count += 1
            return repaired, issues, True
        except Exception as e:
            last_error = e
            print(f"[Retry {infer_count + 1}/3] Error during question generation: {e}")
            infer_count += 1

    fallback = fallback_question_plan(claim, decomposition)
    return fallback, [f"question_generation_failed: {str(last_error)}"], False


def process_data_item(data: Dict[str, Any], plan: str, port: str) -> Dict[str, Any]:
    claim = data["claim"]
    decomposition = data.get("atomic_facts", {})
    if not isinstance(decomposition, dict) or not decomposition.get("atomic_facts"):
        decomposition = {
            "claim": claim,
            "decomposition_type": "simple",
            "atomic_facts": [{
                "id": "f1",
                "text": claim,
                "rely_on": [],
                "constraint": {"negation": None, "time": [], "quantity": []},
            }],
        }
    question_plan, q_issues, q_is_repaired = generate_question_plan(claim, decomposition, plan, port)
    out = dict(data)
    out["question_plan"] = question_plan
    out["question_issues"] = q_issues
    out["question_is_repaired"] = q_is_repaired
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

    partial_func = partial(process_data_item, plan=args.plan, port=args.port)
    results = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(partial_func, data) for data in raws[args.start: args.end]]
        for f in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(f.result())
            except Exception as e:
                print(e)
                continue

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
    parser.add_argument('--dataset', type=str, default='HOVER_subset', help='Dataset name')
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
