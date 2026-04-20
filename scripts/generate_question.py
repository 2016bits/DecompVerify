import os
import re
import json
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm
from openai import OpenAI

from prompts.question_prompts import get_question_prompt


VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")
NEGATION_STRIP_PATTERNS = [
    (re.compile(r"\bdo not have\b", re.IGNORECASE), "have"),
    (re.compile(r"\bdoes not have\b", re.IGNORECASE), "has"),
    (re.compile(r"\bdid not have\b", re.IGNORECASE), "had"),
    (re.compile(r"\bdo not\b", re.IGNORECASE), "do"),
    (re.compile(r"\bdoes not\b", re.IGNORECASE), "does"),
    (re.compile(r"\bdid not\b", re.IGNORECASE), "did"),
    (re.compile(r"\bis not\b", re.IGNORECASE), "is"),
    (re.compile(r"\bare not\b", re.IGNORECASE), "are"),
    (re.compile(r"\bwas not\b", re.IGNORECASE), "was"),
    (re.compile(r"\bwere not\b", re.IGNORECASE), "were"),
    (re.compile(r"\bhas not\b", re.IGNORECASE), "has"),
    (re.compile(r"\bhave not\b", re.IGNORECASE), "have"),
    (re.compile(r"\bno longer\b", re.IGNORECASE), ""),
    (re.compile(r"\bnever\b", re.IGNORECASE), ""),
    (re.compile(r"\bwithout\b", re.IGNORECASE), "with"),
    (re.compile(r"\bnot the same as\b", re.IGNORECASE), "the same as"),
    (re.compile(r"\bnot the same\b", re.IGNORECASE), "the same"),
    (re.compile(r"\bnot equal to\b", re.IGNORECASE), "equal to"),
    (re.compile(r"\bnot equal\b", re.IGNORECASE), "equal"),
    (re.compile(r"\bnot\b", re.IGNORECASE), ""),
]



def build_client_and_model(plan, port='8370'):
    plan = (plan or "").lower()

    if plan in {"local", "localhost", "vllm"} or plan.startswith("plan"):
        client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        return client, "Meta-Llama-3-70B-Instruct-AutoAWQ-4bit", None, None, None, 0.0

    if plan.startswith("azure"):
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set in environment variables.")
        client = OpenAI(api_key=api_key, base_url="https://open1027.openai.azure.com/openai/v1/")
        return client, "gpt-4o", None, {}, 512, 0.7

    if plan.startswith("qwen_plan"):
        client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        return client, "qwen2.5-72b-awq", "You are a helpful assistant.", None, None, 0.7

    if plan.startswith("qc_plan"):
        api_key = os.getenv("AIPING_API_KEY")
        if not api_key:
            raise ValueError("AIPING_API_KEY is not set in environment variables.")
        client = OpenAI(api_key=api_key, base_url="https://www.aiping.cn/api/v1")
        return client, "DeepSeek-V3.2", None, {}, 2048, 0.7

    raise ValueError(
        f"Unsupported plan: {plan}. Supported plans are: local, azure, azure_gpt4o, qc_plan, qwen_plan"
    )



def llm(input_text, plan='local', port='8370'):
    client, model, system_prompt, extra_body, max_tokens, temperature = build_client_and_model(plan, port)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": input_text})

    kwargs = {"model": model, "messages": messages}
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
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
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
    match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    first_brace = text.find("{")
    if first_brace == -1:
        raise ValueError("No JSON found.")

    stack = 0
    in_string = False
    escape = False
    for idx in range(first_brace, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0:
                return text[first_brace:idx + 1]
    raise ValueError("Incomplete JSON block.")



def parse_question_output(output_text):
    return json.loads(extract_json_block(output_text))



def _clean_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()



def infer_answer_slot(question_type, fact_text, main_question):
    fact_text_l = _clean_text(fact_text).lower()
    q_l = _clean_text(main_question).lower()

    explicit_vars = [slot for slot in VAR_PATTERN.findall(fact_text) if slot.startswith("?")]
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
    for pattern, slot in patterns:
        if re.search(pattern, q_l):
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



def strip_fact_negation(text):
    text = _clean_text(text)
    if not text:
        return text
    out = text
    for pattern, repl in NEGATION_STRIP_PATTERNS:
        out = pattern.sub(repl, out)
    out = re.sub(r"\s+", " ", out).strip(" .")
    return out or text



def build_yesno_question(statement):
    statement = _clean_text(statement).rstrip("?.")
    if not statement:
        return ""
    return f"Is it true that {statement}?"



def is_fallback_claim_fact(fact):
    return "fallback_claim_fact" in (fact.get("critical_reasons", []) or [])



def choose_fallback_question(fact, force_relation_yesno=False):
    fact_text = _clean_text(fact.get("text", ""))
    constraint = fact.get("constraint", {}) or {}
    positive_text = strip_fact_negation(fact_text) if constraint.get("negation") is True else fact_text

    if constraint.get("negation") is True:
        return "relation_yesno", build_yesno_question(positive_text), "positive_probe_for_negation"
    if force_relation_yesno:
        return "relation_yesno", build_yesno_question(fact_text), "literal_fact"
    if constraint.get("time"):
        return "time_wh", f"When did {positive_text}?", "literal_fact"
    if constraint.get("quantity"):
        return "quantity_wh", f"What is the quantity for: {positive_text}?", "literal_fact"
    if re.search(r"\b(is|are|was|were|has|have|did|does|do|can|could|will|would|should)\b", fact_text.lower()):
        return "relation_yesno", build_yesno_question(fact_text), "literal_fact"
    return "entity_wh", f"What entity satisfies: {positive_text}?", "literal_fact"



def normalize_question_item(item, atomic_fact):
    if not isinstance(item, dict):
        item = {}

    qtype = _clean_text(item.get("question_type", ""))
    if qtype not in {"relation_yesno", "entity_wh", "time_wh", "quantity_wh"}:
        qtype = ""

    main_question = _clean_text(item.get("main_question", "") or item.get("question", ""))
    question_polarity = _clean_text(item.get("question_polarity", ""))

    constraint = atomic_fact.get("constraint", {}) or {}
    if is_fallback_claim_fact(atomic_fact):
        qtype, main_question, question_polarity = choose_fallback_question(atomic_fact, force_relation_yesno=True)
    elif not qtype or not main_question:
        qtype, main_question, question_polarity = choose_fallback_question(atomic_fact)

    if constraint.get("negation") is True:
        positive_text = strip_fact_negation(atomic_fact.get("text", ""))
        if qtype != "relation_yesno":
            qtype = "relation_yesno"
        if not question_polarity or question_polarity != "positive_probe_for_negation":
            question_polarity = "positive_probe_for_negation"
        main_question = build_yesno_question(positive_text)
    elif not question_polarity:
        question_polarity = "literal_fact"

    search_hints = item.get("search_hints", [])
    if not isinstance(search_hints, list):
        search_hints = [search_hints]
    search_hints = [_clean_text(value) for value in search_hints if _clean_text(value)]
    if atomic_fact.get("text") and atomic_fact.get("text") not in search_hints:
        search_hints.append(atomic_fact.get("text"))
    if constraint.get("negation") is True:
        positive_hint = strip_fact_negation(atomic_fact.get("text", ""))
        if positive_hint and positive_hint not in search_hints:
            search_hints.append(positive_hint)

    constraint_questions = item.get("constraint_questions", [])
    if not isinstance(constraint_questions, list):
        constraint_questions = []

    answer_slot = _clean_text(item.get("answer_slot", ""))
    if is_fallback_claim_fact(atomic_fact):
        answer_slot = None
    elif not answer_slot:
        answer_slot = infer_answer_slot(qtype, atomic_fact.get("text", ""), main_question)

    return {
        "fact_id": atomic_fact.get("id", ""),
        "fact_text": atomic_fact.get("text", ""),
        "rely_on": atomic_fact.get("rely_on", []),
        "constraint": constraint,
        "coverage": atomic_fact.get("coverage", []),
        "critical": atomic_fact.get("critical", False),
        "critical_reasons": atomic_fact.get("critical_reasons", []),
        "main_question": main_question,
        "question_type": qtype,
        "question_polarity": question_polarity,
        "answer_slot": answer_slot,
        "constraint_questions": constraint_questions,
        "search_hints": search_hints,
    }



def normalize_question_result(parsed, decomposition):
    qitems = parsed.get("question_items", [])
    if not isinstance(qitems, list):
        qitems = []

    fact_map = {fact["id"]: fact for fact in decomposition.get("atomic_facts", [])}
    normalized = []
    used = set()

    for item in qitems:
        fact_id = _clean_text(item.get("fact_id", ""))
        if fact_id in fact_map:
            normalized.append(normalize_question_item(item, fact_map[fact_id]))
            used.add(fact_id)

    for fact in decomposition.get("atomic_facts", []):
        if fact["id"] in used:
            continue
        normalized.append(normalize_question_item({}, fact))

    return {
        "claim": decomposition.get("claim", ""),
        "decomposition_type": decomposition.get("decomposition_type", "simple"),
        "question_items": normalized,
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
        except Exception as exc:
            last_error = exc
            print(f"[Retry {infer_count + 1}/3] Error during question generation: {exc}")
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

    with open(in_path, "r", encoding="utf-8") as file_obj:
        raws = json.load(file_obj)
    raws = raws[args.start:args.end]

    partial_func = partial(process_data_item, plan=args.plan, port=args.port)
    results = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(partial_func, data) for data in raws]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Error in future: {exc}")

    results = sorted(results, key=lambda item: item["id"])

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

    with open(out_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=4, ensure_ascii=False)

    print(f"Saved to {out_path}")
    print("Program finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HOVER")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--class_num", type=str, default="2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--port", type=str, default="8370")
    parser.add_argument("--plan", type=str, default="qc")
    parser.add_argument("--t", type=str, default="")
    parser.add_argument("--in_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_decomposed_[T][S]_[E].json")
    parser.add_argument("--out_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_questions_[T][S]_[E].json")
    main(parser.parse_args())
