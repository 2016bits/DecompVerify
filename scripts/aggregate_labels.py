import json
import argparse
import os
import re
from datetime import datetime


MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_label_3way(label):
    label = normalize_text(label).lower()
    if label in {"supports", "support", "supported"}:
        return "supports"
    if label in {"refutes", "refute", "refuted", "contradict", "contradicted"}:
        return "refutes"
    return "not enough info"


def map_3way_to_2way(label):
    label = normalize_label_3way(label)
    if label == "supports":
        return "supports"
    return "refutes"


def safe_float(text):
    try:
        return float(str(text).replace(",", "").strip())
    except Exception:
        return None


def extract_year(text):
    if text is None:
        return None
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", str(text))
    if m:
        return int(m.group(1))
    return None


def parse_date_like(text):
    """
    尽量把 answer 解析成可比较的时间值。
    优先返回：
    - 完整 datetime
    - 年份 int
    """
    if text is None:
        return None

    s = normalize_text(text).lower()

    # YYYY-MM-DD
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass

    # Month day, year
    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),\s*(\d{4})\b",
        s
    )
    if m:
        month = MONTHS[m.group(1)]
        day = int(m.group(2))
        year = int(m.group(3))
        try:
            return datetime(year, month, day)
        except Exception:
            pass

    # Month year
    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b",
        s
    )
    if m:
        month = MONTHS[m.group(1)]
        year = int(m.group(2))
        try:
            return datetime(year, month, 1)
        except Exception:
            pass

    # bare year
    year = extract_year(s)
    if year is not None:
        return year

    return None


def compare_values(left, right, operator):
    """
    支持数值 / 时间 / 文本相等比较
    """
    if left is None or right is None:
        return None

    # datetime 对 datetime
    if isinstance(left, datetime) and isinstance(right, datetime):
        if operator == "=":
            return left == right
        if operator == ">":
            return left > right
        if operator == "<":
            return left < right
        if operator == ">=":
            return left >= right
        if operator == "<=":
            return left <= right
        if operator == "before":
            return left < right
        if operator == "after":
            return left > right
        if operator == "earlier_than":
            return left < right
        if operator == "later_than":
            return left > right

    # int year / int year
    if isinstance(left, int) and isinstance(right, int):
        if operator == "=":
            return left == right
        if operator == ">":
            return left > right
        if operator == "<":
            return left < right
        if operator == ">=":
            return left >= right
        if operator == "<=":
            return left <= right
        if operator == "before":
            return left < right
        if operator == "after":
            return left > right
        if operator == "earlier_than":
            return left < right
        if operator == "later_than":
            return left > right

    # float / float
    lf = safe_float(left)
    rf = safe_float(right)
    if lf is not None and rf is not None:
        if operator == "=":
            return lf == rf
        if operator == ">":
            return lf > rf
        if operator == "<":
            return lf < rf
        if operator == ">=":
            return lf >= rf
        if operator == "<=":
            return lf <= rf

    # 文本比较
    ls = normalize_text(left).lower()
    rs = normalize_text(right).lower()
    if operator in {"=", "same_as"}:
        return ls == rs
    if operator == "different_from":
        return ls != rs

    return None


def build_fact_maps(answer_plan, fact_verification):
    answer_map = {item["fact_id"]: item for item in answer_plan}
    verify_map = {item["fact_id"]: item for item in fact_verification}
    return answer_map, verify_map


def evaluate_single_constraint(constraint, answer_map, verify_map):
    """
    返回:
    - "support"
    - "contradict"
    - "insufficient"
    """
    ctype = normalize_text(constraint.get("type", "")).lower()
    target_facts = constraint.get("target_facts", [])
    operator = normalize_text(constraint.get("operator", ""))
    value = constraint.get("value", "")

    # target facts 必须都存在
    for fid in target_facts:
        if fid not in verify_map or fid not in answer_map:
            return "insufficient"

    # 先看 target facts 自身是否已被否定
    for fid in target_facts:
        vlabel = verify_map[fid]["verification_label"]
        if vlabel == "contradict":
            return "contradict"
        if vlabel == "insufficient":
            return "insufficient"

    # negation：目标事实若 support，则整体 contradict；若 contradict，则整体 support
    if ctype == "negation":
        if len(target_facts) != 1:
            return "insufficient"
        fid = target_facts[0]
        base = verify_map[fid]["verification_label"]
        if base == "support":
            return "contradict"
        if base == "contradict":
            return "support"
        return "insufficient"

    # 其余 constraint 读取 answer
    answers = [answer_map[fid].get("answer", "") for fid in target_facts]

    # comparison: compare fact1 answer vs fact2 answer
    if ctype == "comparison":
        if len(answers) != 2:
            return "insufficient"

        left = parse_date_like(answers[0])
        right = parse_date_like(answers[1])

        if left is None or right is None:
            # fallback number/text
            left = answers[0]
            right = answers[1]

        ok = compare_values(left, right, operator)
        if ok is None:
            return "insufficient"
        return "support" if ok else "contradict"

    # time / quantity: compare fact1 answer vs fixed value
    if ctype in {"time", "quantity"}:
        if len(answers) != 1:
            return "insufficient"

        left = answers[0]

        if ctype == "time":
            left_v = parse_date_like(left)
            right_v = parse_date_like(value)
        else:
            left_v = safe_float(left)
            right_v = safe_float(value)
            if left_v is None or right_v is None:
                # fallback 文本比较
                left_v = left
                right_v = value

        ok = compare_values(left_v, right_v, operator)
        if ok is None:
            return "insufficient"
        return "support" if ok else "contradict"

    return "insufficient"


def aggregate_labels(decomposition, fact_verification, answer_plan, class_num="2"):
    """
    先聚合 fact_verification，再考虑 constraints。
    返回:
    {
      "predicted_label_3way": "...",
      "predicted_label": "...",
      "aggregation_trace": {...}
    }
    """
    constraints = decomposition.get("constraints", []) if decomposition else []
    answer_map, verify_map = build_fact_maps(answer_plan, fact_verification)

    base_labels = [item["verification_label"] for item in fact_verification]

    # 1. 先看子事实本身
    if any(lbl == "contradict" for lbl in base_labels):
        pred_3way = "refutes"
        return {
            "predicted_label_3way": pred_3way,
            "predicted_label": map_3way_to_2way(pred_3way) if str(class_num) == "2" else pred_3way,
            "aggregation_trace": {
                "fact_labels": base_labels,
                "constraint_results": [],
                "decision": "At least one fact is contradicted."
            }
        }

    # 2. 如果没有 contradicted，但有 insufficient，先记着，后面 constraints 也可能 insufficient
    has_insufficient_fact = any(lbl == "insufficient" for lbl in base_labels)

    constraint_results = []
    for constraint in constraints:
        cres = evaluate_single_constraint(constraint, answer_map, verify_map)
        constraint_results.append({
            "constraint": constraint,
            "result": cres
        })

    # 3. constraints 优先级
    if any(item["result"] == "contradict" for item in constraint_results):
        pred_3way = "refutes"
        return {
            "predicted_label_3way": pred_3way,
            "predicted_label": map_3way_to_2way(pred_3way) if str(class_num) == "2" else pred_3way,
            "aggregation_trace": {
                "fact_labels": base_labels,
                "constraint_results": constraint_results,
                "decision": "A constraint is contradicted."
            }
        }

    if has_insufficient_fact or any(item["result"] == "insufficient" for item in constraint_results):
        pred_3way = "not enough info"
        return {
            "predicted_label_3way": pred_3way,
            "predicted_label": map_3way_to_2way(pred_3way) if str(class_num) == "2" else pred_3way,
            "aggregation_trace": {
                "fact_labels": base_labels,
                "constraint_results": constraint_results,
                "decision": "At least one fact or constraint is insufficient."
            }
        }

    # 4. 全部 support
    pred_3way = "supports"
    return {
        "predicted_label_3way": pred_3way,
        "predicted_label": map_3way_to_2way(pred_3way) if str(class_num) == "2" else pred_3way,
        "aggregation_trace": {
            "fact_labels": base_labels,
            "constraint_results": constraint_results,
            "decision": "All facts and constraints are supported."
        }
    }


def process_data_item(data, class_num):
    claim = data["claim"]
    evidence = data.get("gold_evidence", data.get("evidence", ""))
    decomposition = data.get("decomposition", data.get("atomic_facts", None))
    if decomposition is None:
        decomposition = {
            "claim": claim,
            "atomic_facts": [],
            "constraints": []
        }

    question_plan = data.get("question_plan", [])
    answer_plan = data.get("answer_plan", [])
    final_bindings = data.get("final_bindings", {})
    fact_verification = data.get("fact_verification", [])

    aggregation_result = aggregate_labels(
        decomposition=decomposition,
        fact_verification=fact_verification,
        answer_plan=answer_plan,
        class_num=class_num
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
        "fact_verification": fact_verification,
        "predicted_label": aggregation_result["predicted_label"],
        "predicted_label_3way": aggregation_result["predicted_label_3way"],
        "aggregation_trace": aggregation_result["aggregation_trace"]
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
    )

    with open(in_path, "r", encoding="utf-8") as f:
        raws = json.load(f)

    # raws = raws[args.start: args.end]

    results = []
    for data in raws:
        try:
            res = process_data_item(data, class_num=args.class_num)
            results.append(res)
        except Exception as e:
            print(f"Error processing {data.get('id', 'unknown')}: {e}")
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
    parser.add_argument('--end', type=int, default=2000, help='End index')

    parser.add_argument(
        '--in_path',
        type=str,
        default='./data/[DATA]/plan2/[TYPE]_[CLASS]_fact_verify_[T][S]_[E].json',
        help='Input path template'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='./data/[DATA]/plan2/[TYPE]_[CLASS]_final_[T][S]_[E].json',
        help='Output path template'
    )
    parser.add_argument('--t', type=str, default='')

    args = parser.parse_args()
    main(args)
    