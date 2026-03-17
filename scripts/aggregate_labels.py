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

    # 只把明确 support / refute 纳入约束判定，不做强一票否决
    for fid in target_facts:
        vlabel = normalize_text(verify_map[fid].get("verification_label", "")).lower()
        if vlabel == "insufficient":
            return "insufficient"

    # 只把明确 support / refute 纳入约束判定，不做强一票否决
    for fid in target_facts:
        vlabel = normalize_text(verify_map[fid].get("verification_label", "")).lower()
        if vlabel == "insufficient":
            return "insufficient"

    if ctype == "negation":
        if len(target_facts) != 1:
            return "insufficient"
        fid = target_facts[0]
        base = normalize_text(verify_map[fid].get("verification_label", "")).lower()
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

def get_fact_role_weight(fact_item):
    role = normalize_text(fact_item.get("role", "verify")).lower()
    if role == "verify":
        return 1.0
    if role == "bridge":
        return 0.8
    if role == "anchor":
        return 0.6
    return 0.8

def get_fact_source_weight(fact_item):
    # 给后续扩展留接口；如果 verify_atomic_facts.py 以后加入 source/confidence，这里就能直接用
    source = normalize_text(fact_item.get("source", "verified")).lower()
    if source == "verified":
        return 1.0
    if source == "answer_fallback":
        return 0.7
    if source == "parse_fallback":
        return 0.4
    if source == "timeout_fallback":
        return 0.3
    return 0.8

def get_fact_confidence(fact_item):
    conf = fact_item.get("confidence", None)
    try:
        conf = float(conf)
        return max(0.0, min(1.0, conf))
    except Exception:
        return 1.0


def score_fact(fact_item):
    """
    返回:
    {
        "label_3way": "...",
        "score": float,
        "hard_contradict": bool,
        "trace": {...}
    }
    """
    raw_label = fact_item.get("verification_label", "")
    label = normalize_text(raw_label).lower()

    role_w = get_fact_role_weight(fact_item)
    src_w = get_fact_source_weight(fact_item)
    conf_w = get_fact_confidence(fact_item)
    weight = role_w * src_w * conf_w
    
    role = normalize_text(fact_item.get("role", "verify")).lower()
    source = normalize_text(fact_item.get("source", "verified")).lower()

    if label == "support":
        # support 给分稍弱一点，避免几个弱 support 就把整体顶上去
        base = 0.9 if role == "verify" else 0.55
        return {
            "label_3way": "supports",
            "score": base * weight,
            "hard_contradict": False,
            "trace": {
                "raw_label": raw_label,
                "role_weight": role_w,
                "source_weight": src_w,
                "confidence": conf_w,
                "final_weight": weight
            }
        }

    if label == "contradict":
        # contradict 给分更重，尤其是 verify role
        base = -1.35 if role == "verify" else -0.9
        hard_contradict = (
            source in {"verified", "answer_fallback"}
            and conf_w >= 0.6
            and role == "verify"
        )
        return {
            "label_3way": "refutes",
            "score": base * weight,
            "hard_contradict": hard_contradict,
            "trace": {
                "raw_label": raw_label,
                "role_weight": role_w,
                "source_weight": src_w,
                "confidence": conf_w,
                "final_weight": weight
            }
        }

    # insufficient 不再是 0 分，给一个轻微负分
    # 因为 support 应该依赖完整证据链，不确定不能帮助 support 成立
    base = -0.1 if role == "verify" else -0.05
    return {
        "label_3way": "not enough info",
        "score": base * weight,
        "hard_contradict": False,
        "trace": {
            "raw_label": raw_label,
            "role_weight": role_w,
            "source_weight": src_w,
            "confidence": conf_w,
            "final_weight": weight
        }
    }


def score_constraint_result(result, constraint):
    ctype = normalize_text(constraint.get("type", "")).lower()

    # comparison / time / quantity 比普通 constraint 更关键一点
    if ctype in {"comparison", "time", "quantity", "negation"}:
        support_w = 0.8
        refute_w = -1.4
    else:
        support_w = 0.6
        refute_w = -1.0

    if result == "support":
        return support_w
    if result == "contradict":
        return refute_w
    return -0.15

def map_3way_to_2way(label, nei_to="refutes"):
    """
    nei_to:
    - "refutes": 保守策略
    - "supports": 激进策略
    """
    label = normalize_label_3way(label)
    if label == "supports":
        return "supports"
    if label == "refutes":
        return "refutes"
    return "supports" if nei_to == "supports" else "refutes"

def aggregate_labels(decomposition, fact_verification, answer_plan, class_num="2", support_threshold=1.0, refute_threshold=-1.0, nei_to="refutes"):
    """
    新版聚合：
    1. fact / constraint 先分别打分
    2. 再看总分
    3. 保留 3-way，再根据参数映射到 2-way
    """
    constraints = decomposition.get("constraints", []) if decomposition else []
    answer_map, verify_map = build_fact_maps(answer_plan, fact_verification)
    
    fact_details = []
    fact_score_total = 0.0
    hard_contradict = False
    num_support = 0
    num_refute = 0
    num_nei = 0

    for item in fact_verification:
        scored = score_fact(item)
        fact_score_total += scored["score"]
        hard_contradict = hard_contradict or scored["hard_contradict"]

        if scored["label_3way"] == "supports":
            num_support += 1
        elif scored["label_3way"] == "refutes":
            num_refute += 1
        else:
            num_nei += 1

        fact_details.append({
            "fact_id": item.get("fact_id"),
            "verification_label": item.get("verification_label"),
            "score": scored["score"],
            "hard_contradict": scored["hard_contradict"],
            "trace": scored["trace"]
        })

    constraint_results = []
    constraint_score_total = 0.0
    constraint_refute_count = 0
    constraint_support_count = 0
    constraint_nei_count = 0

    for constraint in constraints:
        cres = evaluate_single_constraint(constraint, answer_map, verify_map)
        cscore = score_constraint_result(cres, constraint)
        constraint_score_total += cscore

        if cres == "support":
            constraint_support_count += 1
        elif cres == "contradict":
            constraint_refute_count += 1
        else:
            constraint_nei_count += 1

        constraint_results.append({
            "constraint": constraint,
            "result": cres,
            "score": cscore
        })

    total_score = fact_score_total + constraint_score_total

    # 判决逻辑
    # 1) 高置信硬反驳优先
    # 2) 否则看总分
    # 3) 分数不够则给 NEI
        # 更细的计数
    verify_support = 0
    verify_refute = 0
    verify_nei = 0
    bridge_anchor_support = 0
    bridge_anchor_refute = 0

    for item in fact_verification:
        role = normalize_text(item.get("role", "verify")).lower()
        vlabel = normalize_text(item.get("verification_label", "")).lower()

        if role == "verify":
            if vlabel == "support":
                verify_support += 1
            elif vlabel == "contradict":
                verify_refute += 1
            else:
                verify_nei += 1
        else:
            if vlabel == "support":
                bridge_anchor_support += 1
            elif vlabel == "contradict":
                bridge_anchor_refute += 1

    # 更保守的判决逻辑
    if hard_contradict:
        pred_3way = "refutes"
        decision = "Hard contradiction from verify fact."
    elif constraint_refute_count > 0:
        pred_3way = "refutes"
        decision = "At least one constraint is contradicted."
    elif total_score <= refute_threshold:
        pred_3way = "refutes"
        decision = "Total score falls below refute threshold."
    elif (
        total_score >= support_threshold
        and verify_refute == 0
        and constraint_refute_count == 0
        and verify_support >= max(1, verify_nei)
    ):
        pred_3way = "supports"
        decision = "Sufficient positive support with no explicit contradiction."
    else:
        pred_3way = "not enough info"
        decision = "Evidence is mixed or insufficient under current thresholds."

    pred_label = (
        map_3way_to_2way(pred_3way, nei_to=nei_to)
        if str(class_num) == "2"
        else pred_3way
    )

    return {
        "predicted_label_3way": pred_3way,
        "predicted_label": pred_label,
        "aggregation_trace": {
            "fact_details": fact_details,
            "constraint_results": constraint_results,
            "fact_score_total": fact_score_total,
            "constraint_score_total": constraint_score_total,
            "total_score": total_score,
            "num_support_facts": num_support,
            "num_refute_facts": num_refute,
            "num_nei_facts": num_nei,
            "num_support_constraints": constraint_support_count,
            "num_refute_constraints": constraint_refute_count,
            "num_nei_constraints": constraint_nei_count,
            "hard_contradict": hard_contradict,
            "support_threshold": support_threshold,
            "refute_threshold": refute_threshold,
            "verify_support_facts": verify_support,
            "verify_refute_facts": verify_refute,
            "verify_nei_facts": verify_nei,
            "bridge_anchor_support_facts": bridge_anchor_support,
            "bridge_anchor_refute_facts": bridge_anchor_refute,
            "nei_to": nei_to,
            "decision": decision
        }
    }


def process_data_item(data, class_num, support_threshold=1.0, refute_threshold=-1.0, nei_to="refutes"):
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
        class_num=class_num,
        support_threshold=support_threshold,
        refute_threshold=refute_threshold,
        nei_to=nei_to,
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
        .replace("[PLAN]", args.plan)
    )

    with open(in_path, "r", encoding="utf-8") as f:
        raws = json.load(f)

    # raws = raws[args.start: args.end]

    results = []
    for data in raws:
        try:
            res = process_data_item(data, class_num=args.class_num, 
                                    support_threshold=args.support_threshold,
                                    refute_threshold=args.refute_threshold,
                                    nei_to=args.nei_to)
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

    parser.add_argument(
        '--in_path',
        type=str,
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_fact_verify_[T][S]_[E].json',
        help='Input path template'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_final_[T][S]_[E].json',
        help='Output path template'
    )
    
    parser.add_argument('--plan', type=str, default='plan2.1', help='Which plan version to use for aggregation logic')
    parser.add_argument('--t', type=str, default='')
    parser.add_argument('--support_threshold', type=float, default=1.6)
    parser.add_argument('--refute_threshold', type=float, default=-0.8)
    parser.add_argument('--nei_to', type=str, default="refutes")

    args = parser.parse_args()
    main(args)
    