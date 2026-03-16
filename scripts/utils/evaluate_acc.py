import json
import argparse
from collections import defaultdict, Counter


def normalize_label(label, class_num="2"):
    if label is None:
        return "unknown"

    label = str(label).strip().lower()

    if label in {"supports", "support", "supported"}:
        return "supports"
    if label in {"refutes", "refute", "refuted", "contradict", "contradicted"}:
        return "refutes"
    if label in {"not enough info", "nei", "insufficient", "unknown"}:
        return "not enough info"

    if class_num == "2":
        return "refutes"
    return label


def map_to_2way(label):
    label = normalize_label(label, class_num="3")
    if label == "supports":
        return "supports"
    return "refutes"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_confusion(y_true, y_pred, labels):
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for g, p in zip(y_true, y_pred):
        if g not in matrix:
            matrix[g] = {pred: 0 for pred in labels}
        if p not in matrix[g]:
            matrix[g][p] = 0
        matrix[g][p] += 1
    return matrix


def evaluate(data, class_num="2", use_3way=False):
    overall_gold = []
    overall_pred = []

    by_hop_gold = defaultdict(list)
    by_hop_pred = defaultdict(list)

    by_hop_gold_dist = defaultdict(Counter)
    by_hop_pred_dist = defaultdict(Counter)

    for item in data:
        gold = item.get("label")

        if use_3way:
            pred = item.get("predicted_label_3way", item.get("predicted_label"))
            gold = normalize_label(gold, class_num="3")
            pred = normalize_label(pred, class_num="3")
            label_space = ["supports", "refutes", "not enough info"]
        else:
            pred = item.get("predicted_label")
            gold = normalize_label(gold, class_num="2")
            pred = normalize_label(pred, class_num="2")
            gold = map_to_2way(gold)
            pred = map_to_2way(pred)
            label_space = ["supports", "refutes"]

        hop = item.get("num_hops", "unknown")

        overall_gold.append(gold)
        overall_pred.append(pred)

        by_hop_gold[hop].append(gold)
        by_hop_pred[hop].append(pred)

        by_hop_gold_dist[hop][gold] += 1
        by_hop_pred_dist[hop][pred] += 1

    overall_correct = sum(g == p for g, p in zip(overall_gold, overall_pred))
    overall_acc = overall_correct / len(overall_gold) if overall_gold else 0.0

    by_hop_metrics = {}
    for hop in sorted(by_hop_gold.keys(), key=lambda x: str(x)):
        golds = by_hop_gold[hop]
        preds = by_hop_pred[hop]
        correct = sum(g == p for g, p in zip(golds, preds))
        total = len(golds)
        acc = correct / total if total > 0 else 0.0

        by_hop_metrics[str(hop)] = {
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "gold_distribution": dict(by_hop_gold_dist[hop]),
            "pred_distribution": dict(by_hop_pred_dist[hop]),
        }

    confusion = compute_confusion(overall_gold, overall_pred, label_space)

    return {
        "total_samples": len(overall_gold),
        "class_num": class_num,
        "use_3way": use_3way,
        "overall_accuracy": overall_acc,
        "overall_gold_distribution": dict(Counter(overall_gold)),
        "overall_pred_distribution": dict(Counter(overall_pred)),
        "by_hop_metrics": by_hop_metrics,
        "confusion_matrix": confusion
    }


def print_report(report):
    print("=" * 80)
    print("Evaluation Report")
    print("=" * 80)
    print(f"Total samples: {report['total_samples']}")
    print(f"Class setting: {report['class_num']}")
    print(f"Use 3-way labels: {report['use_3way']}")
    print(f"Overall accuracy: {report['overall_accuracy']:.4f}")
    print()

    print("Overall gold distribution:")
    for k, v in report["overall_gold_distribution"].items():
        print(f"  {k}: {v}")
    print()

    print("Overall pred distribution:")
    for k, v in report["overall_pred_distribution"].items():
        print(f"  {k}: {v}")
    print()

    print("Per-hop metrics:")
    for hop, info in report["by_hop_metrics"].items():
        print(f"  Hop {hop}:")
        print(f"    total={info['total']}, correct={info['correct']}, accuracy={info['accuracy']:.4f}")
        print(f"    gold_distribution={info['gold_distribution']}")
        print(f"    pred_distribution={info['pred_distribution']}")
    print()

    print("Confusion matrix:")
    labels = list(report["confusion_matrix"].keys())
    header = "gold\\pred".ljust(18) + "".join(lbl.ljust(18) for lbl in labels)
    print(header)
    for gold in labels:
        row = gold.ljust(18)
        for pred in labels:
            row += str(report["confusion_matrix"][gold].get(pred, 0)).ljust(18)
        print(row)
    print("=" * 80)


def main(args):
    path1 = (
        args.in_path1
        .replace("[DATA]", args.dataset)
        .replace("[TYPE]", args.data_type)
        .replace("[CLASS]", args.class_num)
        .replace("[T]", args.t)
    )
    path2 = (
        args.in_path2
        .replace("[DATA]", args.dataset)
        .replace("[TYPE]", args.data_type)
        .replace("[CLASS]", args.class_num)
        .replace("[T]", args.t)
    )

    data1 = load_json(path1)
    data2 = load_json(path2)
    data = data1 + data2

    report = evaluate(
        data=data,
        class_num=args.class_num,
        use_3way=args.use_3way
    )

    print_report(report)

    if args.out_path:
        out_path = (
            args.out_path
            .replace("[DATA]", args.dataset)
            .replace("[TYPE]", args.data_type)
            .replace("[CLASS]", args.class_num)
            .replace("[T]", args.t)
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HOVER")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--class_num", type=str, default="2")
    parser.add_argument("--t", type=str, default="")
    parser.add_argument("--use_3way", action="store_true")

    parser.add_argument(
        "--in_path1",
        type=str,
        default="./data/[DATA]/plan2/[TYPE]_[CLASS]_final_[T]0_2000.json",
        help="First input file"
    )
    parser.add_argument(
        "--in_path2",
        type=str,
        default="./data/[DATA]/plan2/[TYPE]_[CLASS]_final_[T]2000_4000.json",
        help="Second input file"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/[DATA]/plan2/[TYPE]_[CLASS]_eval_by_hop[T].json",
        help="Output report file"
    )

    args = parser.parse_args()
    main(args)