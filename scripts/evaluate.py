
import os
import json
import argparse
from collections import Counter, defaultdict


def normalize_label(label, class_num="2"):
    if label is None:
        return "refutes"
    label = str(label).strip().lower()

    if label in {"supports", "support", "supported"}:
        return "supports"

    if label in {
        "refutes", "refute", "refuted",
        "contradict", "contradicted",
        "not enough info", "not enough information",
        "nei", "insufficient", "unknown"
    }:
        return "refutes"

    if class_num == "2":
        return "refutes"
    return label


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


def safe_div(a, b):
    return a / b if b else 0.0


def get_predicted_label(item):
    if "aggregation_result" in item:
        return item["aggregation_result"].get("final_label", "refutes")
    return item.get("predicted_label", "refutes")


def compute_binary_prf(y_true, y_pred, labels):
    prf = {}
    for label in labels:
        tp = sum(1 for g, p in zip(y_true, y_pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(y_true, y_pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(y_true, y_pred) if g == label and p != label)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

        prf[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    macro_precision = round(sum(prf[l]["precision"] for l in labels) / len(labels), 4)
    macro_recall = round(sum(prf[l]["recall"] for l in labels) / len(labels), 4)
    macro_f1 = round(sum(prf[l]["f1"] for l in labels) / len(labels), 4)

    return prf, macro_precision, macro_recall, macro_f1


def evaluate(data, class_num="2"):
    overall_gold = []
    overall_pred = []

    by_hop_gold = defaultdict(list)
    by_hop_pred = defaultdict(list)
    by_hop_gold_dist = defaultdict(Counter)
    by_hop_pred_dist = defaultdict(Counter)

    labels = ["supports", "refutes"]

    for item in data:
        gold = normalize_label(item.get("label"), class_num=class_num)
        pred = normalize_label(get_predicted_label(item), class_num=class_num)
        hop = item.get("num_hops", "unknown")

        overall_gold.append(gold)
        overall_pred.append(pred)

        by_hop_gold[hop].append(gold)
        by_hop_pred[hop].append(pred)

        by_hop_gold_dist[hop][gold] += 1
        by_hop_pred_dist[hop][pred] += 1

    overall_correct = sum(g == p for g, p in zip(overall_gold, overall_pred))
    overall_acc = safe_div(overall_correct, len(overall_gold))

    confusion = compute_confusion(overall_gold, overall_pred, labels)
    prf, macro_precision, macro_recall, macro_f1 = compute_binary_prf(overall_gold, overall_pred, labels)

    by_hop_metrics = {}
    for hop in sorted(by_hop_gold.keys(), key=lambda x: str(x)):
        golds = by_hop_gold[hop]
        preds = by_hop_pred[hop]
        correct = sum(g == p for g, p in zip(golds, preds))
        total = len(golds)
        acc = safe_div(correct, total)

        hop_prf, hop_macro_precision, hop_macro_recall, hop_macro_f1 = compute_binary_prf(golds, preds, labels)

        by_hop_metrics[str(hop)] = {
            "total": total,
            "correct": correct,
            "accuracy": round(acc, 4),
            "gold_distribution": dict(by_hop_gold_dist[hop]),
            "pred_distribution": dict(by_hop_pred_dist[hop]),
            "per_label_metrics": hop_prf,
            "macro_precision": hop_macro_precision,
            "macro_recall": hop_macro_recall,
            "macro_f1": hop_macro_f1,
        }

    return {
        "total_samples": len(overall_gold),
        "class_num": class_num,
        "label_setting": "binary: supports vs refutes (NEI merged into refutes)",
        "overall_accuracy": round(overall_acc, 4),
        "overall_gold_distribution": dict(Counter(overall_gold)),
        "overall_pred_distribution": dict(Counter(overall_pred)),
        "per_label_metrics": prf,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "by_hop_metrics": by_hop_metrics,
        "confusion_matrix": confusion,
    }


def print_report(report):
    print("=" * 80)
    print("Evaluation Report")
    print("=" * 80)
    print(f"Total samples: {report['total_samples']}")
    print(f"Class setting: {report['class_num']}")
    print(f"Label setting: {report['label_setting']}")
    print(f"Overall accuracy: {report['overall_accuracy']:.4f}")
    print(f"Macro precision: {report['macro_precision']:.4f}")
    print(f"Macro recall: {report['macro_recall']:.4f}")
    print(f"Macro F1: {report['macro_f1']:.4f}")
    print()

    print("Overall gold distribution:")
    for k, v in report["overall_gold_distribution"].items():
        print(f"  {k}: {v}")
    print()

    print("Overall pred distribution:")
    for k, v in report["overall_pred_distribution"].items():
        print(f"  {k}: {v}")
    print()

    print("Overall per-label metrics:")
    for label, info in report["per_label_metrics"].items():
        print(
            f"  {label}: "
            f"P={info['precision']:.4f}, "
            f"R={info['recall']:.4f}, "
            f"F1={info['f1']:.4f}, "
            f"TP={info['tp']}, FP={info['fp']}, FN={info['fn']}"
        )
    print()

    print("Per-hop metrics:")
    for hop, info in report["by_hop_metrics"].items():
        print(f"  Hop {hop}:")
        print(f"    total={info['total']}, correct={info['correct']}, accuracy={info['accuracy']:.4f}")
        print(
            f"    macro_precision={info['macro_precision']:.4f}, "
            f"macro_recall={info['macro_recall']:.4f}, "
            f"macro_f1={info['macro_f1']:.4f}"
        )
        print(f"    gold_distribution={info['gold_distribution']}")
        print(f"    pred_distribution={info['pred_distribution']}")
        for label, label_info in info["per_label_metrics"].items():
            print(
                f"    {label}: "
                f"P={label_info['precision']:.4f}, "
                f"R={label_info['recall']:.4f}, "
                f"F1={label_info['f1']:.4f}, "
                f"TP={label_info['tp']}, FP={label_info['fp']}, FN={label_info['fn']}"
            )
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
    if args.results_file_num == 2:
        path1 = (
            args.in_path1
            .replace("[DATA]", args.dataset)
            .replace("[TYPE]", args.data_type)
            .replace("[CLASS]", args.class_num)
            .replace("[T]", args.t)
            .replace("[PLAN]", args.plan)
        )
        path2 = (
            args.in_path2
            .replace("[DATA]", args.dataset)
            .replace("[TYPE]", args.data_type)
            .replace("[CLASS]", args.class_num)
            .replace("[T]", args.t)
            .replace("[PLAN]", args.plan)
        )
        data1 = load_json(path1)
        data2 = load_json(path2)
        data = data1 + data2

    elif args.results_file_num == 1:
        path = (
            args.in_path
            .replace("[DATA]", args.dataset)
            .replace("[TYPE]", args.data_type)
            .replace("[CLASS]", args.class_num)
            .replace("[T]", args.t)
            .replace("[PLAN]", args.plan)
        )
        data = load_json(path)
    else:
        raise ValueError("results_file_num must be 1 or 2")

    report = evaluate(
        data=data,
        class_num=args.class_num,
    )

    print_report(report)

    if args.out_path:
        out_path = (
            args.out_path
            .replace("[DATA]", args.dataset)
            .replace("[TYPE]", args.data_type)
            .replace("[CLASS]", args.class_num)
            .replace("[T]", args.t)
            .replace("[PLAN]", args.plan)
        )
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HOVER")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--class_num", type=str, default="2")
    parser.add_argument("--t", type=str, default="")

    parser.add_argument(
        "--in_path1",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_aggregate_[T]0_2000.json",
        help="First input file"
    )
    parser.add_argument(
        "--in_path2",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_aggregate_[T]2000_4000.json",
        help="Second input file"
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_aggregate_[T].json",
        help="Input file when results_file_num=1"
    )
    parser.add_argument(
        "--results_file_num",
        type=int,
        choices=[1, 2],
        default=1,
        help="Number of result files to read (1 or 2)"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_eval_by_hop[T].json",
        help="Output report file"
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="qc",
        help="Which plan version to use for path templates"
    )

    args = parser.parse_args()
    main(args)
