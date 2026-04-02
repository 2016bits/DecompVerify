import json
import argparse
import os


def build_path(template, args):
    path = template
    path = path.replace("[DATA]", args.dataset)
    path = path.replace("[PLAN]", args.plan)
    path = path.replace("[TYPE]", args.data_type)
    path = path.replace("[CLASS]", args.class_num)
    path = path.replace("[T]", args.t)
    path = path.replace("[S]", str(args.start))
    path = path.replace("[E]", str(args.end))
    return path


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
        "nei", "insufficient", "unknown",
    }:
        return "refutes" if class_num == "2" else "not enough information"

    return "refutes" if class_num == "2" else label


def get_predicted_label(item, class_num="2"):
    if "aggregation_result" in item:
        pred = item["aggregation_result"].get("final_label")
    else:
        pred = item.get("predicted_label_3way") if class_num == "3" else item.get("predicted_label")
        if pred is None:
            pred = item.get("predicted_label")
    return normalize_label(pred, class_num=class_num)


def main(args):
    in_path = build_path(args.in_path, args)
    out_path = build_path(args.out_path, args)

    print(f"Loading from: {in_path}")

    with open(in_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError(f"Expected a list in {in_path}, but got {type(dataset).__name__}.")

    wrong_predictions = []
    for item in dataset:
        gold_label = normalize_label(item.get('label'), class_num=args.class_num)
        predicted_label = get_predicted_label(item, class_num=args.class_num)

        if gold_label != predicted_label:
            wrong_item = dict(item)
            wrong_item["gold_label_normalized"] = gold_label
            wrong_item["predicted_label_normalized"] = predicted_label
            wrong_predictions.append(wrong_item)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(wrong_predictions, f, indent=4, ensure_ascii=False)

    print(f"Saved to {out_path}, total {len(wrong_predictions)} wrong items out of {len(dataset)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='HOVER_subset', help='Dataset name')
    parser.add_argument('--data_type', type=str, default='dev', help='Data type: train/dev/test')
    parser.add_argument('--class_num', type=str, default='2', help='Number of classes: 2/3')
    parser.add_argument('--start', type=int, default=0, help='Optional placeholder value for [S]')
    parser.add_argument('--end', type=int, default=200, help='Optional placeholder value for [E]')

    parser.add_argument(
        '--in_path',
        type=str,
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_aggregate_[T].json',
        help='Input path template'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_wrong_prediction_[T].json',
        help='Output path template'
    )

    parser.add_argument('--plan', type=str, default='azure', help='Which plan version to use')
    parser.add_argument('--t', type=str, default='')

    args = parser.parse_args()
    main(args)
