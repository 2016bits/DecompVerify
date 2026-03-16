import json
import argparse
import random
import os
from collections import defaultdict, Counter


def normalize_label(label):
    s = str(label).strip().lower()
    if s in {"supports", "support", "supported"}:
        return "supports"
    if s in {"refutes", "refute", "refuted", "contradict", "contradicted"}:
        return "refutes"
    return s


def get_group_key(item):
    label = normalize_label(item.get("label", "unknown"))
    hop = item.get("num_hops", "unknown")
    return (label, hop)


def summarize(data, title="Dataset summary"):
    print("=" * 80)
    print(title)
    print(f"Total: {len(data)}")

    label_counter = Counter()
    hop_counter = Counter()
    group_counter = Counter()

    for x in data:
        label = normalize_label(x.get("label", "unknown"))
        hop = x.get("num_hops", "unknown")
        label_counter[label] += 1
        hop_counter[hop] += 1
        group_counter[(label, hop)] += 1

    print("\nLabel distribution:")
    for k, v in sorted(label_counter.items()):
        print(f"  {k}: {v}")

    print("\nHop distribution:")
    for k, v in sorted(hop_counter.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")

    print("\nLabel x Hop distribution:")
    for k, v in sorted(group_counter.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        print(f"  {k}: {v}")
    print("=" * 80)


def stratified_sample(data, sample_size, seed=42, balance_labels=True):
    rng = random.Random(seed)

    groups = defaultdict(list)
    for item in data:
        key = get_group_key(item)
        groups[key].append(item)

    # 每组内部打乱
    for key in groups:
        rng.shuffle(groups[key])

    if not balance_labels:
        # 按 (label, hop) 比例抽样
        total = len(data)
        selected = []

        group_sizes = {}
        for key, items in groups.items():
            ratio = len(items) / total
            group_sizes[key] = int(round(sample_size * ratio))

        # 修正总数误差
        current = sum(group_sizes.values())
        keys = sorted(groups.keys())
        while current < sample_size:
            for k in keys:
                if current >= sample_size:
                    break
                if group_sizes[k] < len(groups[k]):
                    group_sizes[k] += 1
                    current += 1
        while current > sample_size:
            for k in keys[::-1]:
                if current <= sample_size:
                    break
                if group_sizes[k] > 0:
                    group_sizes[k] -= 1
                    current -= 1

        for k, n in group_sizes.items():
            selected.extend(groups[k][:n])

        rng.shuffle(selected)
        return selected

    # label 平衡模式：
    # 先按 label 分总配额，再在每个 label 内按 hop 比例分配
    by_label = defaultdict(list)
    for item in data:
        label = normalize_label(item.get("label", "unknown"))
        by_label[label].append(item)

    labels = sorted(by_label.keys())
    if len(labels) != 2:
        raise ValueError(f"Expected 2 labels for balancing, got {labels}")

    per_label = sample_size // 2
    remainder = sample_size - per_label * 2

    label_targets = {
        labels[0]: per_label,
        labels[1]: per_label
    }
    if remainder > 0:
        # 多出来的 1 条给样本更多的标签
        bigger = max(labels, key=lambda lb: len(by_label[lb]))
        label_targets[bigger] += remainder

    selected = []

    for label in labels:
        label_items = by_label[label]
        label_total = len(label_items)

        # 该 label 内按 hop 分层
        hop_groups = defaultdict(list)
        for item in label_items:
            hop_groups[item.get("num_hops", "unknown")].append(item)

        for hop in hop_groups:
            rng.shuffle(hop_groups[hop])

        hop_sizes = {}
        for hop, items in hop_groups.items():
            ratio = len(items) / label_total
            hop_sizes[hop] = int(round(label_targets[label] * ratio))

        current = sum(hop_sizes.values())
        hops = sorted(hop_groups.keys(), key=lambda x: str(x))

        while current < label_targets[label]:
            for h in hops:
                if current >= label_targets[label]:
                    break
                if hop_sizes[h] < len(hop_groups[h]):
                    hop_sizes[h] += 1
                    current += 1

        while current > label_targets[label]:
            for h in hops[::-1]:
                if current <= label_targets[label]:
                    break
                if hop_sizes[h] > 0:
                    hop_sizes[h] -= 1
                    current -= 1

        for h, n in hop_sizes.items():
            selected.extend(hop_groups[h][:n])

    # 如果由于某些组太小导致数量不够，补抽
    selected_ids = {x["id"] for x in selected}
    if len(selected) < sample_size:
        leftovers = [x for x in data if x["id"] not in selected_ids]
        rng.shuffle(leftovers)
        need = sample_size - len(selected)
        selected.extend(leftovers[:need])

    # 如果超了，裁掉
    if len(selected) > sample_size:
        rng.shuffle(selected)
        selected = selected[:sample_size]

    # 最终按 id 排序，保证稳定
    selected = sorted(selected, key=lambda x: x["id"])
    return selected


def main(args):
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.sample_size > len(data):
        raise ValueError(f"sample_size={args.sample_size} > dataset size={len(data)}")

    summarize(data, title="Original dataset")

    subset = stratified_sample(
        data=data,
        sample_size=args.sample_size,
        seed=args.seed,
        balance_labels=args.balance_labels
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    summarize(subset, title="Sampled subset")
    print(f"Saved subset to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/HOVER/converted_data/dev.json",
        help="Path to original dev json, e.g. ./data/HOVER/converted_data/dev.json"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/HOVER_subset/converted_data/dev.json",
        help="Path to save sampled subset json"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="Number of examples to sample"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--balance_labels",
        action="store_true",
        help="Whether to force supports/refutes balance"
    )

    args = parser.parse_args()
    main(args)