import argparse
import csv
import json
import os
from collections import Counter


def load_report(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def parse_report_arg(value):
    if "=" not in value:
        raise ValueError(f"Report must be formatted as name=path: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Report must be formatted as name=path: {value}")
    return name, path


def parse_aggregate_arg(value):
    if "=" not in value:
        raise ValueError(f"Aggregate must be formatted as name=path: {value}")
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def extract_metrics(name, report):
    per_label = report.get("per_label_metrics", {}) or {}
    supports = per_label.get("supports", {}) or {}
    refutes = per_label.get("refutes", {}) or {}
    return {
        "name": name,
        "total_samples": report.get("total_samples", 0),
        "accuracy": report.get("overall_accuracy", 0.0),
        "macro_precision": report.get("macro_precision", 0.0),
        "macro_recall": report.get("macro_recall", 0.0),
        "macro_f1": report.get("macro_f1", 0.0),
        "supports_precision": supports.get("precision", 0.0),
        "supports_recall": supports.get("recall", 0.0),
        "supports_f1": supports.get("f1", 0.0),
        "refutes_precision": refutes.get("precision", 0.0),
        "refutes_recall": refutes.get("recall", 0.0),
        "refutes_f1": refutes.get("f1", 0.0),
        "gold_distribution": report.get("overall_gold_distribution", {}),
        "pred_distribution": report.get("overall_pred_distribution", {}),
        "by_hop_metrics": report.get("by_hop_metrics", {}) or {},
    }


def add_deltas(rows, baseline_name):
    baseline = None
    for row in rows:
        if row["name"] == baseline_name:
            baseline = row
            break
    if baseline is None:
        raise ValueError(f"Baseline report not found: {baseline_name}")

    for row in rows:
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "supports_f1", "refutes_f1"):
            row[f"delta_{key}"] = round(row[key] - baseline[key], 4)
    return rows


def write_json(path, payload):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=4, ensure_ascii=False)


def write_csv(path, rows):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "name",
        "total_samples",
        "accuracy",
        "delta_accuracy",
        "macro_f1",
        "delta_macro_f1",
        "macro_precision",
        "delta_macro_precision",
        "macro_recall",
        "delta_macro_recall",
        "supports_f1",
        "delta_supports_f1",
        "refutes_f1",
        "delta_refutes_f1",
        "pred_distribution",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row[key], ensure_ascii=False) if isinstance(row.get(key), dict) else row.get(key)
                for key in fieldnames
            })


def format_float(value):
    return f"{float(value):.4f}"


def build_overall_markdown(rows):
    headers = [
        "variant",
        "acc",
        "d_acc",
        "macro_f1",
        "d_macro_f1",
        "supports_f1",
        "refutes_f1",
        "pred_dist",
    ]
    lines = [
        "## Overall",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                row["name"],
                format_float(row["accuracy"]),
                format_float(row["delta_accuracy"]),
                format_float(row["macro_f1"]),
                format_float(row["delta_macro_f1"]),
                format_float(row["supports_f1"]),
                format_float(row["refutes_f1"]),
                json.dumps(row["pred_distribution"], ensure_ascii=False),
            ])
            + " |"
        )
    return "\n".join(lines) + "\n"


def _collect_hops(rows):
    hops = set()
    for row in rows:
        hops.update((row.get("by_hop_metrics") or {}).keys())
    def _key(h):
        try:
            return (0, int(h))
        except (TypeError, ValueError):
            return (1, str(h))
    return sorted(hops, key=_key)


def build_hopwise_markdown(rows, metric, baseline_name):
    hops = _collect_hops(rows)
    if not hops:
        return ""
    baseline = next((row for row in rows if row["name"] == baseline_name), None)

    title_metric = {"macro_f1": "Macro-F1", "accuracy": "Accuracy"}.get(metric, metric)
    headers = ["variant"] + [f"hop{h}" for h in hops] + [f"Δhop{h}" for h in hops]
    lines = [
        f"## Per-hop {title_metric}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        hop_data = row.get("by_hop_metrics") or {}
        cells = [row["name"]]
        for h in hops:
            entry = hop_data.get(h, {})
            cells.append(format_float(entry.get(metric, 0.0)) if entry else "-")
        for h in hops:
            entry = hop_data.get(h, {})
            base_entry = (baseline or {}).get("by_hop_metrics", {}).get(h, {}) if baseline else {}
            if entry and base_entry:
                diff = entry.get(metric, 0.0) - base_entry.get(metric, 0.0)
                cells.append(f"{diff:+.4f}")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_pred_distribution_by_hop_markdown(rows):
    hops = _collect_hops(rows)
    if not hops:
        return ""
    headers = ["variant"] + [f"hop{h} pred (sup/ref)" for h in hops]
    lines = [
        "## Per-hop prediction distribution",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        hop_data = row.get("by_hop_metrics") or {}
        cells = [row["name"]]
        for h in hops:
            entry = hop_data.get(h, {})
            pd = entry.get("pred_distribution", {}) if entry else {}
            sup = pd.get("supports", 0)
            ref = pd.get("refutes", 0)
            cells.append(f"{sup}/{ref}" if entry else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Flip / disagreement analysis vs the baseline aggregate file
# ---------------------------------------------------------------------------


def _binary_label(item):
    agg = item.get("aggregation_result", {}) or {}
    label = agg.get("final_label_binary") or agg.get("final_label") or "refutes"
    return "supports" if str(label).strip().lower() == "supports" else "refutes"


def compute_flip_summary(baseline_items, variant_items):
    base_map = {item["id"]: item for item in baseline_items}
    var_map = {item["id"]: item for item in variant_items}
    ids = sorted(set(base_map) & set(var_map))

    flips = Counter()
    correct_flips = 0
    incorrect_flips = 0
    unchanged = 0
    total = 0
    by_hop = {}
    for _id in ids:
        base = base_map[_id]
        var = var_map[_id]
        gold = str(base.get("label") or var.get("label") or "refutes").strip().lower()
        gold = "supports" if gold == "supports" else "refutes"
        bp = _binary_label(base)
        vp = _binary_label(var)
        total += 1
        hop = base.get("num_hops") or var.get("num_hops")
        hop_key = str(hop) if hop is not None else "unknown"
        bucket = by_hop.setdefault(hop_key, {"flipped": 0, "to_correct": 0, "to_incorrect": 0, "total": 0})
        bucket["total"] += 1
        if bp == vp:
            unchanged += 1
            continue
        flips[(bp, vp, gold)] += 1
        bucket["flipped"] += 1
        if vp == gold and bp != gold:
            correct_flips += 1
            bucket["to_correct"] += 1
        elif vp != gold and bp == gold:
            incorrect_flips += 1
            bucket["to_incorrect"] += 1
    return {
        "total": total,
        "unchanged": unchanged,
        "flipped": total - unchanged,
        "flipped_to_correct": correct_flips,
        "flipped_to_incorrect": incorrect_flips,
        "net_correct": correct_flips - incorrect_flips,
        "flip_breakdown": [
            {
                "from_full": k[0], "to_variant": k[1], "gold": k[2], "count": v,
                "direction": "correct" if k[1] == k[2] else "incorrect",
            }
            for k, v in flips.most_common()
        ],
        "by_hop": by_hop,
    }


def build_flip_markdown(flip_data):
    if not flip_data:
        return ""
    lines = [
        "## Flips vs baseline aggregate",
        "",
        "| variant | flipped | →correct | →incorrect | net | unchanged |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, info in flip_data.items():
        lines.append(
            f"| {name} | {info['flipped']} | {info['flipped_to_correct']} | "
            f"{info['flipped_to_incorrect']} | {info['net_correct']:+d} | {info['unchanged']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_text(path, text):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write(text)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(args):
    rows = []
    for report_arg in args.report:
        name, path = parse_report_arg(report_arg)
        rows.append(extract_metrics(name, load_report(path)))
    rows = add_deltas(rows, args.baseline_name)

    flip_payload = {}
    if args.aggregate:
        baseline_path = None
        variant_paths = {}
        for arg in args.aggregate:
            name, path = parse_aggregate_arg(arg)
            if name == args.baseline_name:
                baseline_path = path
            else:
                variant_paths[name] = path
        if baseline_path is None:
            print("[warn] --aggregate is set but no baseline aggregate provided; skipping flip analysis.")
        else:
            baseline_items = load_report(baseline_path)
            for name, path in variant_paths.items():
                variant_items = load_report(path)
                flip_payload[name] = compute_flip_summary(baseline_items, variant_items)

    if args.out_json:
        json_payload = {"summary": rows, "flips": flip_payload}
        write_json(args.out_json, json_payload)
        print(f"Saved JSON summary to {args.out_json}")
    if args.out_csv:
        write_csv(args.out_csv, rows)
        print(f"Saved CSV summary to {args.out_csv}")

    sections = [
        build_overall_markdown(rows),
        build_hopwise_markdown(rows, "macro_f1", args.baseline_name),
        build_hopwise_markdown(rows, "accuracy", args.baseline_name),
        build_pred_distribution_by_hop_markdown(rows),
        build_flip_markdown(flip_payload) if flip_payload else "",
    ]
    markdown = "\n".join(section for section in sections if section)

    if args.out_md:
        write_text(args.out_md, markdown)
        print(f"Saved Markdown summary to {args.out_md}")

    print(markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True, help="Eval report formatted as name=path")
    parser.add_argument("--aggregate", action="append", default=[], help="Aggregate file formatted as name=path (used for flip analysis vs --baseline_name).")
    parser.add_argument("--baseline_name", type=str, default="full")
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_md", type=str, default="")
    main(parser.parse_args())
