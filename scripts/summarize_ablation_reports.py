import argparse
import csv
import json
import os


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


def write_json(path, rows):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(rows, file_obj, indent=4, ensure_ascii=False)


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


def build_markdown(rows):
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


def write_text(path, text):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write(text)


def main(args):
    rows = []
    for report_arg in args.report:
        name, path = parse_report_arg(report_arg)
        rows.append(extract_metrics(name, load_report(path)))

    rows = add_deltas(rows, args.baseline_name)

    if args.out_json:
        write_json(args.out_json, rows)
        print(f"Saved JSON summary to {args.out_json}")
    if args.out_csv:
        write_csv(args.out_csv, rows)
        print(f"Saved CSV summary to {args.out_csv}")
    markdown = build_markdown(rows)
    if args.out_md:
        write_text(args.out_md, markdown)
        print(f"Saved Markdown summary to {args.out_md}")

    print(markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True, help="Report formatted as name=path")
    parser.add_argument("--baseline_name", type=str, default="full")
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_md", type=str, default="")
    main(parser.parse_args())
