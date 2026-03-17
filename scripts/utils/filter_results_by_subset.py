import json
import argparse
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(args):
    subset = load_json(args.subset_path)
    results = load_json(args.results_path1) + load_json(args.results_path2)

    subset_ids = {x["id"] for x in subset}
    filtered = [x for x in results if x.get("id") in subset_ids]
    filtered = sorted(filtered, key=lambda x: x["id"])

    print(f"Subset size: {len(subset_ids)}")
    print(f"Full results size: {len(results)}")
    print(f"Filtered results size: {len(filtered)}")

    missing_ids = subset_ids - {x["id"] for x in filtered}
    if missing_ids:
        print(f"Warning: {len(missing_ids)} subset ids not found in results.")
        preview = list(sorted(missing_ids))[:20]
        print(f"Missing id examples: {preview}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"Saved filtered results to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_path", type=str, default="./data/HOVER_subset/converted_data/dev.json")
    parser.add_argument("--results_path1", type=str, default="data/HOVER/plan2.1/dev_2_final_0_2000.json")
    parser.add_argument("--results_path2", type=str, default="data/HOVER/plan2.1/dev_2_final_2000_4000.json")
    parser.add_argument("--output_path", type=str, default="./data/HOVER_subset/plan2.1/dev_final_results.json")
    args = parser.parse_args()
    main(args)