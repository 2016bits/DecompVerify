import json
import argparse
import os
from tqdm import tqdm

from get_answer import process_data_item


def need_repair(item):
    if item.get("answer_used_fallback", False):
        return True

    if item.get("answer_issues"):
        return True

    answers = item.get("answer_result", {}).get("answers", [])
    for ans in answers:
        if ans.get("status") == "api_error":
            return True

    return False


def main(args):
    question_path = args.question_path.replace("[DATA]", args.dataset).replace("[PLAN]", args.plan).replace("[TYPE]", args.data_type).replace("[START]", str(args.start)).replace("[END]", str(args.end))
    answer_path = args.answer_path.replace("[DATA]", args.dataset).replace("[PLAN]", args.plan).replace("[TYPE]", args.data_type).replace("[START]", str(args.start)).replace("[END]", str(args.end))
    out_path = args.out_path.replace("[DATA]", args.dataset).replace("[PLAN]", args.plan).replace("[TYPE]", args.data_type).replace("[START]", str(args.start)).replace("[END]", str(args.end))
    with open(question_path, "r", encoding="utf-8") as f:
        question_data = json.load(f)

    with open(answer_path, "r", encoding="utf-8") as f:
        answer_data = json.load(f)

    question_by_id = {x["id"]: x for x in question_data}
    repaired = []
    repair_count = 0

    for old_item in tqdm(answer_data):
        item_id = old_item["id"]

        if not need_repair(old_item):
            repaired.append(old_item)
            continue

        if item_id not in question_by_id:
            print(f"[Skip] missing question item: {item_id}")
            repaired.append(old_item)
            continue

        try:
            new_item = process_data_item(
                question_by_id[item_id],
                plan=args.plan,
                port=args.port,
            )
            repaired.append(new_item)
            repair_count += 1
        except Exception as exc:
            print(f"[Keep old] repair failed for {item_id}: {exc}")
            repaired.append(old_item)

    repaired = sorted(repaired, key=lambda x: x["id"])

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(repaired, f, indent=4, ensure_ascii=False)

    print(f"Repair count: {repair_count}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_2_questions_[START]_[END].json")
    parser.add_argument("--answer_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_2_answers_[START]_[END].json")
    parser.add_argument("--out_path", type=str, default="./data/[DATA]/[PLAN]/[TYPE]_2_answers_[START]_[END]_repaired.json")
    parser.add_argument("--plan", type=str, default="plan6.2")
    parser.add_argument("--port", type=str, default="8370")
    parser.add_argument("--dataset", type=str, default="HOVER")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=4000)
    main(parser.parse_args())