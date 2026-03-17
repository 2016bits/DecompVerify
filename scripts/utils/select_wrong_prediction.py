import json

plan = "plan2.1"

in_path = "./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_final_.json"
in_path = in_path.replace("[DATA]", "HOVER_subset").replace("[PLAN]", plan).replace("[TYPE]", "dev").replace("[CLASS]", "2")

with open(in_path, 'r') as f:
    dataset = json.load(f)

wrong_predictions = [item for item in dataset if item['label'] != item['predicted_label']]

out_path = "./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_wrong_prediction.json"
out_path = out_path.replace("[DATA]", "HOVER_subset").replace("[PLAN]", plan).replace("[TYPE]", "dev").replace("[CLASS]", "2")

with open(out_path, 'w') as f:
    json.dump(wrong_predictions, f, indent=4)
print(f"Saved to {out_path}, total {len(wrong_predictions)} items.")