import json
import pandas as pd
from pathlib import Path

DATASET = "hotpotqa"

conf_path = Path(f"outputs/{DATASET}_confidence_scores.json")
label_path = Path(f"outputs/{DATASET}_correctness_labels.json")

with open(conf_path, "r", encoding="utf-8") as f:
    conf_data = json.load(f)

with open(label_path, "r", encoding="utf-8") as f:
    label_data = json.load(f)

conf_rows = []
for sample in conf_data:
    sample_id = sample["sample_id"]
    question = sample.get("question", "")

    for step in sample["steps"]:
        conf_rows.append({
            "sample_id": sample_id,
            "question": question,
            "step_id": step.get("step_id"),
            "step_text": step.get("content") or step.get("step_text"),
            "confidence_logprob": step.get("confidence_logprob")
        })

label_rows = []
for sample in label_data:
    sample_id = sample["sample_id"]

    for step in sample["per_step"]:
        label_rows.append({
            "sample_id": sample_id,
            "step_id": step.get("step_id"),
            "correctness_label": step.get("label"),
            "ambiguous": step.get("ambiguous"),
            "entailment_prob": step.get("entailment_prob"),
            "neutral_prob": step.get("neutral_prob"),
            "contradiction_prob": step.get("contradiction_prob"),
            "best_evidence": step.get("best_evidence"),
        })

conf_df = pd.DataFrame(conf_rows)
label_df = pd.DataFrame(label_rows)

print("Flattened confidence columns:", conf_df.columns.tolist())
print("Flattened label columns:", label_df.columns.tolist())

merged = conf_df.merge(label_df, on=["sample_id", "step_id"], how="inner")

out_dir = Path("outputs/analysis")
out_dir.mkdir(parents=True, exist_ok=True)



out_path = out_dir / f"{DATASET}_merged.csv"
merged.to_csv(out_path, index=False)

print("Confidence rows:", len(conf_df))
print("Label rows:", len(label_df))
print("Merged rows:", len(merged))
print("Saved:", out_path)
print(merged.head())