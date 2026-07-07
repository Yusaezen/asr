import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

DATASET = "hotpotqa"

path = Path(f"outputs/analysis/{DATASET}_merged.csv")

df = pd.read_csv(path)

print("=" * 60)
print("Calibration Analysis")
print("=" * 60)

print("\nTotal rows:", len(df))
print("\nCorrectness labels:")
print(df["correctness_label"].value_counts())

# Remove ambiguous rows

df = df[df["correctness_label"].isin(["correct", "incorrect"])].copy()

df["is_correct"] = df["correctness_label"].map({
    "correct": 1,
    "incorrect": 0
})

df["confidence_logprob"] = pd.to_numeric(
    df["confidence_logprob"],
    errors="coerce"
)

df = df.dropna(subset=["confidence_logprob"])

print("\nUsable rows:", len(df))

# Confidence statistics

print("\nAverage confidence:")
print(df.groupby("correctness_label")["confidence_logprob"].mean())

stats = df.groupby("correctness_label")["confidence_logprob"].describe()
stats.to_csv(f"outputs/analysis/{DATASET}_confidence_statistics.csv")


# Correlation

corr = df["confidence_logprob"].corr(df["is_correct"])

print("\nPearson Correlation:", corr)


# Check confidence range

print("\nConfidence Range")
print("Minimum:", df["confidence_logprob"].min())
print("Maximum:", df["confidence_logprob"].max())


# Expected Calibration Error

def compute_ece(df, confidence_col, label_col="is_correct", n_bins=10):

    bins = np.linspace(0, 1, n_bins + 1)

    ece = 0

    for i in range(n_bins):

        low = bins[i]
        high = bins[i + 1]

        if i == n_bins - 1:
            bin_df = df[
                (df[confidence_col] >= low)
                &
                (df[confidence_col] <= high)
            ]
        else:
            bin_df = df[
                (df[confidence_col] >= low)
                &
                (df[confidence_col] < high)
            ]

        if len(bin_df) == 0:
            continue

        avg_conf = bin_df[confidence_col].mean()
        accuracy = bin_df[label_col].mean()

        weight = len(bin_df) / len(df)

        ece += weight * abs(avg_conf - accuracy)

    return ece


ece = compute_ece(df, "confidence_logprob")

print("\nExpected Calibration Error:", ece)

# Threshold analysis

threshold_rows = []

for theta in np.arange(0.1, 1.0, 0.1):

    flagged = df[df["confidence_logprob"] < theta]

    if len(flagged) == 0:
        continue

    wrong_rate = 1 - flagged["is_correct"].mean()

    threshold_rows.append({
        "threshold": round(theta, 2),
        "flagged_steps": len(flagged),
        "wrong_rate_among_flagged": wrong_rate
    })

threshold_df = pd.DataFrame(threshold_rows)

threshold_path = Path(
    f"outputs/analysis/{DATASET}_threshold_analysis.csv"
)

threshold_df.to_csv(threshold_path, index=False)

print("\nThreshold Analysis")
print(threshold_df)

# Box Plot

plt.figure(figsize=(6,5))

df.boxplot(
    column="confidence_logprob",
    by="correctness_label"
)

plt.title("Confidence vs Correctness")
plt.suptitle("")
plt.ylabel("Confidence")

plt.savefig(
    f"outputs/analysis/{DATASET}_boxplot.png",
    dpi=300
)

plt.close()


plt.figure(figsize=(7,5))

for label in df["correctness_label"].unique():

    df[
        df["correctness_label"] == label
    ]["confidence_logprob"].hist(
        alpha=0.5,
        label=label
    )

plt.legend()
plt.xlabel("Confidence")
plt.ylabel("Frequency")

plt.savefig(
    f"outputs/analysis/{DATASET}_histogram.png",
    dpi=300
)

plt.close()


manual = df.sample(
    min(30, len(df)),
    random_state=42
).copy()

manual["human_label"] = ""
manual["notes"] = ""

manual.to_csv(
    f"outputs/analysis/{DATASET}_manual_review.csv",
    index=False
)


# Summary JSON

summary = {
    "total_rows_after_filtering": int(len(df)),
    "pearson_correlation": float(corr),
    "expected_calibration_error": float(ece),
    "average_confidence": df.groupby(
        "correctness_label"
    )["confidence_logprob"].mean().to_dict()
}

summary_path = Path(
    f"outputs/analysis/{DATASET}_calibration_summary.json"
)

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)

print("\nSummary saved:", summary_path)

print("\nAll outputs saved in outputs/analysis/")