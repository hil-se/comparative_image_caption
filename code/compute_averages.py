import pandas as pd
import os

df = pd.read_csv("../data/table10_inter-rater_agreement.csv")

# Normalize rater labels
df["rater_i"] = df["rater_i"].str.strip()
df["rater_j"] = df["rater_j"].str.strip()

results = []

for task in df["task"].unique():
    task_df = df[df["task"] == task]

    # Against GT
    gt_df = task_df[
        (task_df["rater_i"] == "GT") | (task_df["rater_j"] == "GT")
    ]
    gt_po = gt_df["acc"].mean()
    gt_kappa = gt_df["kappa"].mean()

    # Between raters (exclude GT)
    rr_df = task_df[
        (task_df["rater_i"] != "GT") & (task_df["rater_j"] != "GT")
    ]
    rr_po = rr_df["acc"].mean()
    rr_kappa = rr_df["kappa"].mean()

    results.append([
        task,
        round(gt_po, 2),
        round(gt_kappa, 2),
        round(rr_po, 2),
        round(rr_kappa, 2)
    ])

# Output
summary = pd.DataFrame(
    results,
    columns=["Task", "Against_GT_p_o", "Against_GT_kappa",
             "Between_Raters_p_o", "Between_Raters_kappa"]
)

print(summary)