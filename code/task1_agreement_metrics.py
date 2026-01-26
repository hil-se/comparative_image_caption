import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Read data
df = pd.read_excel(
    "../results/Image-caption_GT_ VS_Human_Rating.xlsx",
    sheet_name="Regression - Task 1"
)

raters = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

# Majority agreement with GT

df["AVG_P_Rating"] = df[raters].mean(axis=1)

valid = df[["Ground_Truth", "AVG_P_Rating"]].dropna()

mae = np.mean(np.abs(valid["Ground_Truth"] - valid["AVG_P_Rating"]))
pearson_gt = pearsonr(valid["Ground_Truth"], valid["AVG_P_Rating"])[0]
spearman_gt = spearmanr(valid["Ground_Truth"], valid["AVG_P_Rating"])[0]

print("Task 1: Majority Agreement with Ground Truth")
print("MAE:", round(mae, 3))
print("Pearson:", round(pearson_gt, 3))
print("Spearman:", round(spearman_gt, 3))
print()


# Inter-rater reliability 

ratings = df[raters].dropna()

pearsons = []
spearmans = []

for i in range(len(raters)):
    for j in range(i + 1, len(raters)):
        pearsons.append(pearsonr(ratings[raters[i]], ratings[raters[j]])[0])
        spearmans.append(spearmanr(ratings[raters[i]], ratings[raters[j]])[0])

print("Task 1: Inter-Rater Reliability")
print("Inter-rater Pearson:", round(np.mean(pearsons), 3))
print("Inter-rater Spearman:", round(np.mean(spearmans), 3))
