import numpy as np
import pandas as pd
from collections import Counter


def to_dir(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip().upper()
        if x == "A":
            return 1
        if x == "B":
            return -1
    return None



def make_gt_choice_from_scores(df, col_a, col_b, out_col="GT"):
    gt = []
    for a, b in zip(df[col_a], df[col_b]):
        if pd.isna(a) or pd.isna(b) or a == b:
            gt.append(np.nan)
        elif a > b:
            gt.append("A")
        else:
            gt.append("B")
    df[out_col] = gt
    return df



def generate_pairs_from_choices(x1, x2):
    pairs = {"A": [], "B": [], "agree": []}
    for a, b in zip(x1, x2):
        d1 = to_dir(a)
        d2 = to_dir(b)
        if d1 is None or d2 is None:
            continue
        pairs["A"].append(d1)
        pairs["B"].append(d2)
        pairs["agree"].append(d1 == d2)
    return pairs


# Accuracy + Cohen’s kappa

def acc_kappa(pairs):
    n = len(pairs["agree"])
    if n == 0:
        return np.nan, np.nan, np.nan

    po = np.mean(pairs["agree"])

    count_A = Counter(pairs["A"])
    count_B = Counter(pairs["B"])

    pe = (count_A[1] * count_B[1] + count_A[-1] * count_B[-1]) / (n ** 2)

    if pe == 1:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)

    return po, pe, kappa



# Majority vote
def majority_vote(row, raters):
    votes = [v for v in row[raters] if isinstance(v, str)]
    if len(votes) == 0:
        return np.nan
    return Counter(votes).most_common(1)[0][0]


def run_task(xlsx_path, sheet_name, out_csv_path, gt_a_col, gt_b_col):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Ground truth decision
    df = make_gt_choice_from_scores(df, gt_a_col, gt_b_col, out_col="GT")

    raters = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]


    # Majority Agreement with GT
    df["Majority"] = df.apply(lambda r: majority_vote(r, raters), axis=1)
    valid = df[["GT", "Majority"]].dropna()

    maj_po = np.mean(valid["GT"] == valid["Majority"])

    pA = np.mean(valid["GT"] == "A")
    pB = np.mean(valid["GT"] == "B")
    maj_pe = pA ** 2 + pB ** 2

    if maj_pe == 1:
        maj_kappa = 1.0
    else:
        maj_kappa = (maj_po - maj_pe) / (1 - maj_pe)

 
    # Inter-rater agreement
    all_raters = ["GT"] + raters
    pair_rows = []

    po_vals, pe_vals, kappa_vals = [], [], []

    for i in range(len(all_raters)):
        for j in range(i + 1, len(all_raters)):
            r1, r2 = all_raters[i], all_raters[j]
            pairs = generate_pairs_from_choices(df[r1], df[r2])
            po, pe, kappa = acc_kappa(pairs)

            pair_rows.append({
                "Pair": f"{r1}/{r2}",
                "Acc": f"{po:.2f}",
                "Kappa": f"{kappa:.2f}",
            })

            # Inter-rater only (exclude GT)
            if r1 != "GT" and r2 != "GT":
                po_vals.append(po)
                pe_vals.append(pe)
                kappa_vals.append(kappa)

    # Save pairwise CSV
    pd.DataFrame(pair_rows).to_csv(out_csv_path, index=False)

   
    # Print results
    print(f"\n{sheet_name}")

    print("Task: Majority Agreement with Ground Truth")
    print(f"Observed agreement (Po): {maj_po:.2f}")
    print(f"Expected agreement (Pe): {maj_pe:.2f}")
    print(f"Cohen’s κ: {maj_kappa:.2f}")

    print("\nTask: Inter-Rater Agreement")
    print(f"Observed agreement (Po): {np.mean(po_vals):.2f}")
    print(f"Expected agreement (Pe): {np.mean(pe_vals):.2f}")
    print(f"Cohen’s κ: {np.mean(kappa_vals):.2f}")


if __name__ == "__main__":
    xlsx_path = "../results/Image-caption_GT_ VS_Human_Rating.xlsx"

    run_task(
        xlsx_path=xlsx_path,
        sheet_name="Comparative - Task 2",
        out_csv_path="../results/human_subject/comparative_task2_agreement.csv",
        gt_a_col="ImageA_CaptionA_GT",
        gt_b_col="ImageB_CaptionB_GT",
    )

    run_task(
        xlsx_path=xlsx_path,
        sheet_name="Same_Image -  Task 3 ",
        out_csv_path="../results/human_subject/sameimage_task3_agreement.csv",
        gt_a_col="ImageA_CaptionA_GT",
        gt_b_col="ImageA_CaptionB_GT",
    )
