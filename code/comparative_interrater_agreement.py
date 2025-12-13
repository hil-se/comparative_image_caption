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
    if x in (1, -1):
        return int(x)
    return None


def make_gt_choice_from_scores(df, col_a="ImageA_CaptionA_GT", col_b="ImageB_CaptionB_GT", out_col="GT"):
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


def acc_kappa(pairs):
    n = len(pairs["agree"])
    if n == 0:
        return float("nan"), float("nan")
    acc = np.sum(pairs["agree"]) / n
    count_A = Counter(pairs["A"])
    count_B = Counter(pairs["B"])
    pe = (count_A[1] * count_B[1] + count_A[-1] * count_B[-1]) / (n ** 2)
    if pe == 1:
        kappa = 1.0 if acc == 1 else 0.0
    else:
        kappa = 1 - (1 - acc) / (1 - pe)
    return acc, kappa


def run_task(xlsx_path, sheet_name, out_csv_path, gt_a_col, gt_b_col):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df = make_gt_choice_from_scores(df, col_a=gt_a_col, col_b=gt_b_col, out_col="GT")

    raters = ["GT", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    results = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            r1, r2 = raters[i], raters[j]
            pairs = generate_pairs_from_choices(df[r1], df[r2])
            acc, kappa = acc_kappa(pairs)
            results.append({
                "Pair": f"{r1}/{r2}",
                "Acc": f"{acc:.2f}",
                "Kappa": f"{kappa:.2f}",
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_csv_path, index=False)
    print("\n", sheet_name)
    print(result_df)


if __name__ == "__main__":
    xlsx_path = "../results/Image-caption_GT_ VS_Human_Rating.xlsx"
    print(pd.ExcelFile(xlsx_path).sheet_names)


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
