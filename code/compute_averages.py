import pandas as pd

def summarize_agreement(csv_path, task_name):
    df = pd.read_csv(csv_path)

  
    df.columns = [c.strip().lower() for c in df.columns]

    df[["rater_i", "rater_j"]] = df["pair"].str.split("/", expand=True)

  
    df["rater_i"] = df["rater_i"].replace("Ground_Truth", "GT")
    df["rater_j"] = df["rater_j"].replace("Ground_Truth", "GT")

    # against GT
    gt_df = df[
        (df["rater_i"] == "GT") | (df["rater_j"] == "GT")
    ]

    # between raters
    rr_df = df[
        (df["rater_i"] != "GT") & (df["rater_j"] != "GT")
    ]

    gt_po = gt_df["acc"].mean()
    gt_kappa = gt_df["kappa"].mean()

    rr_po = rr_df["acc"].mean()
    rr_kappa = rr_df["kappa"].mean()

    return [
        task_name,
        round(gt_po, 2),
        round(gt_kappa, 2),
        round(rr_po, 2),
        round(rr_kappa, 2),
    ]


results = []

results.append(
    summarize_agreement(
        "../results/human_subject/regression_comparative_agreement.csv",
        "Task 1"
    )
)

results.append(
    summarize_agreement(
        "../results/human_subject/comparative_task2_agreement.csv",
        "Task 2"
    )
)

results.append(
    summarize_agreement(
        "../results/human_subject/sameimage_task3_agreement.csv",
        "Task 3"
    )
)

table12 = pd.DataFrame(
    results,
    columns=[
        "Task",
        "Against_GT_p_o",
        "Against_GT_kappa",
        "Between_Raters_p_o",
        "Between_Raters_kappa",
    ]
)

print(table12)
