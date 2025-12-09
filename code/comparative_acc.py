import numpy as np
import pandas as pd



def generate_pairs(gt):
    n = len(gt)
    pairs = {}
    for i in range(n):
        for j in range(i+1, n):
            if gt[i]>gt[j]:
                pairs[(i, j)] = 1
            elif gt[i]<gt[j]:
                pairs[(i, j)] = -1
    return pairs

def comparative_accuracy(x, pairs):
    tp = 0
    p = 0
    for tup in pairs:
        if x[tup[0]]>x[tup[1]]:
            if pairs[tup] == 1:
                tp += 1
                p += 1
            else:
                p += 1
        elif x[tup[0]]<x[tup[1]]:
            if pairs[tup] == -1:
                tp += 1
                p += 1
            else:
                p += 1
    return float(tp) / p


df = pd.read_csv("../data/Image-caption-rating.csv")
raters = ["P1","P2","P3","P4","P5","P6","P7","P8"]

pairs = generate_pairs(df["Ground_Truth"])
accs = {}
for rater in raters:
    accs[rater] = comparative_accuracy(df[rater], pairs)
result = pd.DataFrame(accs,index=[1])
result.to_csv("../results/human_subject/regression_comparative_accuracy.csv", index=False)
print(accs)
