# Controlled image--caption results

## Integrity

The final package contains four integer-budget files (`N=1` through `5`) and
two ViLBERT fractional-budget files (`N=0.1` through `0.9`), totaling 880
result rows. Condition keys are unique, the planned seeds are complete, and
the result hashes match their metadata. Fractional metadata is marked as
reconstructed because the original host and Slurm identifiers were not
retained.

## Regression representation comparison

| Dataset | Representation | Pearson | Spearman | Kendall tau-c |
|---|---|---:|---:|---:|
| Flickr-Expert | ViLBERT | 0.7657 +/- 0.0090 | 0.6624 +/- 0.0172 | 0.5366 +/- 0.0144 |
| Flickr-Expert | CLIP ViT-B/32 | 0.6583 +/- 0.0124 | 0.5772 +/- 0.0175 | 0.4601 +/- 0.0147 |
| VICR | ViLBERT | 0.9035 +/- 0.0024 | 0.8844 +/- 0.0025 | 0.8059 +/- 0.0039 |
| VICR | CLIP ViT-B/32 | 0.8372 +/- 0.0045 | 0.8342 +/- 0.0036 | 0.7389 +/- 0.0044 |

ViLBERT is the stronger representation in this controlled comparison. CLIP
is retained as a negative regression baseline, not used as the primary
pairwise representation in the paper.

## Hinge versus Bradley--Terry at N=1

| Dataset | Objective | Pearson | Spearman | Kendall tau-c |
|---|---|---:|---:|---:|
| Flickr-Expert | Hinge | 0.7278 | 0.6626 | 0.5364 |
| Flickr-Expert | Bradley--Terry | 0.7333 | 0.6684 | 0.5414 |
| VICR | Hinge | 0.8652 | 0.8783 | 0.7969 |
| VICR | Bradley--Terry | 0.8757 | 0.8786 | 0.7975 |

Bradley--Terry is numerically higher in all six cells. None of the paired
two-sided Wilcoxon tests is significant after Holm correction (adjusted
`p = 0.2441` to `1.0000`; VICR Pearson is closest at `p = 0.0586`). The result
is therefore a numerical difference, not evidence of a general objective
advantage.

## Pair-budget sensitivity

Most of the low-budget improvement occurs by approximately `N=0.5`. On
Flickr-Expert, mean Spearman rises from 0.6012/0.5977 at `N=0.1` to
0.6577/0.6622 at `N=0.5` for hinge/Bradley--Terry. On VICR, it rises from
0.8558/0.8599 to 0.8746/0.8742. The objectives remain close across the grid.

## Corrected same-image evaluation

| Dataset | Hinge accuracy | Bradley--Terry accuracy | Holm-adjusted p |
|---|---:|---:|---:|
| Flickr-Expert | 0.8385 +/- 0.0102 | 0.8446 +/- 0.0053 | 0.1211 |
| VICR | 0.8387 +/- 0.0063 | 0.8393 +/- 0.0088 | 0.7129 |

The image-disjoint protocol eliminates reversed-pair leakage. Neither
objective difference is statistically significant.
