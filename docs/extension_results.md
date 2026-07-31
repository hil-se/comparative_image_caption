# Image-caption extension results

## Status and scope

The complete controlled matrix requested by Dr. Yu finished successfully:

- datasets: VICR and Flickr-Expert;
- representations: released ViLBERT embeddings and CLIP ViT-B/32;
- objectives: regression, hinge pairwise loss, and Bradley-Terry loss;
- pairwise sample sizes: N=1 through N=5;
- pair conditions: different-image and same-image;
- ten seeds per condition.

The four full result files contain 130 rows each. Their SHA-256 hashes match
their metadata, every planned condition is present, and there are no duplicate
condition keys. The files and checkpoints remain on TIGRIS under
`/home/xx4455/paper-projects/artifacts/comparative_image_caption`.

Values below are seed means. Standard deviations are shown where they are most
useful for interpreting the primary comparisons.

## Validation against the paper and released results

### Regression

The released VICR regression results closely match the paper:

| Source | Pearson | Spearman | Kendall |
|---|---:|---:|---:|
| Paper | 0.908 | 0.887 | 0.811 |
| Released `baseline.csv` | 0.9080 | 0.8878 | 0.8119 |
| Controlled official-split run | 0.9035 | 0.8844 | 0.8059 |

The controlled run is slightly lower because it consistently uses the official
partition, training-only feature standardization, and the corrected evaluation
pipeline. It still validates the main regression result.

### Comparative model

The unchanged released comparative program reruns successfully. Its aggregate
output is close to the checked-in `compare.csv`:

| Source, averaged over N=1–5 | Pearson | Spearman | Kendall |
|---|---:|---:|---:|
| Checked-in output | 0.8591 | 0.8780 | 0.7979 |
| Unchanged rerun | 0.8599 | 0.8779 | 0.7979 |

The paper reports 0.874/0.880/0.800. That row appears to be based most closely
on N=1 rather than the mean over all five N settings: the checked-in N=1 means
are 0.8764/0.8826/0.8040. The table should therefore identify which N setting
or aggregation rule produced the reported row.

### Same-image evaluation

The released same-image program is not a valid held-out evaluation. It combines
the official partitions, standardizes using all examples, generates both
orientations of every pair, and randomly divides those oriented pairs. As a
result, 80.45% of its test pairs have the exact reversed pair in training.

The paired protocol ablation gives:

| Protocol | Test accuracy | Reversed-pair leakage |
|---|---:|---:|
| Released random-pair split | 0.9810 +/- 0.0017 | 80.45% |
| Official image-disjoint split | 0.8072 +/- 0.0064 | 0% |

The final controlled ViLBERT hinge run, using the corrected image-disjoint
protocol and ten seeds, obtains 0.8387 +/- 0.0063 same-image accuracy. This is
the result that should be used going forward. The paper's reported 0.848 is
near it, but the released evaluation code does not directly reproduce that
number.

## New experiment 1: CLIP versus ViLBERT

The regression baseline provides the cleanest representation comparison:

| Dataset | Representation | General pair accuracy | Same-image accuracy | Pearson | Spearman | Kendall |
|---|---|---:|---:|---:|---:|---:|
| Flickr-Expert | ViLBERT | 0.8706 +/- 0.0097 | 0.8296 +/- 0.0114 | 0.7657 +/- 0.0090 | 0.6624 | 0.5366 |
| Flickr-Expert | CLIP | 0.8178 +/- 0.0092 | 0.7802 +/- 0.0129 | 0.6583 +/- 0.0124 | 0.5772 | 0.4601 |
| VICR | ViLBERT | 0.9212 +/- 0.0019 | 0.8377 +/- 0.0062 | 0.9035 +/- 0.0024 | 0.8844 | 0.8059 |
| VICR | CLIP | 0.8850 +/- 0.0019 | 0.7960 +/- 0.0104 | 0.8372 +/- 0.0045 | 0.8342 | 0.7389 |

Relative to ViLBERT, CLIP is lower by:

- 5.27 percentage points in general pair accuracy, 4.94 points in same-image
  accuracy, and 0.107 Pearson on Flickr-Expert;
- 3.62 percentage points in general pair accuracy, 4.17 points in same-image
  accuracy, and 0.066 Pearson on VICR.

The tested CLIP representation is therefore not an improvement over the
released ViLBERT representation. This negative result is consistent across
both datasets and all principal metrics.

## New experiment 2: Bradley-Terry versus hinge

For different-image training, the best settings by general pair accuracy are:

| Dataset | Representation | Objective and N | General pair accuracy | Same-image accuracy | Pearson |
|---|---|---|---:|---:|---:|
| Flickr-Expert | ViLBERT | Bradley-Terry, N=2 | 0.8775 | 0.8455 | 0.7342 |
| Flickr-Expert | ViLBERT | Hinge, N=4 | 0.8727 | 0.8442 | 0.7314 |
| Flickr-Expert | CLIP | Bradley-Terry, N=5 | 0.8406 | 0.8058 | 0.6457 |
| Flickr-Expert | CLIP | Hinge, N=4 | 0.8378 | 0.7986 | 0.6388 |
| VICR | ViLBERT | Bradley-Terry, N=4 | 0.9212 | 0.8489 | 0.8796 |
| VICR | ViLBERT | Hinge, N=4 | 0.9206 | 0.8476 | 0.8707 |
| VICR | CLIP | Bradley-Terry, N=5 | 0.8734 | 0.7887 | 0.7812 |
| VICR | CLIP | Hinge, N=3 | 0.8727 | 0.7834 | 0.7714 |

Paired seed comparisons give a more reliable interpretation than selecting
only the best setting:

- On Flickr-Expert with ViLBERT, Bradley-Terry improves general pair accuracy
  at every N by 0.32 to 0.56 percentage points. It wins 40 of the 50 paired
  N-by-seed comparisons. The N=2 through N=5 general-accuracy differences have
  exploratory paired 95% confidence intervals above zero.
- On VICR with ViLBERT, general and same-image accuracy are effectively tied.
  Bradley-Terry raises Pearson by about 0.010 at every N, with 44 wins in 50
  paired comparisons.
- With CLIP, Bradley-Terry is worse at small N and becomes competitive only at
  N=4 or N=5. It is not consistently better than hinge on either dataset.

Bradley-Terry is thus a modest, representation- and data-dependent improvement.
The evidence supports it most clearly for Flickr-Expert with ViLBERT, not as a
general replacement for hinge loss.

## New experiment 3: Flickr-Expert

The full Flickr-Expert experiment succeeds and provides an independent
generalization test. The main conclusions are:

- ViLBERT transfers better than CLIP.
- Bradley-Terry gives small but repeatable gains over hinge with ViLBERT.
- More sampled comparisons generally help CLIP; its best result occurs at
  N=5.
- The best pairwise models improve general and same-image pair accuracy over
  their corresponding regression baselines, although regression retains the
  strongest calibrated rating correlation for ViLBERT.

## Same-image-trained models

| Dataset | Representation | Hinge accuracy | Bradley-Terry accuracy |
|---|---|---:|---:|
| Flickr-Expert | ViLBERT | 0.8385 | 0.8446 |
| Flickr-Expert | CLIP | 0.7884 | 0.7881 |
| VICR | ViLBERT | 0.8387 | 0.8393 |
| VICR | CLIP | 0.7816 | 0.7693 |

Same-image training does not provide a universal Bradley-Terry advantage. The
useful gain is again concentrated in Flickr-Expert with ViLBERT.

The VICR same-image-trained CLIP models specialize strongly: their same-image
accuracy remains around 0.77--0.78, but general pair accuracy and Pearson
correlation fall substantially. This should be described as a transfer failure
between pair conditions, not used as evidence of a data or implementation
error.

## Interpretation and recommendation

1. Keep ViLBERT as the primary representation. Report CLIP ViT-B/32 as a
   controlled negative result rather than an architectural upgrade.
2. Present Bradley-Terry as a conditional refinement. Emphasize its consistent
   Flickr-Expert/ViLBERT gain and its VICR Pearson gain, while stating that pair
   accuracy is otherwise tied or mixed.
3. Include Flickr-Expert as the strongest new generalization experiment.
4. Use only the corrected official image-disjoint same-image protocol.
5. Do not compare pairwise-model MSE or MAE directly with regression: pairwise
   latent scores are not calibrated to the rating scale. Use pair accuracy and
   rank correlations for those comparisons.

These results are sufficient to move from validation to the requested new
experiment write-up. No additional large GPU run is required before discussing
the findings with Dr. Yu.
