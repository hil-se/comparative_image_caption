# Modeling Image--Caption Rating from Comparative Judgments

This repository contains the replication code and controlled results for the
image--caption paper. The primary model uses fixed ViLBERT joint cross-modal
representations; CLIP ViT-B/32 is included as a controlled regression
representation baseline.

## Controlled protocol

- Datasets: VICR and Flickr-Expert.
- Splits: official train/validation/test partitions, with every image and all
  of its captions confined to one partition.
- Targets: the mean human rating is rounded to the nearest integer on the
  1--5 scale. The rounded mean is used for regression and to derive pairwise
  preferences; ties after rounding are omitted.
- Features: fixed pretrained representations, standardized using the training
  partition only.
- Scorer: `1024 -> 64 -> 1`, ReLU activations, and dropout 0.2.
- Objectives: mean-squared-error regression, hinge ranking, and
  Bradley--Terry logistic loss.
- Pair budget: `N = M / n_train`, where `M` is the number of sampled training
  pairs. The tested grid is `N = 0.1, 0.2, ..., 1.0, 2, 3, 4, 5`.
- Repetitions: ten matched seeds (`0` through `9`). Hinge and Bradley--Terry
  use the same sampled pairs for each seed and budget.
- Optimization: Adam, learning rate `1e-3`, batch size 256, at most 500
  epochs, validation-loss early stopping with patience 15, and best-weight
  restoration.
- Same-image evaluation: official image-disjoint splits and one orientation
  per unordered caption pair, preventing reversed-pair leakage.

`src/run_extension_experiments.py` is the authoritative controlled runner.
The original `regression.py`, `compare.py`, and `compare_same_image.py`
entrypoints are retained only to document the released workflow; they do not
implement all controls used for the final paper.

## Repository layout

```text
comparative_image_caption/
├── src/
│   ├── run_extension_experiments.py   # authoritative controlled runner
│   ├── build_split_manifests.py       # official split manifests
│   ├── clip_embeddings.py             # CLIP feature extraction
│   └── embeddings/                    # serialized fixed representations
├── manifests/                         # official split manifests and summaries
├── results/extensions/controlled/     # final CSVs and metadata
├── code/                              # human-study analysis
├── results/human_subject/             # human-study outputs
├── jobs/tigris/                       # Slurm launch scripts
└── docs/                              # protocol and result notes
```

## Reproducing the controlled runs

Install TensorFlow, NumPy, pandas, and SciPy in the target environment. From
the repository root, run one command per dataset and representation. For
example, the complete ViLBERT/VICR grid is:

```bash
python src/run_extension_experiments.py \
  --dataset vicr \
  --representation vilbert \
  --embedding-tag vilbert \
  --embeddings-dir src/embeddings \
  --output results/extensions/controlled/vicr-vilbert-all.csv \
  --rating-mode rounded_mean \
  --objectives regression,hinge,bradley_terry \
  --n-values 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5 \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

Use `--dataset flickr_expert` for Flickr-Expert. CLIP runs use
`--representation clip-vit-b32` and
`--embedding-tag openai-clip-vit-base-patch32` after extracting the matching
features. Each output CSV receives a metadata sidecar containing source and
result SHA-256 digests.

## Main results

Regression representation comparison (mean +/- sample standard deviation
over ten seeds):

| Dataset | Representation | Pearson | Spearman | Kendall tau-c |
|---|---|---:|---:|---:|
| Flickr-Expert | ViLBERT | 0.7657 +/- 0.0090 | 0.6624 +/- 0.0172 | 0.5366 +/- 0.0144 |
| Flickr-Expert | CLIP ViT-B/32 | 0.6583 +/- 0.0124 | 0.5772 +/- 0.0175 | 0.4601 +/- 0.0147 |
| VICR | ViLBERT | 0.9035 +/- 0.0024 | 0.8844 +/- 0.0025 | 0.8059 +/- 0.0039 |
| VICR | CLIP ViT-B/32 | 0.8372 +/- 0.0045 | 0.8342 +/- 0.0036 | 0.7389 +/- 0.0044 |

ViLBERT different-image pairwise results at `N=1`:

| Dataset | Objective | Pearson | Spearman | Kendall tau-c |
|---|---|---:|---:|---:|
| Flickr-Expert | Hinge | 0.7278 | 0.6626 | 0.5364 |
| Flickr-Expert | Bradley--Terry | 0.7333 | 0.6684 | 0.5414 |
| VICR | Hinge | 0.8652 | 0.8783 | 0.7969 |
| VICR | Bradley--Terry | 0.8757 | 0.8786 | 0.7975 |

All six paired metric tests at `N=1` are nonsignificant after Holm correction.
The Bradley--Terry differences should therefore be described as numerical,
not as established improvements.

Corrected ViLBERT same-image-trained test accuracy:

| Dataset | Hinge | Bradley--Terry |
|---|---:|---:|
| Flickr-Expert | 0.8385 +/- 0.0102 | 0.8446 +/- 0.0053 |
| VICR | 0.8387 +/- 0.0063 | 0.8393 +/- 0.0088 |

The human study reports average between-rater agreement of 0.85/0.69
(`p_o`/kappa) for direct-rating-derived comparisons, 0.95/0.85 for
different-image comparisons, and 0.90/0.78 for same-image comparisons. These
are descriptive results from a small study.

## Result provenance

The final machine-readable outputs are under
`results/extensions/controlled/`. The two fractional-budget metadata files
are explicitly marked as reconstructed because the original host and Slurm
job identifiers were not retained; their result hashes and input hashes are
recorded without inventing missing provenance.

Supported by NSF Grant No. 2245796.
