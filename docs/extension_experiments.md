# Controlled image--caption extension protocol

The extension compares fixed ViLBERT and CLIP ViT-B/32 representations,
hinge and Bradley--Terry pairwise objectives, and VICR and Flickr-Expert under
one controlled evaluation protocol.

## Controls

- Preserve the official image-disjoint train, validation, and test partitions.
- Round each image--caption pair's mean human rating to the nearest integer on
  the 1--5 scale; use this value for regression and derived preferences.
- Omit preference ties after rounding.
- Fit feature standardization on the training partition only.
- Use the same `1024 -> 64 -> 1` ReLU/dropout-0.2 scorer for each fixed
  representation.
- Use matched seeds and sampled pairs for hinge and Bradley--Terry.
- Define `N = M / n_train`, where `M` is the sampled training-pair count.
- Sweep `N = 0.1, 0.2, ..., 1.0, 2, 3, 4, 5` with ten seeds.
- Generate one orientation per unordered pair and keep images within their
  official partitions for same-image evaluation.
- Report Pearson, Spearman, Kendall tau-c, MSE, MAE, general pair accuracy,
  and same-image pair accuracy.

CLIP extraction uses `openai/clip-vit-base-patch32`; its 512-dimensional image
and text projections are separately L2-normalized and concatenated. ViLBERT
remains the primary joint cross-modal representation.

The authoritative runner is `src/run_extension_experiments.py`. Final outputs
and SHA-256 metadata are in `results/extensions/controlled/`.

## Data provenance

VICR and Flickr-Expert use the official manifests under `manifests/`. Image
archives, model caches, and generated embeddings are external artifacts and
are not committed. The manifests and result metadata provide stable source
hashes for the controlled analysis.
