# CLIP, Bradley-Terry, and Flickr-Expert extensions

These experiments answer the three extension questions requested by Dr. Yu:

1. Compare CLIP image/text representations with the released ViLBERT features.
2. Compare Bradley-Terry logistic loss with the released hinge objective.
3. Run the complete controlled model suite on Flickr-Expert.

## Controls

- Preserve the official train, validation, and test partitions serialized in
  the repository.
- Fit standardization on training features only.
- Use the same seeds and sampled comparison pairs for hinge and
  Bradley-Terry models.
- Use one orientation of each unordered pair and never move images between
  splits.
- Report Pearson, Spearman, Kendall tau-c, MSE, MAE, general pair accuracy,
  and same-image pair accuracy.
- Run ten seeds (0 through 9) and comparative sample sizes N=1 through N=5.

CLIP extraction uses `openai/clip-vit-base-patch32`, separately L2-normalizing
its 512-dimensional image and text projections before concatenation. The model
API is documented at https://huggingface.co/docs/transformers/model_doc/clip.

## Data provenance

VICR contains 8,990 COCO val2014 images and 1,000 Flickr8k images; the latter
use the same dead `validatedicr.org` URLs as Flickr-Expert. Both datasets
resolve images from local, external archives on TIGRIS. COCO val2014 comes
from `http://images.cocodataset.org/zips/val2014.zip`. The Flickr8k files are
obtained from
https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip,
and the archive checksum is recorded on TIGRIS.

The generated image files, model cache, CLIP embeddings, logs, checkpoints,
and full result CSVs live beneath
`/home/xx4455/paper-projects/artifacts/comparative_image_caption` and are not
committed to Git.

The completed aggregate analysis and interpretation are recorded in
[`extension_results.md`](extension_results.md).
