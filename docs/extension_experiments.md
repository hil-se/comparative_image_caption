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

VICR image URLs point to the COCO image host and are cached outside Git on
TIGRIS. The Flickr-Expert records use Flickr8k filenames but their original
`validatedicr.org` host no longer resolves. The matching Flickr8k files are
obtained from
https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip,
and the archive checksum is recorded on TIGRIS.

The generated image files, model cache, CLIP embeddings, logs, checkpoints,
and full result CSVs live beneath
`/home/xx4455/paper-projects/artifacts/comparative_image_caption` and are not
committed to Git.
