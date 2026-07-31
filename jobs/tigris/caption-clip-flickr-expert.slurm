#!/bin/bash -l
#SBATCH --clusters=tigris
#SBATCH --partition=tigris
#SBATCH --account=rc-onboard
#SBATCH --job-name=caption-clip-flickr-expert
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xx4455/paper-projects/logs/caption-clip-flickr-expert-%j.out
#SBATCH --error=/home/xx4455/paper-projects/logs/caption-clip-flickr-expert-%j.err

set -euo pipefail

spack env activate default-ml-aarch64-25050701
artifact_root=/home/xx4455/paper-projects/artifacts/comparative_image_caption
export HF_HOME="${artifact_root}/huggingface"

cd "${SLURM_SUBMIT_DIR}"
python src/clip_embeddings.py \
    --dataset flickr_expert \
    --source-dir src/embeddings \
    --output-dir "${artifact_root}/embeddings/clip-vit-b32" \
    --image-cache "${artifact_root}/datasets/flickr-cache" \
    --flickr8k-root "${artifact_root}/datasets/flickr8k" \
    --model-name openai/clip-vit-base-patch32 \
    --revision 8092f5b35a22023f7a822152e20837ac59cb91a3 \
    --batch-size 256
