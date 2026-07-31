#!/bin/bash -l
#SBATCH --clusters=tigris
#SBATCH --partition=tigris
#SBATCH --account=rc-onboard
#SBATCH --job-name=caption-extension-smoke
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xx4455/paper-projects/logs/caption-extension-smoke-%j.out
#SBATCH --error=/home/xx4455/paper-projects/logs/caption-extension-smoke-%j.err

set -euo pipefail

spack env activate default-ml-aarch64-25050701
artifact_root=/home/xx4455/paper-projects/artifacts/comparative_image_caption

cd "${SLURM_SUBMIT_DIR}"
python src/run_extension_experiments.py \
    --dataset flickr_expert \
    --representation clip-vit-b32 \
    --embedding-tag openai-clip-vit-base-patch32 \
    --embeddings-dir "${artifact_root}/embeddings/clip-vit-b32" \
    --output "${artifact_root}/results/flickr-expert-clip-smoke.csv" \
    --seeds 0 \
    --n-values 1 \
    --epochs 2 \
    --patience 2 \
    --verbose 2 \
    --overwrite
