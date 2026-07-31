#!/bin/bash -l
#SBATCH --clusters=tigris
#SBATCH --partition=tigris
#SBATCH --account=rc-onboard
#SBATCH --job-name=caption-extension-full
#SBATCH --array=0-3
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xx4455/paper-projects/logs/caption-extension-full-%A_%a.out
#SBATCH --error=/home/xx4455/paper-projects/logs/caption-extension-full-%A_%a.err

set -euo pipefail

spack env activate default-ml-aarch64-25050701
artifact_root=/home/xx4455/paper-projects/artifacts/comparative_image_caption

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        dataset=vicr
        representation=vilbert
        embedding_tag=vilbert
        embeddings_dir="${SLURM_SUBMIT_DIR}/src/embeddings"
        output_name=vicr-vilbert
        ;;
    1)
        dataset=vicr
        representation=clip-vit-b32
        embedding_tag=openai-clip-vit-base-patch32
        embeddings_dir="${artifact_root}/embeddings/clip-vit-b32"
        output_name=vicr-clip-vit-b32
        ;;
    2)
        dataset=flickr_expert
        representation=vilbert
        embedding_tag=vilbert
        embeddings_dir="${SLURM_SUBMIT_DIR}/src/embeddings"
        output_name=flickr-expert-vilbert
        ;;
    3)
        dataset=flickr_expert
        representation=clip-vit-b32
        embedding_tag=openai-clip-vit-base-patch32
        embeddings_dir="${artifact_root}/embeddings/clip-vit-b32"
        output_name=flickr-expert-clip-vit-b32
        ;;
    *)
        echo "Unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 1
        ;;
esac

cd "${SLURM_SUBMIT_DIR}"
python src/run_extension_experiments.py \
    --dataset "${dataset}" \
    --representation "${representation}" \
    --embedding-tag "${embedding_tag}" \
    --embeddings-dir "${embeddings_dir}" \
    --output "${artifact_root}/results/${output_name}.csv" \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --n-values 1,2,3,4,5 \
    --epochs 500 \
    --patience 15 \
    --verbose 0
