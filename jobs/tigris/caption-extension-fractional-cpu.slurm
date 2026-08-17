#!/bin/bash -l
#SBATCH --clusters=sporc
#SBATCH --partition=onboard
#SBATCH --account=loop
#SBATCH --qos=qos_tier3
#SBATCH --job-name=caption-frac-cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=FAIL,END
#SBATCH --output=/scratch/xx4455/paper-projects/logs/caption-frac-cpu-%A_%a.out
#SBATCH --error=/scratch/xx4455/paper-projects/logs/caption-frac-cpu-%A_%a.err

set -euo pipefail

spack env activate default-ml-x86_64-25050601
artifact_root=/scratch/xx4455/paper-projects/artifacts/comparative_image_caption

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        dataset=vicr
        output_name=vicr-vilbert-fractional-cpu
        ;;
    1)
        dataset=flickr_expert
        output_name=flickr-expert-vilbert-fractional-cpu
        ;;
    *)
        echo "Unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 1
        ;;
esac

cd "${SLURM_SUBMIT_DIR}"
python src/run_extension_experiments.py \
    --dataset "${dataset}" \
    --representation vilbert \
    --embedding-tag vilbert \
    --embeddings-dir "${SLURM_SUBMIT_DIR}/src/embeddings" \
    --output "${artifact_root}/results/${output_name}.csv" \
    --objectives hinge,bradley_terry \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --n-values 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --epochs 500 \
    --patience 15 \
    --skip-same-image-models \
    --verbose 0
