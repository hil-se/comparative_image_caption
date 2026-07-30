#!/bin/bash -l
#SBATCH --clusters=tigris
#SBATCH --partition=tigris
#SBATCH --account=rc-onboard
#SBATCH --job-name=caption-same-image-ablation
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xx4455/paper-projects/logs/caption-same-image-ablation-%j.out
#SBATCH --error=/home/xx4455/paper-projects/logs/caption-same-image-ablation-%j.err

set -euo pipefail

spack env activate default-ml-aarch64-25050701
cd "${SLURM_SUBMIT_DIR}/src"

python compare_same_image_protocol_ablation.py --verbose 2
