#!/bin/bash -l
#SBATCH --clusters=tigris
#SBATCH --partition=tigris
#SBATCH --account=rc-onboard
#SBATCH --job-name=caption-prepare-coco-val2014
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xx4455/paper-projects/logs/caption-prepare-coco-val2014-%j.out
#SBATCH --error=/home/xx4455/paper-projects/logs/caption-prepare-coco-val2014-%j.err

set -euo pipefail

artifact_root=/home/xx4455/paper-projects/artifacts/comparative_image_caption
archive_dir="${artifact_root}/source_archives"
dataset_dir="${artifact_root}/datasets/coco2014"
archive_path="${archive_dir}/val2014.zip"
source_url=http://images.cocodataset.org/zips/val2014.zip

mkdir -p "${archive_dir}" "${dataset_dir}"
/usr/bin/curl --fail --location --retry 5 --retry-all-errors \
    --continue-at - --output "${archive_path}" "${source_url}"

sha256sum "${archive_path}" > "${archive_path}.sha256"
unzip -tq "${archive_path}"
if [[ ! -f "${dataset_dir}/.complete" ]]; then
    unzip -q -n "${archive_path}" -d "${dataset_dir}"
    touch "${dataset_dir}/.complete"
fi

image_count=$(find "${dataset_dir}/val2014" -type f -iname '*.jpg' | wc -l)
echo "Prepared ${image_count} COCO val2014 images beneath ${dataset_dir}/val2014"
if (( image_count != 40504 )); then
    echo "Expected exactly 40504 COCO val2014 images" >&2
    exit 1
fi
