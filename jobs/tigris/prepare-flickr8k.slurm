#!/bin/bash -l
#SBATCH --clusters=tigris
#SBATCH --partition=tigris
#SBATCH --account=rc-onboard
#SBATCH --job-name=caption-prepare-flickr8k
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xx4455/paper-projects/logs/caption-prepare-flickr8k-%j.out
#SBATCH --error=/home/xx4455/paper-projects/logs/caption-prepare-flickr8k-%j.err

set -euo pipefail

artifact_root=/home/xx4455/paper-projects/artifacts/comparative_image_caption
archive_dir="${artifact_root}/source_archives"
dataset_dir="${artifact_root}/datasets/flickr8k"
archive_path="${archive_dir}/flickr8k.zip"
source_url=https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip

mkdir -p "${archive_dir}" "${dataset_dir}"
if [[ ! -f "${archive_path}" ]]; then
    /usr/bin/curl --fail --location --retry 5 --retry-all-errors \
        --output "${archive_path}" "${source_url}"
fi

echo "$(sha256sum "${archive_path}")" > "${archive_path}.sha256"
if [[ ! -f "${dataset_dir}/.complete" ]]; then
    unzip -q "${archive_path}" -d "${dataset_dir}"
    touch "${dataset_dir}/.complete"
fi

image_count=$(find "${dataset_dir}" -type f -iname '*.jpg' | wc -l)
echo "Prepared ${image_count} Flickr8k images beneath ${dataset_dir}"
if (( image_count < 8000 )); then
    echo "Expected at least 8000 Flickr8k images" >&2
    exit 1
fi
